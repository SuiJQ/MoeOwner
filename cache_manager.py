"""
cache_manager.py — Hybrid Radix-Tree / Block-Based KV Cache Manager

Provides:
  - Block: a single KV cache block descriptor.
  - HybridCache: manages block allocation, prefix matching via radix index,
    reference counting, garbage collection, and GPU-memory-aware sizing.
"""

from __future__ import annotations

import hashlib
import logging
from array import array

logger = logging.getLogger(__name__)


class Block:
    """A single KV-cache block descriptor.

    Uses ``__slots__`` with ``hash_chain`` stored as ``array('i')``
    for compact memory layout and fast integer iteration.
    """

    __slots__ = ("block_id", "hash_chain", "kv_tensor", "ref_count")

    def __init__(
        self,
        block_id: int,
        token_ids: list[int],
        kv_tensor: object = None,
        ref_count: int = 1,
    ) -> None:
        self.block_id = block_id
        self.hash_chain: array = array("i", token_ids)
        self.kv_tensor = kv_tensor
        self.ref_count = ref_count

    def __repr__(self) -> str:
        return f"Block(id={self.block_id}, chain_len={len(self.hash_chain)}, refs={self.ref_count})"


class HybridCache:
    """
    A hybrid radix-tree / block-based KV cache.

    Features:
      - Incremental SHA-256 token hashing.
      - GPU-memory-aware total_blocks calculation.
      - Free-block queue for O(1) allocation.
      - Radix index for longest-prefix matching with LRU cache.
      - Reference-count-based GC.
      - Hash-tree prefetch on leaf cache hit.

    [Bug 2] Radix index uses ``dict[str, set[int]]`` so multiple blocks
    sharing the same prefix hash are all tracked — no 1-to-1 overwrite.
    """

    _PREFETCH_FANOUT_MAX: int = 3
    _MEM_FRACTION: float = 0.50

    def __init__(
        self,
        block_size: int = 16,
        hidden_size: int = 4096,
        total_blocks: int | None = None,
        mem_fraction: float | None = None,
    ) -> None:
        self.block_size = block_size
        self.hidden_size = hidden_size

        if mem_fraction is not None:
            self._MEM_FRACTION = mem_fraction

        if total_blocks is not None:
            self.total_blocks = total_blocks
        else:
            self.total_blocks = self._compute_total_blocks(
                block_size, hidden_size, self._MEM_FRACTION
            )

        self._slot_bytes = self.block_size * self.hidden_size * 2 * 2

        self.free_block_queue: list[int] = list(range(self.total_blocks))
        self.allocated_blocks: dict[int, Block] = {}
        # [Bug 2] set[int] instead of int: multiple blocks can share a prefix.
        self.radix_index: dict[str, set[int]] = {}

        self._hash_tree: dict[str, list[str]] = {}

        # [Bug 9] LRU cache keyed on tuple of first 8 tokens, not hash().
        self._match_cache: dict[tuple, tuple[int | None, int]] = {}
        self._match_cache_lru: list[tuple] = []
        self._match_cache_max = 256

        self._prefetch_stream: object | None = None

        logger.info(
            "HybridCache initialized: block_size=%d, total_blocks=%d, mem_frac=%.2f",
            self.block_size,
            self.total_blocks,
            self._MEM_FRACTION,
        )

    # ------------------------------------------------------------------
    # Memory-aware sizing
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_total_blocks(
        block_size: int = 16, hidden_size: int = 4096, mem_fraction: float = 0.50
    ) -> int:
        try:
            import torch  # noqa: PLC0415

            free_mem, _ = torch.cuda.mem_get_info()
            slot_bytes = block_size * hidden_size * 2 * 2
            total = int((free_mem * mem_fraction) / slot_bytes)
            logger.info("GPU free mem: %d bytes -> total_blocks=%d", free_mem, total)
            return max(total, 1)
        except (ImportError, RuntimeError, AttributeError):
            logger.warning("torch.cuda unavailable; using default total_blocks=10000")
            return 10_000

    def get_gpu_memory_slot(self, block_id: int) -> int:
        return block_id * self._slot_bytes

    # ------------------------------------------------------------------
    # Incremental hash
    # ------------------------------------------------------------------

    @staticmethod
    def _incremental_hash(previous_hex: str, token: int) -> str:
        prev_bytes = bytes.fromhex(previous_hex)
        prev_digest = hashlib.sha256(prev_bytes).digest()
        token_bytes = token.to_bytes(4, "little", signed=True)
        return hashlib.sha256(prev_digest + token_bytes).hexdigest()

    @staticmethod
    def _hash_single_token(token: int) -> str:
        token_bytes = token.to_bytes(4, "little", signed=True)
        return hashlib.sha256(token_bytes).hexdigest()

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def allocate(self, prompt_tokens: list[int]) -> Block:
        """
        Allocate a new block for the given prompt tokens.

        [Bug 2] Uses set[int] in radix_index so shared prefixes don't collide.
        """
        if not prompt_tokens:
            raise ValueError("allocate() requires at least one token")

        free = self.free_block_queue
        blocks = self.allocated_blocks
        radix = self.radix_index
        tree = self._hash_tree

        if not free:
            raise RuntimeError("No free blocks available in HybridCache")

        block_id = free.pop()

        new_block = Block(
            block_id=block_id,
            token_ids=prompt_tokens,
            kv_tensor=None,
            ref_count=1,
        )
        blocks[block_id] = new_block

        cumulative = self._hash_single_token(prompt_tokens[0])
        prev_hash: str = cumulative
        radix.setdefault(cumulative, set()).add(block_id)
        tree.setdefault(cumulative, [])

        for token in prompt_tokens[1:]:
            cumulative = self._incremental_hash(cumulative, token)
            radix.setdefault(cumulative, set()).add(block_id)
            tree.setdefault(prev_hash, []).append(cumulative)
            prev_hash = cumulative

        logger.debug(
            "Allocated block %d (tokens=%s...)",
            block_id,
            str(prompt_tokens[:4])[:-1],
        )
        return new_block

    # ------------------------------------------------------------------
    # Prefix matching
    # ------------------------------------------------------------------

    def match_prefix(self, prompt_tokens: list[int]) -> tuple[int | None, list[int]]:
        """
        Find the longest prefix of ``prompt_tokens`` that exists in the cache.

        [Bug 2] Tracks candidate block sets through each step via intersection,
        so only consistent prefix paths are returned.  When no set is consistent
        with the prompt, returns the last consistent prefix.
        """
        blocks = self.allocated_blocks
        radix = self.radix_index
        match_cache = self._match_cache
        match_lru = self._match_cache_lru

        # [Bug 9] Use tuple of first 8 tokens directly as cache key.
        cache_key = tuple(prompt_tokens[:8])
        if cache_key in match_cache:
            block_id, matched_len = match_cache[cache_key]
            match_lru.remove(cache_key)
            match_lru.append(cache_key)
            if block_id is not None:
                block = blocks.get(block_id)
                if block is not None:
                    block.ref_count += 1
                if block is None:
                    # Stale cache entry — block was freed; invalidate and fall
                    # through to the fresh lookup path below.
                    match_cache.pop(cache_key, None)
                    match_lru.remove(cache_key)
                else:
                    if matched_len == len(prompt_tokens):
                        self._try_prefetch_children(block)
                    return block_id, prompt_tokens[matched_len:]
            return None, prompt_tokens

        cumulative_hash: str = ""
        last_matched_block_id: int | None = None
        split_index: int = 0
        prev_candidates: set[int] | None = None

        for i, token in enumerate(prompt_tokens):
            if i == 0:
                cumulative_hash = self._hash_single_token(token)
            else:
                cumulative_hash = self._incremental_hash(cumulative_hash, token)

            step_candidates = radix.get(cumulative_hash, set())
            if not step_candidates:
                break

            if prev_candidates is not None:
                consistent = step_candidates & prev_candidates
            else:
                consistent = step_candidates

            if not consistent:
                break

            last_matched_block_id = next(iter(consistent))
            split_index = i + 1
            prev_candidates = consistent

        result = (last_matched_block_id, split_index)
        match_cache[cache_key] = result
        match_lru.append(cache_key)
        if len(match_lru) > self._match_cache_max:
            old_key = match_lru.pop(0)
            match_cache.pop(old_key, None)

        if last_matched_block_id is not None:
            block = blocks.get(last_matched_block_id)
            if block is not None:
                block.ref_count += 1
            if split_index == len(prompt_tokens) and block is not None:
                self._try_prefetch_children(block)
            return last_matched_block_id, prompt_tokens[split_index:]
        else:
            return None, prompt_tokens

    # ------------------------------------------------------------------
    # [Step 7] Hash-tree prefetch
    # ------------------------------------------------------------------

    def _try_prefetch_children(self, block: Block) -> None:
        try:
            import torch  # noqa: PLC0415

            chain = block.hash_chain
            if len(chain) == 0:
                return
            leaf_hash = self._hash_single_token(chain[0])
            for tok in chain[1:]:
                leaf_hash = self._incremental_hash(leaf_hash, tok)

            children = self._hash_tree.get(leaf_hash, [])
            if not children:
                return

            children = children[: self._PREFETCH_FANOUT_MAX]
            radix = self.radix_index
            allocated = self.allocated_blocks

            total_prefetch = len(children) * self._slot_bytes
            free_mem, _ = torch.cuda.mem_get_info()
            if free_mem < total_prefetch * 2:
                return

            if self._prefetch_stream is None:
                self._prefetch_stream = torch.cuda.Stream(priority=-1)

            with torch.cuda.stream(self._prefetch_stream):
                for child_hash in children:
                    cids = radix.get(child_hash, set())
                    for cid in cids:
                        child_block = allocated.get(cid)
                        if child_block is None:
                            continue
                        kv = child_block.kv_tensor
                        if kv is None:
                            continue
                        if kv.device.type != "cuda":
                            _ = kv.to(device="cuda", non_blocking=True)
        except RuntimeError as _rexc:
            logger.warning("_try_prefetch_children RuntimeError: %s", _rexc)
        except Exception as _eexc:
            logger.warning("_try_prefetch_children unexpected error: %s", _eexc)

    # ------------------------------------------------------------------
    # Free / reference-count management
    # ------------------------------------------------------------------

    def free_block(self, block_id: int) -> None:
        """
        Decrease the reference count of a block.

        [Bug 2] Removes block_id from radix_index sets rather than deleting
        straight dict entries.
        """
        free = self.free_block_queue
        blocks = self.allocated_blocks
        radix = self.radix_index

        block = blocks.get(block_id)
        if block is None:
            logger.warning("free_block: block %d not found", block_id)
            return

        block.ref_count -= 1
        logger.debug("free_block %d: ref_count now %d", block_id, block.ref_count)

        if block.ref_count <= 0:
            free.append(block_id)
            blocks.pop(block_id, None)
            stale = [h for h, bids in radix.items() if block_id in bids]
            for h in stale:
                radix[h].discard(block_id)
                if not radix[h]:
                    del radix[h]
            logger.debug(
                "Block %d recycled to free queue, removed %d radix entries",
                block_id,
                len(stale),
            )

    # ------------------------------------------------------------------
    # Garbage collection
    # ------------------------------------------------------------------

    def gc(self) -> int:
        """[Bug 2] Iterates set values and prunes stale block_ids per hash."""
        removed = 0
        stale_keys: list[str] = []
        blocks = self.allocated_blocks
        radix = self.radix_index

        for h, bids in radix.items():
            stale = [bid for bid in bids if bid not in blocks]
            if stale:
                for bid in stale:
                    bids.discard(bid)
                    removed += 1
                if not bids:
                    stale_keys.append(h)

        for key in stale_keys:
            del radix[key]

        if removed:
            logger.info("GC removed %d stale radix entries", removed)
        return removed

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def used_blocks(self) -> int:
        return len(self.allocated_blocks)

    @property
    def free_blocks(self) -> int:
        return len(self.free_block_queue)

    def stats(self) -> dict[str, int]:
        return {
            "total_blocks": self.total_blocks,
            "used_blocks": self.used_blocks,
            "free_blocks": self.free_blocks,
            "radix_entries": len(self.radix_index),
        }
