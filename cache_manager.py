"""
cache_manager.py — Hybrid Radix-Tree / Block-Based KV Cache Manager

Provides:
  - Block: a single KV cache block descriptor.
  - HybridCache: manages block allocation, prefix matching via radix index,
    reference counting, garbage collection, and GPU-memory-aware sizing.

Design doc: see project README for the full architecture specification.
"""

from __future__ import annotations

import hashlib
import logging

logger = logging.getLogger(__name__)


class Block:
    """A single KV-cache block descriptor."""

    __slots__ = ("block_id", "next_block_id", "physical_address", "ref_count", "token_ids_hash")

    def __init__(
        self,
        block_id: int,
        physical_address: int,
        token_ids_hash: str,
        ref_count: int = 1,
        next_block_id: int | None = None,
    ) -> None:
        self.block_id = block_id
        self.physical_address = physical_address
        self.token_ids_hash = token_ids_hash
        self.ref_count = ref_count
        self.next_block_id = next_block_id

    def __repr__(self) -> str:
        return (
            f"Block(id={self.block_id}, phys={self.physical_address}, "
            f"hash={self.token_ids_hash[:12]}..., refs={self.ref_count}, "
            f"next={self.next_block_id})"
        )


class HybridCache:
    """
    A hybrid radix-tree / block-based KV cache.

    Features:
      - Incremental SHA-256 token hashing (NOT hashlib.update; uses digest chains).
      - GPU-memory-aware total_blocks calculation via torch.cuda.mem_get_info().
      - Free-block queue for O(1) allocation.
      - Radix index for longest-prefix matching.
      - Reference-count-based garbage collection.
    """

    def __init__(
        self,
        block_size: int = 16,
        hidden_size: int = 4096,
        total_blocks: int | None = None,
    ) -> None:
        """
        Args:
            block_size: Number of tokens per block.
            hidden_size: Hidden dimension size (used for memory accounting).
            total_blocks: If given, overrides the automatic GPU-memory-based count.
        """
        self.block_size = block_size
        self.hidden_size = hidden_size

        if total_blocks is not None:
            self.total_blocks = total_blocks
        else:
            self.total_blocks = self._compute_total_blocks(block_size, hidden_size)

        # Byte-equivalent size of one "slot" in GPU memory (float16 × 2 matrices × k/v).
        self._slot_bytes = self.block_size * self.hidden_size * 2 * 2

        # Free-block queue (LIFO for locality).
        self.free_block_queue: list[int] = list(range(self.total_blocks))

        # Live block descriptors.
        self.allocated_blocks: dict[int, Block] = {}

        # Radix index: cumulative-token-hash → block_id.
        self.radix_index: dict[str, int] = {}

        # LRU cache for match_prefix results.
        # Cache key: hash of first 8 tokens. Value: (block_id | None, split_index).
        self._match_cache: dict[int, tuple[int | None, int]] = {}
        self._match_cache_lru: list[int] = []
        self._match_cache_max = 256

        logger.info(
            "HybridCache initialized: block_size=%d, total_blocks=%d, slot_bytes=%d",
            self.block_size,
            self.total_blocks,
            self._slot_bytes,
        )

    # ------------------------------------------------------------------
    # Memory-aware sizing
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_total_blocks(block_size: int = 16, hidden_size: int = 4096) -> int:
        """
        Determine total_blocks dynamically from available GPU memory.

        Formula:
            total_blocks = int((free_mem * 0.85) / (block_size * hidden_size * 2 * 2))

        Falls back to 10 000 when torch/cuda is unavailable.

        Args:
            block_size: Number of tokens per block (used in memory formula).
            hidden_size: Hidden dimension of the model (used in memory formula).
        """
        try:
            import torch  # noqa: PLC0415

            free_mem, _ = torch.cuda.mem_get_info()  # bytes
            slot_bytes = block_size * hidden_size * 2 * 2
            total = int((free_mem * 0.85) / slot_bytes)
            logger.info("GPU free mem: %d bytes → total_blocks=%d", free_mem, total)
            return max(total, 1)
        except (ImportError, RuntimeError, AttributeError):
            logger.warning("torch.cuda unavailable; using default total_blocks=10000")
            return 10_000

    # ------------------------------------------------------------------
    # GPU memory slot (simplified)
    # ------------------------------------------------------------------

    def get_gpu_memory_slot(self, block_id: int) -> int:
        """
        Simplified offset calculation for the physical memory slot.

        Returns the byte offset into pre-allocated cache memory.
        """
        return block_id * self._slot_bytes

    # ------------------------------------------------------------------
    # Incremental hash
    # ------------------------------------------------------------------

    @staticmethod
    def _incremental_hash(previous_hex: str, token: int) -> str:
        """
        Compute the next cumulative hash.

        IMPORTANT: Uses SHA-256 chaining — NOT ``hashlib.update()``.
            hash_n = SHA256(SHA256(previous_hex_bytes).digest() + token_bytes).hexdigest()

        Args:
            previous_hex: The cumulative hex hash from the previous step.
            token: The next integer token ID.

        Returns:
            New cumulative hex hash.
        """
        prev_bytes = bytes.fromhex(previous_hex)
        prev_digest = hashlib.sha256(prev_bytes).digest()
        token_bytes = token.to_bytes(4, "little", signed=True)
        return hashlib.sha256(prev_digest + token_bytes).hexdigest()

    @staticmethod
    def _hash_single_token(token: int) -> str:
        """Initial hash for the very first token in a sequence."""
        token_bytes = token.to_bytes(4, "little", signed=True)
        return hashlib.sha256(token_bytes).hexdigest()

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def allocate(self, prompt_tokens: list[int]) -> Block:
        """
        Allocate a new block for the given prompt tokens.

        Pops a block_id from the free queue, builds the cumulative hash,
        registers ALL intermediate cumulative hashes in ``radix_index``
        (so that ``match_prefix`` can match any prefix), and stores the
        final hash in the Block descriptor.

        Raises
        ------
        RuntimeError
            If no free blocks remain.
        """
        if not self.free_block_queue:
            raise RuntimeError("No free blocks available in HybridCache")

        block_id = self.free_block_queue.pop()
        token_hash = self._incremental_hash_from_tokens(prompt_tokens)

        physical_address = self.get_gpu_memory_slot(block_id)
        new_block = Block(
            block_id=block_id,
            physical_address=physical_address,
            token_ids_hash=token_hash,
            ref_count=1,
            next_block_id=None,
        )
        self.allocated_blocks[block_id] = new_block

        # Register ALL intermediate cumulative hashes so match_prefix
        # can match any prefix of the block's token sequence.
        self._register_all_hashes(prompt_tokens, block_id)

        logger.debug(
            "Allocated block %d (final_hash=%s...)",
            block_id,
            token_hash[:12],
        )
        return new_block

    def _incremental_hash_from_tokens(self, tokens: list[int]) -> str:
        """Build the cumulative hex hash for a list of tokens."""
        if not tokens:
            raise ValueError("Cannot hash an empty token list")
        h = self._hash_single_token(tokens[0])
        for token in tokens[1:]:
            h = self._incremental_hash(h, token)
        return h

    def _register_all_hashes(self, tokens: list[int], block_id: int) -> None:
        """Register every intermediate cumulative hash in radix_index."""
        cumulative = self._hash_single_token(tokens[0])
        self.radix_index[cumulative] = block_id
        for token in tokens[1:]:
            cumulative = self._incremental_hash(cumulative, token)
            self.radix_index[cumulative] = block_id

    # ------------------------------------------------------------------
    # Prefix matching
    # ------------------------------------------------------------------

    def match_prefix(
        self, prompt_tokens: list[int]
    ) -> tuple[int | None, list[int]]:
        """
        Find the longest prefix of ``prompt_tokens`` that exists in the cache.

        Uses an LRU cache (keyed by hash of first 8 tokens) for O(1) hit
        performance on repeated lookups, plus fast token-by-token traversal
        with result caching on cache misses.

        Returns
        -------
        (matched_block_id, remaining_tokens)
            If a prefix was found, ``matched_block_id`` is the last (deepest)
            block id whose cumulative hash is in the radix index, and
            ``remaining_tokens`` are the unmatched suffix.
            If nothing matched, ``(None, prompt_tokens)`` is returned.
        """
        # Step 1: LRU cache lookup (keyed by first 8 tokens hash)
        cache_key = hash(tuple(prompt_tokens[:8]))
        if cache_key in self._match_cache:
            block_id, matched_len = self._match_cache[cache_key]
            # LRU hit promotion
            self._match_cache_lru.remove(cache_key)
            self._match_cache_lru.append(cache_key)
            if block_id is not None:
                self.allocated_blocks[block_id].ref_count += 1
                return block_id, prompt_tokens[matched_len:]
            return None, prompt_tokens

        # Step 2: Token-by-token traversal matching cumulative hashes
        cumulative_hash: str = ""
        last_matched_block_id: int | None = None
        split_index = 0

        for i, token in enumerate(prompt_tokens):
            if i == 0:
                cumulative_hash = self._hash_single_token(token)
            else:
                cumulative_hash = self._incremental_hash(cumulative_hash, token)

            if cumulative_hash in self.radix_index:
                last_matched_block_id = self.radix_index[cumulative_hash]
                split_index = i + 1
            else:
                break

        # Step 3: Update LRU cache
        result = (last_matched_block_id, split_index)
        self._match_cache[cache_key] = result
        self._match_cache_lru.append(cache_key)
        if len(self._match_cache_lru) > self._match_cache_max:
            old_key = self._match_cache_lru.pop(0)
            self._match_cache.pop(old_key, None)

        if last_matched_block_id is not None:
            # Bump ref count for the matched block.
            self.allocated_blocks[last_matched_block_id].ref_count += 1
            return last_matched_block_id, prompt_tokens[split_index:]
        else:
            return None, prompt_tokens

    # ------------------------------------------------------------------
    # Free / reference-count management
    # ------------------------------------------------------------------

    def free_block(self, block_id: int) -> None:
        """
        Decrease the reference count of a block.

        If ref_count reaches zero, the block is evicted: ``block_id`` is
        recycled to ``free_block_queue`` and the block is removed from
        ``allocated_blocks``.  Radix-index entries pointing to this block
        are cleaned up with a compound key guard.
        """
        block = self.allocated_blocks.get(block_id)
        if block is None:
            logger.warning("free_block: block %d not found", block_id)
            return

        block.ref_count -= 1
        logger.debug("free_block %d: ref_count now %d", block_id, block.ref_count)

        if block.ref_count <= 0:
            # Recycle the block id.
            self.free_block_queue.append(block_id)
            self.allocated_blocks.pop(block_id, None)
            # Clean up radix_index entries pointing to this block
            # (compound key guard: only delete if still points to us).
            stale = [
                h for h, bid in self.radix_index.items()
                if bid == block_id
            ]
            for h in stale:
                # Compound key guard: verify the entry still points to our
                # block_id before deleting (protects against reuse race).
                if self.radix_index.get(h) == block_id:
                    del self.radix_index[h]
            logger.debug(
                "Block %d recycled to free queue, removed %d radix entries",
                block_id,
                len(stale),
            )

    # ------------------------------------------------------------------
    # Garbage collection
    # ------------------------------------------------------------------

    def gc(self) -> int:
        """
        Garbage-collect stale entries from ``radix_index``.

        Removes entries whose target block is no longer alive (evicted
        from ``allocated_blocks``).  Uses a compound key guard so that a
        hash re-used by a newly allocated block is not prematurely
        deleted.

        Returns
        -------
        int
            Number of entries removed from ``radix_index``.
        """
        removed = 0
        stale_keys: list[str] = []

        for h, bid in self.radix_index.items():
            block = self.allocated_blocks.get(bid)
            if block is None:
                stale_keys.append(h)

        for key in stale_keys:
            # Compound key guard: only delete if the entry still points
            # to the same (now-dead) block_id as when we inspected it.
            bid = self.radix_index.get(key)
            if bid is not None and bid not in self.allocated_blocks:
                del self.radix_index[key]
                removed += 1

        if removed:
            logger.info("GC removed %d stale radix entries", removed)
        return removed

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def used_blocks(self) -> int:
        """Number of currently allocated blocks."""
        return len(self.allocated_blocks)

    @property
    def free_blocks(self) -> int:
        """Number of blocks in the free queue."""
        return len(self.free_block_queue)

    def stats(self) -> dict[str, int]:
        """Return a snapshot of usage statistics."""
        return {
            "total_blocks": self.total_blocks,
            "used_blocks": self.used_blocks,
            "free_blocks": self.free_blocks,
            "radix_entries": len(self.radix_index),
        }
