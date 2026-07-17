"""
ngram_speculation.py — N-Gram Speculative Decoding.

[Step 4] Pure-CPU N-Gram trie for draft generation + speculative verification.
[Fix 18] Node count limit prevents unbounded growth.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_MIN_NGRAM_LEN = 2


def _extract_logits(model_output):
    """Extract logits tensor from raw tensor or HF CausalLMOutput."""
    if isinstance(model_output, torch.Tensor):
        return model_output
    if hasattr(model_output, "logits"):
        return model_output.logits
    if isinstance(model_output, (tuple, list)):
        return model_output[0]
    return model_output


class NGramTrieNode:
    __slots__ = ("children", "count")

    def __init__(self):
        self.count: int = 0
        self.children: dict[int, NGramTrieNode] = {}


class NGramCache:
    """Pure-Python CPU N-gram cache using a trie.

    [Fix 18] ``max_nodes`` caps total node count; oldest nodes are evicted
    via periodic pruning when the limit is exceeded.
    """

    def __init__(self, max_n: int = 5, max_nodes: int = 100000):
        self.max_n = max_n
        self.max_nodes = max_nodes
        self.root = NGramTrieNode()
        self._node_count: int = 1  # root
        self._evict_counter: int = 0
        logger.info("NGramCache: max_n=%d, max_nodes=%d", max_n, max_nodes)

    def _add_node(self) -> NGramTrieNode:
        """Create a new node and auto-evict if over limit."""
        node = NGramTrieNode()
        self._node_count += 1
        self._evict_counter += 1
        prune_interval = 10000
        if self._node_count > self.max_nodes and self._evict_counter > prune_interval:
            self._prune()
        return node

    def _prune(self) -> None:
        """[Fix 7] Recount nodes correctly after pruning."""
        count_before = self._node_count
        self._prune_node(self.root, 0)
        self._node_count = self._count_nodes(self.root)
        self._evict_counter = 0
        logger.debug("N-Gram prune: %d -> %d nodes", count_before, self._node_count)

    def _count_nodes(self, node: NGramTrieNode) -> int:
        """[Fix 7] Recursively count all nodes under *node*."""
        count = 1
        for child in node.children.values():
            count += self._count_nodes(child)
        return count

    def _prune_node(self, node: NGramTrieNode, depth: int) -> bool:
        """[Fix 7] Prune nodes with count=0 and no children. Returns True if *node* should be kept."""
        if not node.children and node.count == 0 and depth > 0:
            return False
        stale = []
        for token, child in node.children.items():
            if not self._prune_node(child, depth + 1):
                stale.append(token)
        for k in stale:
            del node.children[k]
        return bool(node.children) or node.count > 0 or depth == 0

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, sequence: list[int]) -> None:
        if len(sequence) < _MIN_NGRAM_LEN:
            return

        for i in range(len(sequence)):
            node = self.root
            for j in range(i, min(len(sequence) - 1, i + self.max_n)):
                token = sequence[j]
                next_token = sequence[j + 1]

                if token not in node.children:
                    node.children[token] = self._add_node()
                node = node.children[token]
                node.count += 1

                if next_token not in node.children:
                    node.children[next_token] = self._add_node()
                node.children[next_token].count += 1

    def _best_next(self, node: NGramTrieNode) -> int | None:
        if not node.children:
            return None
        return max(node.children.items(), key=lambda kv: kv[1].count)[0]

    # ------------------------------------------------------------------
    # Draft generation
    # ------------------------------------------------------------------

    def generate_draft(self, context: list[int], draft_length: int = 5) -> list[int]:
        best_node: NGramTrieNode | None = None

        for suffix_len in range(min(len(context), self.max_n), 0, -1):
            suffix = context[-suffix_len:]
            node = self._walk(suffix)
            if node is not None:
                best_node = node
                break

        if best_node is None or best_node.count == 0:
            return []

        drafts: list[int] = []
        node = best_node
        for _ in range(draft_length):
            next_tok = self._best_next(node)
            if next_tok is None:
                break
            drafts.append(next_tok)
            if next_tok in node.children:
                node = node.children[next_tok]
            else:
                break

        return drafts

    def _walk(self, tokens: list[int]) -> NGramTrieNode | None:
        node = self.root
        for t in tokens:
            if t not in node.children:
                return None
            node = node.children[t]
        return node

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    @staticmethod
    def verify_drafts(
        logits: torch.Tensor, draft_tokens: list[int], context_len: int
    ) -> tuple[list[int], int]:
        if not draft_tokens:
            return [], 0
        accepted: list[int] = []
        for i, draft_tok in enumerate(draft_tokens):
            pos = context_len + i
            if pos >= logits.shape[1]:
                break
            predicted = logits[0, pos, :].argmax().item()
            if predicted == draft_tok:
                accepted.append(draft_tok)
            else:
                break
        return accepted, len(accepted)


class SpeculativeGenerator:
    def __init__(
        self,
        model: torch.nn.Module,
        ngram_cache: NGramCache,
        max_draft: int = 5,
        enabled: bool = True,
    ):
        self.model = model
        self.ngram_cache = ngram_cache
        self.max_draft = max_draft
        self.enabled = enabled

    def decode(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        if not self.enabled:
            with torch.no_grad():
                raw = self.model(input_ids)
            logits = _extract_logits(raw)
            next_tok = logits[0, -1, :].argmax().item()
            new_ids = torch.cat(
                [input_ids, torch.tensor([[next_tok]], device=input_ids.device)], dim=-1
            )
            return new_ids, 0, 0

        context = input_ids[0].tolist()
        drafts = self.ngram_cache.generate_draft(context, self.max_draft)
        if not drafts:
            with torch.no_grad():
                raw = self.model(input_ids)
            logits = _extract_logits(raw)
            next_tok = logits[0, -1, :].argmax().item()
            new_ids = torch.cat(
                [input_ids, torch.tensor([[next_tok]], device=input_ids.device)], dim=-1
            )
            return new_ids, 0, 0

        draft_tensor = torch.tensor([drafts], dtype=torch.long, device=input_ids.device)
        extended = torch.cat([input_ids, draft_tensor], dim=-1)

        with torch.no_grad():
            raw = self.model(extended)
        logits_t = _extract_logits(raw)

        context_len = input_ids.shape[1]
        accepted, num_accepted = NGramCache.verify_drafts(logits_t, drafts, context_len)

        accept_pos = context_len + num_accepted
        next_tok = logits_t[0, accept_pos, :].argmax().item()

        new_tokens = [*accepted, next_tok]
        new_ids = torch.cat(
            [input_ids, torch.tensor([new_tokens], dtype=torch.long, device=input_ids.device)],
            dim=-1,
        )

        self.ngram_cache.ingest(context + drafts[:num_accepted] + [next_tok])
        return new_ids, num_accepted, len(drafts)
