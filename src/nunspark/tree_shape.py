# src/nunspark/tree_shape.py
from __future__ import annotations


class TreeShape:
    """A fixed speculative-tree topology, numbered breadth-first from the root (0).

    `branching[k]` is the number of children every depth-`k` node has, so the tree
    has `len(branching)` levels of children and paths of length `len(branching)+1`.
    The leaves are all at maximum depth; `paths` lists every root-to-leaf path as a
    sequence of node ids (length `path_len`). `node_locator[node]` gives one
    `(path_index, depth)` that lands on `node`, used to read a node's logits back out
    of the batched `[num_paths, path_len, ...]` verify output (shared prefixes dedup).
    """

    def __init__(self, branching: list[int]):
        if not branching or any((not isinstance(c, int)) or c < 1 for c in branching):
            raise ValueError(f"branching must be a non-empty list of positive ints, got {branching!r}")
        self.branching = list(branching)

        self.parent: dict[int, int] = {0: -1}
        self.depth: dict[int, int] = {0: 0}
        self.children: dict[int, list[int]] = {0: []}

        frontier = [0]
        nid = 1
        for k, c in enumerate(branching):
            new_frontier: list[int] = []
            for p in frontier:
                for _ in range(c):
                    self.parent[nid] = p
                    self.depth[nid] = k + 1
                    self.children[nid] = []
                    self.children[p].append(nid)
                    new_frontier.append(nid)
                    nid += 1
            frontier = new_frontier

        self.num_nodes = nid
        self.leaves = list(frontier)                       # all at max depth
        self.path_len = len(branching) + 1
        self.paths = [self._path_to(leaf) for leaf in self.leaves]
        self.num_paths = len(self.paths)

        self.node_locator: dict[int, tuple[int, int]] = {}
        for pi, path in enumerate(self.paths):
            for dep, node in enumerate(path):
                # setdefault: first path wins, so every node (incl. shared prefixes)
                # gets a single stable (path_index, depth) slot. Consumers reading a
                # node's logits out of the [num_paths, path_len, ...] verify tensor
                # rely on this determinism.
                self.node_locator.setdefault(node, (pi, dep))

        leaf_set = set(self.leaves)
        self.internal_nodes = [n for n in range(self.num_nodes) if n not in leaf_set]

    def _path_to(self, leaf: int) -> list[int]:
        seq: list[int] = []
        n = leaf
        while n != -1:
            seq.append(n)
            n = self.parent[n]
        return list(reversed(seq))
