# tests/test_tree_shape.py
import pytest
from nunspark.tree_shape import TreeShape


def test_chain_shape():
    # branching [1,1,1] -> a linear chain: 4 nodes, 1 leaf, 1 path of length 4.
    s = TreeShape([1, 1, 1])
    assert s.num_nodes == 4
    assert s.num_paths == 1
    assert s.path_len == 4
    assert s.paths == [[0, 1, 2, 3]]
    assert s.parent[3] == 2 and s.parent[0] == -1
    assert s.depth[3] == 3


def test_branching_shape_counts():
    # [2,2] -> root(0) + 2 + 4 = 7 nodes, 4 leaves, paths of length 3.
    s = TreeShape([2, 2])
    assert s.num_nodes == 7
    assert s.num_paths == 4
    assert s.path_len == 3
    # root has 2 children, each child has 2 children
    assert len(s.children[0]) == 2
    assert all(len(s.children[c]) == 2 for c in s.children[0])
    # every path starts at root and ends at a distinct leaf
    assert all(p[0] == 0 for p in s.paths)
    assert sorted(p[-1] for p in s.paths) == s.leaves


def test_node_locator_points_into_paths():
    s = TreeShape([2, 2])
    for node in range(s.num_nodes):
        pi, dep = s.node_locator[node]
        assert s.paths[pi][dep] == node
        assert s.depth[node] == dep


def test_internal_nodes_are_non_leaves():
    s = TreeShape([3, 2])
    internal = set(s.internal_nodes)
    assert 0 in internal
    assert internal.isdisjoint(s.leaves)
    assert internal | set(s.leaves) == set(range(s.num_nodes))


def test_invalid_branching_rejected():
    with pytest.raises(ValueError):
        TreeShape([2, 0])
    with pytest.raises(ValueError):
        TreeShape([])
    with pytest.raises(ValueError):
        TreeShape([2, -1])
