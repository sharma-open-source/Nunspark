# tests/test_build_draft_tree.py
import json

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.llama import Model, ModelArgs

from nunspark.tree_shape import TreeShape
from nunspark.tree_spec import build_draft_tree


def _draft(tiny_model_dir):
    config = json.loads((tiny_model_dir / "config.json").read_text())
    model = Model(ModelArgs.from_dict(config))
    model.load_weights(list(mx.load(str(tiny_model_dir / "model.safetensors")).items()))
    mx.eval(model.parameters())
    return model


def config_vocab(model_dir):
    return json.loads((model_dir / "config.json").read_text())["vocab_size"]


def test_fills_every_node_and_internal_logits(tiny_model_dir):
    draft = _draft(tiny_model_dir)
    shape = TreeShape([2, 2])
    committed = [3, 7, 42, 1]
    token, draft_logits = build_draft_tree(draft, committed, shape, temp=0.0)

    # every node gets a token; root token is the last committed token
    assert set(token) == set(range(shape.num_nodes))
    assert token[0] == committed[-1]
    # every internal node carries a draft distribution (its children's source)
    assert set(draft_logits) == set(shape.internal_nodes)
    assert all(draft_logits[n].shape == (config_vocab(tiny_model_dir),) for n in shape.internal_nodes)


def test_temp0_children_are_top_c(tiny_model_dir):
    # At temp=0 a node's children must be the top-c tokens of its draft distribution.
    draft = _draft(tiny_model_dir)
    shape = TreeShape([3])     # root + top-3 children
    committed = [3, 7, 42, 1]
    token, draft_logits = build_draft_tree(draft, committed, shape, temp=0.0)
    top3 = mx.argsort(-draft_logits[0])[:3].tolist()
    assert [token[c] for c in shape.children[0]] == top3


def test_temp_pos_fills_tree_with_valid_tokens(tiny_model_dir):
    # Exercise the temp>0 sampling branch end-to-end: every node filled, internal-node
    # draft logits present, all tokens within vocab. (Distribution correctness of the
    # downstream rejection sampling is covered in test_tree_sampling.py.)
    draft = _draft(tiny_model_dir)
    shape = TreeShape([3, 2])
    committed = [3, 7, 42, 1]
    vocab = config_vocab(tiny_model_dir)
    mx.random.seed(0)
    token, draft_logits = build_draft_tree(draft, committed, shape, temp=0.8)

    assert set(token) == set(range(shape.num_nodes))
    assert token[0] == committed[-1]
    assert set(draft_logits) == set(shape.internal_nodes)
    assert all(0 <= token[n] < vocab for n in range(shape.num_nodes))


def test_chain_temp0_matches_sequential_draft_argmax(tiny_model_dir):
    # A [1,1,1] tree at temp=0 is just greedy draft decoding; verify against a
    # plain sequential argmax rollout of the draft.
    draft = _draft(tiny_model_dir)
    committed = [3, 7, 42, 1]
    shape = TreeShape([1, 1, 1])
    token, _ = build_draft_tree(draft, committed, shape, temp=0.0)

    cache = make_prompt_cache(draft)
    logits = draft(mx.array(committed)[None], cache=cache)[:, -1, :]
    expected = []
    for _ in range(3):
        nxt = int(mx.argmax(logits, axis=-1).item())
        expected.append(nxt)
        logits = draft(mx.array([nxt])[None], cache=cache)[:, -1, :]

    path = shape.paths[0]                       # [0,1,2,3]
    assert [token[n] for n in path[1:]] == expected
