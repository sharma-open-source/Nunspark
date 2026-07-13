import pytest

from nunspark.packer import pack


@pytest.fixture
def tiny_packed_dir(tiny_model_dir_with_tokenizer, tmp_path):
    """A tiny packed NunSpark model (with tokenizer), ready to stream."""
    out = tmp_path / "tiny-packed"
    pack(tiny_model_dir_with_tokenizer, out)
    return out
