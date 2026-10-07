import pytest

from simplescale.cli import _pool, _tp


def test_submit_arguments():
    assert [_tp(str(value)) for value in (1, 2, 4, 8)] == [1, 2, 4, 8]
    assert _pool("h200_lowest=32") == ("h200_lowest", 32)
    with pytest.raises(Exception):
        _tp("3")
