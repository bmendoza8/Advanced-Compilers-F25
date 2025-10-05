import pytest
from cfg_utils import (
    get_path_lengths,
    reverse_postorder,
    find_back_edges,
    is_reducible,
)
def test_get_path_lengths():
    cfg = {"entry": ["A"], "A": ["B"], "B": ["C"], "C": []}
    assert get_path_lengths(cfg, "entry") == {"entry": 0, "A": 1, "B": 2, "C": 3}

def test_reverse_postorder():
    cfg = {"entry": ["A"], "A": ["B"], "B": []}
    assert reverse_postorder(cfg, "entry") == ["entry", "A", "B"]

def test_find_back_edges():
    cfg = {"A": ["B"], "B": ["A"]}
    back_edges = find_back_edges(cfg, "A")
    assert ("B", "A") in back_edges

def test_is_reducible():
    cfg = {"A": ["B"], "B": ["A"]}
    assert is_reducible(cfg, "A") is True
