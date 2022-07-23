from __future__ import annotations

import numpy as np
import pytest
import torch

from deepspeedcube.min_heap import MinHeap


test_size = torch.Size([3, 4])
test_dtype = torch.float16

def _get_test_array() -> torch.Tensor:
    # Use .to so the tests also work with int types
    return torch.randn(test_size).to(test_dtype)

def test_heap():
    """ Test basic heap functionality. """

    heap = MinHeap(test_size, test_dtype)
    assert len(heap) == 0

    test_arr = _get_test_array()
    heap.insert(torch.pi, test_arr)
    assert len(heap) == 1

    key_min, arr_min = heap.min()
    assert np.isclose(key_min, torch.pi)
    assert arr_min.dtype == test_dtype
    assert torch.all(arr_min == test_arr)

    key_min, arr_min = heap.extract_min()
    assert np.isclose(key_min, torch.pi)
    assert arr_min.dtype == test_dtype
    assert torch.all(arr_min == test_arr)
    assert len(heap) == 0

    with pytest.raises(ValueError):
        heap.extract_min()

def test_insert_order():
    """ Test that the heap is automatically expanded and the order is correct. """
    num = int(1.5 * MinHeap._init_heap_size)

    perm = torch.randperm(num)
    keys = torch.arange(num).float()[perm]
    data = torch.randn((num, *test_size)).to(test_dtype)[perm]

    heap = MinHeap(test_size, test_dtype)

    for i in range(num):
        heap.insert(keys[i], data[i])
        assert len(heap) == i + 1

    # Test that they are correctly ordered when extracting
    sort_idx = torch.argsort(keys)
    for i in range(num):
        key, dat = heap.extract_min()
        assert np.isclose(key, keys[sort_idx[i]])
        assert torch.all(dat == data[sort_idx[i]])
        assert len(heap) == num - i - 1


def test_multiple():
    """ Test that multiple inserts and extracts yield the same results. """
    num = int(1.5 * MinHeap._init_heap_size)

    perm = torch.randperm(num)
    keys = torch.arange(num).float()[perm]
    data = torch.randn((num, *test_size)).to(test_dtype)[perm]

    heap1 = MinHeap(test_size, test_dtype)
    heap2 = MinHeap(test_size, test_dtype)

    for i in range(num):
        heap1.insert(keys[i], data[i])
    heap2.insert_multiple(keys, data)

    assert len(heap1) == len(heap2)

    while len(heap2):
        n = min(len(heap1), 66)
        keys1 = torch.empty(n, dtype=torch.float)
        data1 = torch.empty((n, *test_size), dtype=test_dtype)
        for i in range(n):
            keys1[i], data1[i] = heap1.extract_min()

        keys2, data2 = heap2.extract_min_multiple(n)
        assert torch.all(keys1 == keys2)
        assert torch.all(data1 == data2)

        for key1, key2 in zip(keys2[:-1], keys2[1:]):
            assert key1 < key2
        assert len(keys2) <= n
    assert len(heap2) == 0
