import sys
import os
import torch  # type: ignore
from torch.utils.data import TensorDataset

# Ensure repository root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import trainUtils as tu
tu.Subset = torch.utils.data.Subset


def test_value_to_list_scalar():
    assert tu.value_to_list(5) == [5]


def test_value_to_list_iterable():
    assert tu.value_to_list((1, 2)) == [1, 2]


def test_random_sample_subset_length():
    dataset = TensorDataset(torch.arange(10))
    subset = tu.random_sample(dataset, 3)
    assert len(subset) == 3


def test_random_sample_subset_contents():
    torch.manual_seed(0)
    dataset = TensorDataset(torch.arange(10))
    subset = tu.random_sample(dataset, 4)
    items = [subset[i][0] for i in range(len(subset))]
    assert len(set(items)) == 4
    assert all(0 <= x < 10 for x in items)
