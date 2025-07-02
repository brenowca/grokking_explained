import sys
import types
import random
import os

# Ensure repository root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Provide minimal torch stubs if torch is missing
try:
    import torch  # type: ignore
except ModuleNotFoundError:
    torch = types.ModuleType("torch")

    class FakeTensor:
        def __init__(self, data):
            self.data = list(data)
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            if isinstance(idx, slice):
                return FakeTensor(self.data[idx])
            return self.data[idx]
        def tolist(self):
            return list(self.data)
        def item(self):
            assert len(self.data) == 1
            return self.data[0]

    def arange(n):
        return FakeTensor(range(n))

    def randperm(n):
        arr = list(range(n))
        random.shuffle(arr)
        return FakeTensor(arr)

    def manual_seed(seed):
        random.seed(seed)

    torch.arange = arange
    torch.randperm = randperm
    torch.manual_seed = manual_seed
    torch.device = lambda x: x
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    class TensorDataset:
        def __init__(self, data):
            self.data = list(data)
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return (self.data[idx],)

    class Subset:
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = list(indices)
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, idx):
            return self.dataset[self.indices[idx]]

    utils = types.ModuleType("utils")
    data = types.ModuleType("utils.data")
    data.TensorDataset = TensorDataset
    data.Subset = Subset
    utils.data = data

    torch.utils = utils
    nn_mod = types.ModuleType("nn")
    class Module: pass
    class ModuleList(list): pass
    def Linear(*args, **kwargs):
        return None
    def ReLU(*args, **kwargs):
        return None
    nn_mod.Module = Module
    nn_mod.ModuleList = ModuleList
    nn_mod.Linear = Linear
    nn_mod.ReLU = ReLU
    torch.nn = nn_mod
    torch.optim = types.ModuleType("optim")
    
    sys.modules['torch'] = torch
    sys.modules['torch.utils'] = utils
    sys.modules['torch.utils.data'] = data
    sys.modules['torch.nn'] = nn_mod
    sys.modules['torch.optim'] = torch.optim

from torch.utils.data import TensorDataset
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
