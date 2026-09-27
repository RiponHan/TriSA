import torch
import torch.nn as nn
from torch.utils.data import Dataset


class PWDataset(Dataset):
    def __init__(self, Xs, Ys, Ts, api_nums) -> None:
        super().__init__()
        self.Xs = Xs # [N, B]
        self.Ys = Ys # [N, X] X 为调用的api的数量
        self.Ts = Ts
        self.api_nums = api_nums

    def __len__(self):
        return len(self.Xs)

    def __getitem__(self, index):
        x = self.Xs[index] # [B]
        y = self.Ys[index] # [X]
        y = nn.functional.one_hot(torch.LongTensor(y), num_classes=self.api_nums) # [C, N] N = api_nums
        y = y.sum(dim=0).float()
        t = self.Ts[index] / 30
        return torch.tensor(x), y, torch.tensor(t)