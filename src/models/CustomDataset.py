import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, train_data):
        self.data = torch.from_numpy(train_data).to(torch.float32)

    def __getitem__(self, index):
        x = self.data[index]
        return x

    def __len__(self):
        return len(self.data)
