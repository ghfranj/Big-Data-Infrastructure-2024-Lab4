import torch
from torch.utils.data import Dataset, DataLoader


class TwitterDataset(Dataset):
    def __init__(self, x_data, y_data):
        inputs = x_data
        outputs = y_data
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        self.errors = 0

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, idx):
        inputs = self.inputs[idx].toarray()
        outputs = self.outputs[idx]
        try:
            return {
                'input': torch.tensor(inputs).float(),
                'output': torch.tensor(int(outputs)).float()
            }
        except :
            if self.errors < 5:
                self.errors = self.errors + 1
                return self.__getitem__((idx+1) % self.__len__())
            else:
                return {
                    'input': torch.tensor(inputs).float(),
                    'output': torch.tensor(int(outputs)).float()
                }

