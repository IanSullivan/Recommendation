from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class NCFDataSet(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.data_len = len(self.data)
        self.customers = np.asarray(self.data.iloc[:, 1])
        self.items = np.asarray(self.data.iloc[:, 2])
        self.prices = np.asarray(self.data.iloc[:, 3])
        self.labels = np.asarray(self.data.iloc[:, 4])

    def __getitem__(self, index):
        return self.customers[index], self.items[index], self.prices[index], self.labels[index]

    def __len__(self):
        return self.data_len



