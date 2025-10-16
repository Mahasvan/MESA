import math
from torch.utils.data import TensorDataset
from typing import Dict
import numpy as np

class CustomDataLoader:
    def __init__(self, dataset: TensorDataset, batch_size: int, gamma: float):
        self.batches = {}
        self.batch_losses = {}
        self.dataset = dataset
        self.batch_size = batch_size
        self.gamma = gamma
        self._make_batches()

    def __len__(self):
        return len(self.batches)

    def _make_batches(self):
        for i in range(0, len(self.dataset), self.batch_size):
            batch = self.dataset[i : i + self.batch_size]
            self.batches[i] = batch


    def update_losses(self, losses: Dict[int, float]):
        self.batch_losses.update(losses)

    def generate(self, dynamic: bool = False):
        # if its dynamic, it only returns 20% of the dataset (if gamma is 0.2)
        if dynamic and len(self.batch_losses) == len(self.batches):
            # print("Matched losses")
            order = np.argsort(list(self.batch_losses.values()))[::-1]
            count = math.ceil(len(order) * self.gamma)
            keys = list(self.batches.keys())
            for i in order[:count]:
                yield keys[i], self.batches[keys[i]]
        else:
            for batch_id, batch in self.batches.items():
                yield batch_id, batch
