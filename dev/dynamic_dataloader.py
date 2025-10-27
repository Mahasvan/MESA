import math
from torch.utils.data import TensorDataset
from typing import Dict
import numpy as np
from collections import defaultdict

class CustomDataLoader:
    def __init__(self, dataset: TensorDataset, batch_size: int, gamma: float):
        self._batches = {}
        self._batch_losses = {}
        self._batch_counts = defaultdict(int)
        self._dataset = dataset
        self._batch_size = batch_size
        self._gamma = gamma
        self._make_batches()

    def __len__(self):
        return len(self._batches)

    def _make_batches(self):
        self.reset_batch_counts()
        for i in range(0, len(self._dataset), self._batch_size):
            batch = self._dataset[i: i + self._batch_size]
            self._batches[i] = batch

    def reset_batch_counts(self):
        self._batch_counts = defaultdict(int)

    def get_batch_counts(self):
        return self._batch_counts

    def update_losses(self, losses: Dict[int, float]):
        self._batch_losses.update(losses)

    def generate(self, dynamic: bool = False):
        # if its dynamic, it only returns 20% of the dataset (if gamma is 0.2)
        if dynamic and len(self._batch_losses) == len(self._batches):
            # print("Matched losses")
            order = np.argsort(list(self._batch_losses.values()))[::-1]
            count = math.ceil(len(order) * self._gamma)
            keys = list(self._batches.keys())
            for i in order[:count]:
                self._batch_counts[keys[i]] += 1
                yield keys[i], self._batches[keys[i]]
        else:
            for batch_id, batch in self._batches.items():
                self._batch_counts[batch_id] += 1
                yield batch_id, batch
