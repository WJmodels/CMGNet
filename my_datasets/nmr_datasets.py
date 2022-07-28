import random
import torch
import numpy as np
from torch.utils.data import Dataset


class NmrDataset(Dataset):
    def __init__(self, **kwargs):
        super(NmrDataset, self).__init__()
        self.scale = kwargs.pop(
            'scale') if 'scale' in kwargs.keys() else 10
        self.min_value = kwargs.pop(
            'min_value') if 'min_value' in kwargs.keys() else -50
        self.max_value = kwargs.pop(
            'max_value') if 'max_value' in kwargs.keys() else 350
        self.augment_prob = kwargs.pop(
            'augment_prob') if 'augment_prob' in kwargs.keys() else 0.5
        # augment_range unit ±ppm
        self.augment_range = kwargs.pop(
            'augment_range') if 'augment_range' in kwargs.keys() else 1.5
        self.units = (self.max_value - self.min_value) * self.scale
        self.start = kwargs.pop(
            'start') if 'start' in kwargs.keys() else 189
        self.end = kwargs.pop(
            'end') if 'end' in kwargs.keys() else 190
        self.number_start = kwargs.pop(
            'number_start') if 'number_start' in kwargs.keys() else 191

    def get_value(self, value, augment_prob):
        if random.random() < augment_prob:
            return value
        else:
            return 0

    # def move_parallelled(self, nmr):
    #     nmr = list(set(nmr))
    #     new_nmr = []
    #     offset = (random.random() * 2 - 1) * self.augment_range
    #     for value in nmr:
    #         if random.random() < self.augment_prob:
    #             new_nmr.append(value + offset)
    #         else:
    #             new_nmr.append(value)
    #     return self.fill_item(new_nmr)

    def fill_item(self, nmr, want_token=True):
        nmr = list(set(nmr))
        nmr = [round((value - self.min_value) * self.scale) for value in nmr]
        nmr = [min(max(0, value), self.units-1) for value in nmr]
        nmr.sort()
        if want_token:
            nmr = [189] + [value + self.number_start for value in nmr] + [190]
            item = {"input_ids": nmr,
                    "attention_mask": [1 for _ in range(len(nmr))]}
            return item
        else:
            item = np.zeros(self.units)
            item[nmr] = 1
            item = torch.from_numpy(item).to(torch.float32)
            item_ = {"input_ids": [189, 191, 190],
                     "attention_mask": [1, 1, 1]}
            return item, item_

    def augment_item(self, nmr, want_token=True):
        nmr = list(set(nmr))
        offset = np.random.normal(
            loc=0.0, scale=self.augment_range/2, size=len(nmr))
        offset = [min(max(-self.augment_range, i), self.augment_range)
                  for i in offset]
        offset = [self.get_value(i, self.augment_prob) for i in offset]
        new_nmr = [nmr_item + offset_item for nmr_item,
                   offset_item in zip(nmr, offset)]
        return self.fill_item(new_nmr, want_token=want_token)
