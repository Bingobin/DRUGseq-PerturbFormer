import numpy as np
import torch
from torch.utils.data import Dataset

class DrugSeqDataset(Dataset):
    def __init__(self, expr_mat, scores_3, labels_cls, meta_ids, well_ids, indices):
        self.idx = indices
        self.X = torch.tensor(expr_mat[indices], dtype=torch.float32)
        self.y3 = torch.tensor(scores_3[indices], dtype=torch.float32)
        self.y_cls = torch.tensor(labels_cls[indices], dtype=torch.float32)
        self.meta = np.array(meta_ids)[indices]
        self.well = np.array(well_ids)[indices]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return (self.X[i],
                self.y3[i],
                self.y_cls[i],
                self.idx[i],
                self.meta[i],
                self.well[i])
