import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class GraphSkelDataset(Dataset):

    def __init__(self, skel_path, gloss_path, file_path, transform=None):

        with open(skel_path, mode='r', encoding='utf-8') as src_file:
            with open(*)
        self.skel_path =
