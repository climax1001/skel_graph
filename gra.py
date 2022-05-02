import pandas as pd
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from torchtext.legacy import data
import io
import sys
import io
import numpy as np
class GraphSkelDataset(Dataset):

    def __init__(self, skel_path, gloss_path, file_path, transform=None, mode = None):
        if mode == 'train':
            self.mode = 'train'
        elif mode == 'test':
            self.mode = 'test'

        self.EOS_TOKEN = '</s>'
        self.BOS_TOKEN = '<s>'
        self.PAD_TOKEN = '<pad>'
        self.UNK_TOKEN = '<unk>'
        self.trg_size = 151
        self.level = "word"
        self.tok_fun = lambda s: list(s) if self.level == "char" else s.split()
        self.TARGET_PAD = 0.0

        def tokenize_features(features):
            features = torch.as_tensor(features)
            ft_list = torch.split(features, 1, dim=0)
            return [ft.squeeze() for ft in ft_list]

        def stack_features(features, something):
            return torch.stack([torch.stack(ft, dim=0) for ft in features], dim=0)

        self.lowercase = False
        self.src_field = data.Field(init_token=None, eos_token=self.EOS_TOKEN,
                       pad_token=self.PAD_TOKEN, tokenize=self.tok_fun,
                       batch_first=True, lower=self.lowercase,
                       unk_token=self.UNK_TOKEN,
                       include_lengths=True)

        self.files_field = data.RawField()
        self.trg_field =  data.Field(sequential=True,
                               use_vocab=False,
                               dtype=torch.float32,
                               batch_first=True,
                               include_lengths=False,
                               pad_token=torch.ones((self.trg_size,))*self.TARGET_PAD,
                               preprocessing=tokenize_features,
                               postprocessing=stack_features,)

        self.examples = []
        fields = (self.src_field, self.trg_field, self.files_field)
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1]), ('file_paths', fields[2])]

        with io.open(skel_path, mode='r', encoding='utf-8') as trg_file, \
                io.open(gloss_path, mode='r', encoding='utf-8') as src_file, \
                    io.open(file_path, mode='r', encoding='utf-8') as file_path:

            i = 0

            for src_line, trg_line, files_line in zip(src_file, trg_file, file_path):
                i += 1
                src_line, trg_line, files_line = src_line.strip(), trg_line.strip(), files_line.strip()
                trg_line = trg_line.split(' ')

                if len(trg_line) == 1:
                    break

                trg_line = [(float(joint) + 1e-8) for joint in trg_line]
                trg_line = np.array(trg_line).reshape(-1,151)
                # print('trg : ', trg_line)
                # print('src : ', src_line)
                # print('file : ', files_line)

                if src_line != '' and trg_line != '':
                    self.examples.append(data.Example.fromlist([src_line,trg_line,files_line], fields))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return


skel_path = '/home/juncislab/pythonProject1/data/train_skels.txt'
gloss_path = '/home/juncislab/pythonProject1/data/train_gloss.txt'
file_path = '/home/juncislab/pythonProject1/data/train_files.txt'
sdata=GraphSkelDataset(skel_path = skel_path, gloss_path = gloss_path, file_path=file_path)
print(sdata.__len__())