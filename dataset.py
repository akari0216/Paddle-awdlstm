import paddle
from paddle.io import Dataset, Subset
import numpy as np
from model import *
import warnings
from tokenizer import *
from paddlenlp.datasets import MapDataset
import random
from functools import partial
import pandas as pd

warnings.filterwarnings("ignore")

def get_data(fpath):
    # class_dict = {
    #     "1": "World",
    #     "2": "Sports",
    #     "3": "Business",
    #     "4": "Sci/Tech"
    # }

    #对齐torch的字典顺序
    cls_map = {
        1: 3,
        2: 2,
        3: 0,
        4: 1
    }

    def convert_example(df):
        df["text"] = df["title"] + [" "] + df["description"]
        df.drop(columns=["title", "description"], inplace=True)
        df["label"] = df["label"].map(cls_map)
        return df

    headers = ["label", "title", "description"]
    df_train = pd.read_csv(fpath, names=headers)
    df_train = convert_example(df_train)

    label_list = df_train["label"].values.tolist()
    return df_train["text"].values.tolist(), label_list


class LMDateset(Dataset):
    def __init__(self, data, is_val = False, val_ratio=0.1, is_test=False):

        if is_test:
            self._data = data
        else:
            split = int(len(data) * (1-val_ratio))
            if is_val:
                self._data = data[split:]
            else:
                self._data = data[:split]
        self._is_test = is_test

    def __len__(self):
        return len(self._data)
    def __getitem__(self, idx):
        return self._data[idx]


#Class
class PadChunk(object):
    def __init__(self,
                 pad_idx = 1,
                 seq_len=72,
                 pad_first = True):
        self.pad_idx = pad_idx
        self.seq_len = seq_len
        self.pad_first = pad_first

    def __call__(self, data):
        original_length = [ele.shape[0] for ele in data]
        max_size = max(original_length)
        ret = []
        for i, arr in enumerate(data):
            if arr.shape[0] == max_size:
                ret.append(arr)
            else:
                ret.append(self.pad_chunk(arr, max_size))
        return np.array(ret)

    def pad_chunk(self, x, pad_len=10):
        "Pad `x` by adding padding by chunks of size `seq_len`"
        l = pad_len - x.shape[0]
        pad_chunk = np.zeros((l // self.seq_len) * self.seq_len) + self.pad_idx
        pad_res = np.zeros(l % self.seq_len) + self.pad_idx
        x1 = np.concatenate([pad_chunk, x, pad_res]) if self.pad_first else np.concatenate([x, pad_chunk, pad_res])
        return x1


def convert_example(example, tokenizer, is_lm=True, backwards=False):
    if is_lm:
        inputs, labels = tokenizer.get_item(example)
        return paddle.to_tensor(inputs), paddle.to_tensor(labels)

    return tokenizer.get_item(example)

def create_dataloader(dataset,
                      trans_fn=None,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == "train":
        sampler = paddle.io.DistributedBatchSampler(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        sampler = paddle.io.BatchSampler(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    dataloader = paddle.io.DataLoader(dataset, batch_sampler=sampler, drop_last=False, collate_fn=batchify_fn)
    return dataloader

def lm_dataloader(tokenizer, batch_size=64, train_flag=0, batchify_func=None):
    trans_func = partial(convert_example, tokenizer=tokenizer, is_lm=batchify_func is None)
    tok_idx = tokenizer.create_idxs()

    if train_flag == 0:
        #training
        random.shuffle(tok_idx)
        trainset = LMDateset(tok_idx, is_val=False, val_ratio=0.1, is_test=False)  # train
        train_ds = MapDataset(trainset)
        return create_dataloader(
            train_ds,
            mode='train',
            batch_size=batch_size,
            batchify_fn=batchify_func,
            trans_fn=trans_func)
    elif train_flag == 1:
        valset = LMDateset(tok_idx, is_val=True, val_ratio=0.1, is_test=False)  # eval
        val_ds = MapDataset(valset)
        return create_dataloader(
            val_ds,
            mode='eval',
            batch_size=batch_size,
            batchify_fn=batchify_func,
            trans_fn=trans_func)

    else:
        testset = LMDateset(tok_idx, is_val=False, val_ratio=0.1, is_test=True)  # test
        test_ds = MapDataset(testset)
        return create_dataloader(
            test_ds,
            mode='test',
            batch_size=batch_size,
            batchify_fn=batchify_func,
            trans_fn=trans_func)

