from functools import partial
import os
import time

import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset,MapDataset
from paddlenlp.transformers import SkepTokenizer

tokenizer_model_name = "skep_ernie_2.0_large_en"

def load_dataset(datafiles):
    def read(data_path):
        with open(data_path,"r",encoding="utf-8") as f:
            for line in f:
                s = line.split('","')
                label,text,text_pair = s[0].strip('"'),s[1],s[2].strip('"\n')
                yield {"label":label,"text":text,"text_pair":text_pair}

    if isinstance(datafiles, str):
        return MapDataset(list(read(datafiles)))
    elif isinstance(datafiles, list) or isinstance(datafiles, tuple):
        return [MapDataset(list(read(datafile))) for datafile in datafiles]

train = load_dataset("./config/ag_news/ag_news/train.csv")
test = load_dataset("./config/ag_news/ag_news/test.csv")

tokenizer = SkepTokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_model_name)

def convert_example(example,
                    tokenizer,
                    max_seq_length=512,
                    is_test=False):

    encoded_inputs = tokenizer(
    text=example["text"],
    text_pair=example["text_pair"],
    max_seq_len=max_seq_length)

    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    else:
        return input_ids, token_type_ids

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
    dataloader = paddle.io.DataLoader(
        dataset, batch_sampler=sampler, collate_fn=batchify_fn)
    return dataloader

max_seq_length = 256
batch_size=16

trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length)

batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=1),  # input_ids
    Pad(axis=0, pad_val=1),  # token_type_ids
    Stack(dtype="int64")  # labels
): [data for data in fn(samples)]

train_data_loader = create_dataloader(
    train,
    mode='train',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)

for step,batch in enumerate(train_data_loader):
    if step <= 1:
        input_ids, token_type_ids, labels = batch
        print("input_ids:",input_ids)
        print("token_type_ids:",token_type_ids)
        print("labels:",labels)
    else:
        break