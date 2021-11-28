import sys 
sys.path.append('/home/aistudio/external-libraries')

from functools import partial
import os
import time
import spacy
from spacy.symbols import ORTH
from paddlenlp.datasets import MapDataset
import re
import html
from collections.abc import Iterable,Iterator,Generator,Sequence
from collections import Counter, defaultdict

import paddle
from paddle.io import Dataset, Subset
from paddlenlp.datasets import MapDataset
import pandas as pd
import numpy as np
from paddlenlp.data import Pad, Stack, Tuple
from utils import get_language_model, get_text_classifier, SlantedTriangularLR
from model import *
import paddle.nn.functional as F
import paddlenlp
import random

"""
本文件以lm为例
"""

#special tokens
UNK, PAD, BOS, EOS, FLD, TK_REP, TK_WREP, TK_UP, TK_MAJ = "xxunk xxpad xxbos xxeos xxfld xxrep xxwrep xxup xxmaj".split()
default_text_spec_tok = [UNK, PAD, BOS, EOS, FLD, TK_REP, TK_WREP, TK_UP, TK_MAJ]

# Cell
_re_spec = re.compile(r'([/#\\])')

def spec_add_spaces(t):
    "Add spaces around / and #"
    return _re_spec.sub(r' \1 ', t)

# Cell
_re_space = re.compile(' {2,}')

def rm_useless_spaces(t):
    "Remove multiple spaces"
    return _re_space.sub(' ', t)


# Cell
_re_rep = re.compile(r'(\S)(\1{2,})')

def replace_rep(t):
    "Replace repetitions at the character level: cccc -- TK_REP 4 c"
    def _replace_rep(m):
        c,cc = m.groups()
        return f' {TK_REP} {len(cc)+1} {c} '
    return _re_rep.sub(_replace_rep, t)

# Cell
_re_wrep = re.compile(r'(?:\s|^)(\w+)\s+((?:\1\s+)+)\1(\s|\W|$)')

# Cell
def replace_wrep(t):
    "Replace word repetitions: word word word word -- TK_WREP 4 word"
    def _replace_wrep(m):
        c,cc,e = m.groups()
        return f' {TK_WREP} {len(cc.split())+2} {c} {e}'
    return _re_wrep.sub(_replace_wrep, t)

# Cell
def fix_html(x):
    "Various messy things we've seen in documents"
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace('nbsp;', ' ').replace(
        '#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace('<br />', "\n").replace(
        '\\"', '"').replace('<unk>',UNK).replace(' @.@ ','.').replace(' @-@ ','-').replace('...',' …')
    return html.unescape(x)

# Cell
_re_all_caps = re.compile(r'(\s|^)([A-Z]+[^a-z\s]*)(?=(\s|$))')

# Cell
def replace_all_caps(t):
    "Replace tokens in ALL CAPS by their lower version and add `TK_UP` before."
    def _replace_all_caps(m):
        tok = f'{TK_UP} ' if len(m.groups()[1]) > 1 else ''
        return f"{m.groups()[0]}{tok}{m.groups()[1].lower()}"
    return _re_all_caps.sub(_replace_all_caps, t)

# Cell
_re_maj = re.compile(r'(\s|^)([A-Z][^A-Z\s]*)(?=(\s|$))')

# Cell
def replace_maj(t):
    "Replace tokens in Sentence Case by their lower version and add `TK_MAJ` before."
    def _replace_maj(m):
        tok = f'{TK_MAJ} ' if len(m.groups()[1]) > 1 else ''
        return f"{m.groups()[0]}{tok}{m.groups()[1].lower()}"
    return _re_maj.sub(_replace_maj, t)

# Cell
def lowercase(t, add_bos=True, add_eos=False):
    "Converts `t` to lowercase"
    return (f'{BOS} ' if add_bos else '') + t.lower().strip() + (f' {EOS}' if add_eos else '')

# Cell
def replace_space(t):
    "Replace embedded spaces in a token with unicode line char to allow for split/join"
    return t.replace(' ', '▁')



defaults_text_proc_rules = [fix_html, replace_rep, replace_wrep, spec_add_spaces, rm_useless_spaces,
                            replace_all_caps, replace_maj, lowercase]
defaults_text_postproc_rules = [replace_space]


def noop (x=None, *args, **kwargs):
    "Do nothing"
    return x

# Cell
def compose(*funcs):
    "Create a function that composes all functions in `funcs`, passing along remaining `*args` and `**kwargs` to all"
    funcs = list(funcs)
    if len(funcs)==0: return noop
    if len(funcs)==1: return funcs[0]
    def _inner(x, *args, **kwargs):
        for f in funcs: x = f(x, *args, **kwargs)
        return x
    return _inner

# Cell
def maps(*args):
    "Like `map`, except funcs are composed first"
    f = compose(*args[:-1])
    def _f(b): return f(b)
    return map(_f, args[-1])


class SpacyTokenizer():
    "Spacy tokenizer for `lang`"
    def __init__(self, lang='en', special_toks=None, buf_sz=5000):
        if special_toks is None:
            self.special_toks = default_text_spec_tok

        nlp = spacy.blank(lang)
        for w in self.special_toks: nlp.tokenizer.add_special_case(w, [{ORTH: w}])
        self.pipe, self.buf_sz = nlp.pipe, buf_sz

    def __call__(self, items):
        return (list(map(str, list(doc))) for doc in self.pipe(map(str, items), batch_size=self.buf_sz))

class CLSTokenizer:
    min_freq = 3
    max_vocab = 60000
    special_toks = default_text_spec_tok
    vocab = None
    vocab_size = None
    o2i = None
    spacy_tok = None
    full_tokens = []
    full_lens = []
    cumlens = 0
    totlen = 0

    def __init__(self, text, label, bs=2, seq_len=72):
        self.label = label
        self.bs = bs
        self.seq_len = seq_len
        self.spacy_tok = SpacyTokenizer()
        tok_res = self.get_spacy_toks(text)
        char_tokens = []
        self.full_tokens = []
        for tr in tok_res:
            char_tokens += tr
            self.full_lens.append(len(tr))
            self.full_tokens.append(tr)
        self.totlen = len(self.full_tokens)

        count = Counter(char_tokens)
        self.vocab = self.make_vocab(count)
        self.vocab_size = len(self.vocab)
        self.o2i = defaultdict(int, {v: k for k, v in enumerate(self.vocab) if v != 'xxfake'})

        self.full_tokens = self.convert_token_id(self.full_tokens)
        # print(self.vocab_size, len(self.full_tokens), np.asarray(self.full_tokens).shape, self.full_tokens[0], self.totlen)

    def get_spacy_toks(self, text):
        tok_res = self.spacy_tok(maps(*defaults_text_proc_rules, text))
        return (list(maps(*defaults_text_postproc_rules, o)) for o in tok_res)

    def make_vocab(self, count):
        vocab = [o for o, c in count.most_common(self.max_vocab) if c >= self.min_freq]
        for o in reversed(self.special_toks):  # Make sure all special tokens are in the vocab
            if o in vocab: vocab.remove(o)
            vocab.insert(0, o)
        vocab = vocab[:self.max_vocab]
        return vocab + [f'xxfake' for i in range(0, 8 - len(vocab) % 8)]

    def __call__(self, o):
        words = self.get_spacy_toks(o)
        return self.convert_token_id(words)

    def convert_token_id(self, words):
        all = []
        for o_ in list(words):
            tmp = []
            for x in o_:
                tmp.append(self.o2i[x])
            all.append(tmp)
        return all

    def decode(self, o):
        return [self.vocab[o_] for o_ in o]

    def create_idxs(self):
        return [i for i in range(self.totlen)]

    def doc_idx(self, i):
        if i<0: i=self.totlen+i
        docidx = np.searchsorted(self.cumlens, i+1)-1
        cl = self.cumlens[docidx]
        return docidx, i-cl

    def get_item(self, idx):
        return np.array(self.full_tokens[idx]), np.array(self.label[idx], dtype="int64")


class TestCLSTokenizer:
    special_toks = default_text_spec_tok
    vocab = None
    vocab_size = None
    o2i = None
    spacy_tok = None
    full_tokens = []
    full_lens = []
    cumlens = 0
    totlen = 0

    def __init__(self, text, label, vocab, vocab_size, o2i, bs=2, seq_len=72):
        self.label = label
        self.bs = bs
        self.seq_len = seq_len
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.o2i = o2i
        self.spacy_tok = SpacyTokenizer()
        tok_res = self.get_spacy_toks(text)
        char_tokens = []
        self.full_tokens = []
        for tr in tok_res:
            char_tokens += tr
            self.full_lens.append(len(tr))
            self.full_tokens.append(tr)
        self.totlen = len(self.full_tokens)
        self.full_tokens = self.convert_token_id(self.full_tokens)

    def get_spacy_toks(self, text):
        tok_res = self.spacy_tok(maps(*defaults_text_proc_rules, text))
        return (list(maps(*defaults_text_postproc_rules, o)) for o in tok_res)

    def __call__(self, o):
        words = self.get_spacy_toks(o)
        return self.convert_token_id(words)

    def convert_token_id(self, words):
        all = []
        for o_ in list(words):
            tmp = []
            for x in o_:
                tmp.append(self.o2i[x])
            all.append(tmp)
        return all

    def decode(self, o):
        return [self.vocab[o_] for o_ in o]

    def create_idxs(self):
        return [i for i in range(self.totlen)]

    def get_item(self, idx):
        return np.array(self.full_tokens[idx]), np.array(self.label[idx], dtype="int64")


class CLSDateset(Dataset):
    def __init__(self, data, is_val=False, val_ratio=0.1, is_test=False):
        if is_test:
            self._data = data
        else:
            split = int(len(data) * (1 - val_ratio))
            if is_val:
                self._data = data[split:]
            else:
                self._data = data[:split]
        self._is_test = is_test

    def __len__(self):
        return len(self._data)
    def __getitem__(self, idx):
        return self._data[idx]


def convert_example(example, tokenizer):
    inputs, labels = tokenizer.get_item(example)
    return paddle.to_tensor(inputs), paddle.to_tensor(labels)

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

    dataloader = paddle.io.DataLoader(dataset, batch_sampler=sampler,  collate_fn=batchify_fn)
    return dataloader

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
        # print(data, max_size)
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

#Cell
def load_encoder(model,encoder_path):
    print("loading encoder")
    encoder_dict = paddle.load(encoder_path)
    state_dict = model.state_dict()
    new_state_dict = {}
    for key in encoder_dict.keys():
        new_state_dict[key] = encoder_dict[key]
    
    for key in state_dict.keys():
        if "encoder" not in key:
            new_state_dict[key] = state_dict[key]

    model.set_state_dict(new_state_dict)
    print("encoder loaded")
    return model


###########################################下面是测试#########################################
epochs = 1
BATCH_SIZE = 64
pdparams = "/d/hulei/pd_match/papers_reproduce/Paddle-awdlstm/new_down/Paddle-awdlstm/wt103-fwd2.pdparams"
encoder_path = "/d/hulei/pd_match/papers_reproduce/Paddle-awdlstm/new_down/Paddle-awdlstm/unfreeze_lm.pdparams"
class_dict = {
    "1": "World",
    "2": "Sports",
    "3": "Business",
    "4": "Sci/Tech"
}
labels = []
texts = []
with open("./config/ag_news/ag_news/train.csv", "r", encoding="utf-8") as f:
    for line in f:
        s = line.split('","')
        label, title, text = s[0], s[1], s[2]
        text = title + " " + text
        label = re.sub("\D", "", label)
        labels.append(int(label) - 1)
        texts.append(text)

tokenizer = CLSTokenizer(texts, labels, bs=BATCH_SIZE)
tok_idx = tokenizer.create_idxs()
random.shuffle(tok_idx)
trainset = CLSDateset(tok_idx, is_val=False, val_ratio=0.1, is_test=False)
valset = CLSDateset(tok_idx, is_val=True, val_ratio=0.1, is_test=False)
train_ds = MapDataset(trainset)
val_ds = MapDataset(valset)

trans_func = partial(convert_example, tokenizer=tokenizer)
batchify_fn = lambda samples, fn=Tuple(
    PadChunk(), #padding to max_len
    Stack()  # labels
): [data for data in fn(samples)]

train_data_loader = create_dataloader(
    train_ds,
    mode='train', 
    batch_size=BATCH_SIZE, 
    batchify_fn=batchify_fn, 
    trans_fn=trans_func)

val_data_loader = create_dataloader(
    val_ds,
    mode='train',
    batch_size=BATCH_SIZE,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)

cls_num = 4
model = get_text_classifier(
    AWD_LSTM, 
    tokenizer.vocab_size, 
    cls_num, seq_len=72, 
    drop_mult=0.5,  
    max_len=72*20,
    param_path=pdparams)

@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader):
    model.eval()
    losses = []
    acc_s = []
    for batch in data_loader:
        input_ids, label_ids = batch
        logits = model(input_ids)
        output = logits[0]
        loss = criterion(output.reshape((-1, output.shape[-1])), label_ids.flatten().astype("int64"))
        losses.append(loss.numpy())

        x1 = paddle.argmax(output, -1).flatten()
        acc =  (x1 == label_ids.flatten()).astype("float32").mean()
        acc_s .append(acc)

    print("eval loss: %.5f, accu: %.5f" % (np.mean(losses),  np.mean(acc_s)))
    model.train()
    return np.mean(acc_s)

#load encoder
model = load_encoder(model,encoder_path)

# 交叉熵损失函数
criterion = paddle.nn.loss.CrossEntropyLoss()
metric = paddle.metric.Accuracy()


def train(
    model,
    train_data_loader,
    epochs=1,
    lr=2.5e-3,
    criterion=criterion,
    metric=metric,
    freezing=None,
    pre_train=None,
    save_model_name = "test_cls.pdparams"):

    if pre_train is not None:
        print("loaded pre_train", pre_train)
        state_dict = paddle.load(pre_train)
        model.set_state_dict(state_dict)

    if freezing is not None:
        print("freezing to last layer")

        if freezing == -1:
            for param in model.parameters():
                if param.name.startswith("linear_1"):
                    param.trainable = True
                else:
                    param.trainable = False
        elif freezing == -2:
            for param in model.parameters():
                if param.name.startswith("linear_1") or param.name.startswith("batch_norm1d_1"):
                    param.trainable = True
                else:
                    param.trainable = False
        elif freezing == -3:
            for param in model.parameters():
                if param.name.startswith("linear_1") or param.name.startswith("batch_norm1d_1") or param.name.startswith("linear_0") :
                    param.trainable = True
                else:
                    param.trainable = False

    #学习率
    num_training_steps = len(train_data_loader) * epochs
    print("lr scheduler: ", lr, num_training_steps, epochs)
    lr_scheduler = SlantedTriangularLR(lr, num_training_steps)

    #定义优化器
    optimizer = paddle.optimizer.Adam(
        learning_rate=lr_scheduler,
        beta1 = 0.7,
        beta2 = 0.99,
        weight_decay=0.01,
        parameters=model.parameters())

    global_step = 0
    tic_train = time.time()
    best_accu = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for step, batch in enumerate(train_data_loader, start=1):
            input_ids, label_ids = batch
            # 喂数据给model
            logits = model(input_ids)
            output = logits[0]
            # 计算损失函数值

            loss = criterion(output.reshape((-1, output.shape[-1])), label_ids.flatten().astype("int64"))

            # 预测分类概率值
            x1 = paddle.argmax(output, -1).flatten()
            acc = (x1 == label_ids.flatten()).astype("float32").mean()

            global_step += 1
            if global_step % 100 == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f,speed: %.2f step/s, lr: %.7f, acc: %.5f"
                    % (global_step, epoch, step, loss,
                       10 / (time.time() - tic_train), lr_scheduler.get_lr(), acc))
                # scheduler.get_lr()
                tic_train = time.time()

            # 反向梯度回传，更新参数
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            lr_scheduler.step()

        # 评估当前训练的模型
        accu = evaluate(model, criterion, metric, val_data_loader)
        print("validated : ", accu)
        if accu > best_accu:
            best_accu = accu
            paddle.save(model.state_dict(), save_model_name)

    return model

# clas_model = train(
#     model,
#     train_data_loader,
#     epochs=3,
#     save_model_name = "cls_freeze_1.pdparams"
#     )


# clas_model = train(
#     model,
#     train_data_loader,
#     epochs=3,
#     save_model_name = "cls_freeze_2.pdparams",
#     pre_train="cls_freeze_1.pdparams"
#     )

# clas_model = train(
#     model,
#     train_data_loader,
#     epochs=3,
#     save_model_name = "cls_freeze_3.pdparams",
#     pre_train="cls_freeze_2.pdparams"
#     )


clas_model = train(
    model,
    train_data_loader,
    epochs=30,
    save_model_name = "cls_unfreeze.pdparams",
    pre_train="cls_freeze_3.pdparams"
    )

#==================运行测试集==============
test_labels = []
test_texts = []
with open("./config/ag_news/ag_news/test.csv", "r", encoding="utf-8") as f:
    for line in f:
        s = line.split('","')
        label,title,text = s[0].strip('"'),s[1],s[2].strip('"\n')
        text = title + " " + text
        test_labels.append(class_dict[label])
        test_texts.append(text)

test_tokenizer = TestCLSTokenizer(test_texts, test_labels, tokenizer.vocab, tokenizer.vocab_size,tokenizer.o2i, bs=1)
tok_idx = test_tokenizer.create_idxs()
testset = CLSDateset(tok_idx, is_test=True)
test_ds = MapDataset(testset)

trans_func = partial(convert_example, tokenizer=tokenizer)
batchify_fn = lambda samples, fn=Tuple(
    PadChunk(), #padding to max_len
    Stack()  # labels
): [data for data in fn(samples)]

test_data_loader = create_dataloader(
    test_ds,
    mode='test',
    batch_size=BATCH_SIZE,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)
accu = evaluate(clas_model, criterion, metric, test_data_loader)
print("tested : ", accu)










