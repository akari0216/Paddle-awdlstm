from functools import partial
import os
import time
from paddlenlp.datasets import MapDataset
import re
import html
from collections.abc import Iterable,Iterator,Generator,Sequence
from collections import Counter, defaultdict
import sys
sys.path.append('/home/aistudio/external-libraries')

import spacy
from spacy.symbols import ORTH

import paddle
from paddle.io import Dataset, Subset
import pandas as pd
import numpy as np
from paddlenlp.data import Pad, Stack, Tuple
from utils import get_language_model,get_text_classifier
from model import *
import paddle.nn.functional as F
import paddlenlp
import warnings

warnings.filterwarnings("ignore")

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

#Cell
def save_encoder(model,encoder_path):
    print("saving encoder")
    encoder_dict = {}
    state_dict = model.state_dict()
    for key in state_dict.keys():
        if "encoder" in key:
            encoder_dict[key] = state_dict[key]
    paddle.save(encoder_dict,encoder_path)
    print("encoder saved")

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

class LMTokenizer:
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


    def __init__(self, text, bs=2, seq_len=72):
        # self.label = label
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

        corpus = ((sum(self.full_lens)-1) / bs) * bs
        self.bl = corpus//bs # bl stands for batch length
        self.n_batches = self.bl//(seq_len) + int(self.bl%seq_len!=0)
        # self.n = int(self.n_batches * bs)
        # print()

        #避免len不足，又不能pad，遍历的时候又不能顾虑，所以在这里干掉
        self.n = int((self.n_batches - 1) * bs)

        # print(len(self.full_lens), self.full_lens[:3])
        self.cumlens = np.cumsum(([0] + self.full_lens))
        self.totlen = self.cumlens[-1]

        # print("cumlens", self.cumlens.shape, self.totlen)

        self.last_len = self.bl - (self.n_batches - 1) * seq_len

        count = Counter(char_tokens)
        self.vocab = self.make_vocab(count)
        self.vocab_size = len(self.vocab)
        self.o2i = defaultdict(int, {v: k for k, v in enumerate(self.vocab) if v != 'xxfake'})
        self.full_tokens = self.convert_token_id(self.full_tokens)
        print("--->", self.vocab_size, len(self.full_tokens), np.asarray(self.full_tokens).shape, self.full_tokens[0])

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
        return [i for i in range(self.n)]

    def doc_idx(self, i):
        if i<0: i=self.totlen+i
        docidx = np.searchsorted(self.cumlens, i+1)-1
        cl = self.cumlens[docidx]
        return docidx, i-cl

    def get_item(self, seq):
        # print("seq",  seq)
        # seq = 22152
        if seq is None: seq = 0
        if seq>=self.n: raise IndexError
        i_stop = self.last_len if seq//self.bs==self.n_batches-1 else self.seq_len
        i_start = int((seq%self.bs)*self.bl + (seq//self.bs)*self.seq_len)
        # print(i_start, i_stop)

        i_stop = int(i_stop + i_start + 1)

        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = self.totlen+1
        # print(i_start, i_stop)

        st_d, st_i = self.doc_idx(i_start)
        en_d,en_i = self.doc_idx(i_stop)
        res = self.full_tokens[st_d][st_i:(en_i if st_d==en_d else sys.maxsize)]
        for b in range(st_d+1,en_d): res+= self.full_tokens[b]
        if st_d!=en_d and en_d<len(self.full_tokens): res += self.full_tokens[en_d][:en_i]
        # print("debug", st_d, st_i, en_d, en_i, len(res))

        if  len(res) != 73:
            # print(seq,  i_start,  i_stop)
            # print("debug", st_d, st_i, en_d, en_i, len(res))
            return None, None
        return res[:-1], res[1:]


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

class LMDateset(Dataset):
    def __init__(self, data, is_test=False):
        self._data = data
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


def convert_example(example, tokenizer):
    inputs, labels =  tokenizer.get_item(example)
    # print(paddle.to_tensor(inputs), paddle.to_tensor(labels))
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

    dataloader = paddle.io.DataLoader(dataset, batch_sampler=sampler, drop_last=False)
    return dataloader


###########################################下面是测试#########################################
BATCH_SIZE = 64
class_dict = {
    "1": "World",
    "2": "Sports",
    "3": "Business",
    "4": "Sci/Tech"
}
labels = []
texts = []
with open("/home/aistudio/awdlstm2/config/ag_news/ag_news/train.csv", "r", encoding="utf-8") as f:
    for line in f:
        s = line.split('","')
        label,title,text = s[0].strip('"'),s[1],s[2].strip('"\n')
        text = title + " " + text
        labels.append(class_dict[label])
        texts.append(text)

tokenizer = LMTokenizer(texts, bs=BATCH_SIZE)
# print(tokenizer(["business is not sports"]))
# print(tokenizer.get_item(5998))
# print(convert_example(1, tokenizer))

baseset = LMDateset(tokenizer.create_idxs())
ds = MapDataset(baseset)


# val_labels = []
# val_texts = []
# with open("/home/aistudio/awdlstm2/config/ag_news/ag_news/test.csv", "r", encoding="utf-8") as f:
#     for line in f:
#         s = line.split('","')
#         label,title,text = s[0].strip('"'),s[1],s[2].strip('"\n')
#         text = title + " " + text
#         val_labels.append(class_dict[label])
#         val_texts.append(text)
# val_tokenizer = LMTokenizer(texts, bs=1)
# baseset = LMDateset(tokenizer.create_idxs())
# ds = MapDataset(baseset)



trans_func = partial(convert_example, tokenizer=tokenizer)
batchify_fn = lambda samples, fn=Tuple(
    Stack()  # labels
): [data for data in fn(samples)]

train_data_loader = create_dataloader(
    ds, 
    mode='train', 
    batch_size=BATCH_SIZE, 
    batchify_fn=None, 
    trans_fn=trans_func)

awd_lstm_lm_config = dict(emb_sz=400, n_hid=1152, n_layers=3, pad_token=1, bidir=False, output_p=0.1,
                          hidden_p=0.15, input_p=0.25, embed_p=0.02, weight_p=0.2, tie_weights=True, out_bias=True)

awd_lstm_clas_config = dict(emb_sz=400, n_hid=1152, n_layers=3, pad_token=1, bidir=False, output_p=0.4,
                            hidden_p=0.3, input_p=0.4, embed_p=0.05, weight_p=0.5)

pdparams = "/home/aistudio/awdlstm2/config/wt103-fwd2.pdparams"
model = get_language_model(
    AWD_LSTM, 
    tokenizer.vocab_size,
    config=awd_lstm_lm_config,
    param_path=pdparams)
# print("model now:",model)
# print("now state dicts:",model.state_dict().keys())
# for  k in model.state_dict().keys():
#     if "decoder.weight" in k :
#         print(k, model.state_dict()[k])
# exit()
# print("length of state dicts:",len(model.state_dict()))



# 训练轮次
epochs = 1
num_training_steps = len(train_data_loader) * epochs
print(num_training_steps, int(num_training_steps*0.1))
scheduler = paddlenlp.transformers.LinearDecayWithWarmup(1e-3, num_training_steps, 0.1)
lr = 1e-3

# Adam优化器
optimizer = paddle.optimizer.AdamW(
    learning_rate=lr,
    parameters=model.parameters())
# 交叉熵损失函数
criterion = paddle.nn.loss.CrossEntropyLoss()
metric = paddle.metric.Accuracy()
ckpt_dir = "/home/aistudio/work"

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


# 开启训练
def train(
    model,
    data_loader,
    epochs=1,
    lr=0.001,
    optimizer=optimizer,
    criterion=criterion,
    metric=metric,
    freezing=False,
    unfreezing_num=0):

    if freezing == True:
        print("freezing")
        freezed_state_dict = {}
        state_dict = model.state_dict()
        for key in state_dict.keys():
            freezed_state_dict[key] = paddle.to_tensor(state_dict[key].clone(),stop_gradient=True)
        model.set_state_dict(freezed_state_dict)

    global_step = 0
    tic_train = time.time()
    best_accu = 0
    for epoch in range(1, epochs + 1):
        for step, batch in enumerate(data_loader, start=1):
            input_ids, label_ids = batch
            # 喂数据给model
            logits = model(input_ids)
            output = logits[0]
            # 计算损失函数值

            loss = criterion(output.reshape((-1, output.shape[-1])), label_ids.flatten().astype("int64"))

            # 预测分类概率值

            global_step += 1
            if global_step % 100 == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f,speed: %.2f step/s, lr: %.7f"
                    % (global_step, epoch, step, loss,
                    10 / (time.time() - tic_train),lr))
                    # scheduler.get_lr()
                tic_train = time.time()

            # 反向梯度回传，更新参数
            loss.backward()
            # scheduler.step()
            optimizer.step()
            optimizer.clear_grad()

        # 评估当前训练的模型
        accu = evaluate(model, criterion, metric, data_loader)
        print(accu)

    return model

##以下部分不要删
#first train for freezing
# train_model = train(
#     model=model,
#     data_loader=train_data_loader,
#     epochs=epochs,
#     lr=5e-3,
#     freezing=True)

#unfreeze an save encoder
# train_model = train(
#     model=train_model,
#     data_loader=train_data_loader,
#     epochs=10,
#     lr=5e-4)

#save the encoder
encoder_path = "encoder.pdparams"
# save_encoder(train_model,encoder_path)

labels_clas = []
texts_clas = []
with open("/home/aistudio/awdlstm2/config/ag_news/ag_news/train.csv", "r", encoding="utf-8") as f:
    for line in f:
        s = line.split('","')
        label, title, text = s[0], s[1], s[2]
        text = title + " " + text
        label = re.sub("\D", "", label)
        labels_clas.append(int(label) - 1)
        texts_clas.append(text)

tokenizer_clas = CLSTokenizer(texts_clas,labels_clas,bs=BATCH_SIZE)

baseset_clas = LMDateset(tokenizer_clas.create_idxs())
ds_clas = MapDataset(baseset_clas)

cls_num = 4
clas_model = get_text_classifier(
    AWD_LSTM, 
    tokenizer_clas.vocab_size, 
    cls_num, 
    seq_len=72, 
    drop_mult=0.5,  
    max_len=72*20,
    config=awd_lstm_clas_config,
    param_path=pdparams)

#load encoder
clas_model = load_encoder(clas_model,encoder_path)

trans_func_clas = partial(convert_example, tokenizer=tokenizer_clas)

batchify_fn_padchunked = lambda samples, fn=Tuple(
    PadChunk(), #padding to max_len
    Stack()  # labels
): [data for data in fn(samples)]

clas_data_loader = create_dataloader(
    ds_clas, 
    mode='train', 
    batch_size=BATCH_SIZE, 
    batchify_fn=batchify_fn_padchunked, 
    trans_fn=trans_func_clas)


clas_model = train(
    model=clas_model,
    data_loader=clas_data_loader,
    epochs=1,
    lr=2.5e-2,
    freezing=True
)




