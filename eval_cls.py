from utils import get_language_model, SlantedTriangularLR, get_text_classifier, load_keys
from model import *
from tokenizer import *
import numpy as np
import paddle
from dataset import lm_dataloader, get_data, PadChunk
import time
from functools import partial
from paddlenlp.data import Pad, Stack, Tuple
import paddle.nn.functional as F


BATCH_SIZE = 64
train_texts, train_labels = get_data("train.csv")
tokenizer = CLSTokenizer(train_texts, train_labels, bs=BATCH_SIZE)
test_texts, test_labels =  get_data("test.csv")
test_tokenizer = TestCLSTokenizer(test_texts, test_labels, tokenizer.vocab, tokenizer.vocab_size,tokenizer.o2i, bs=1)


criterion = paddle.nn.loss.CrossEntropyLoss()
metric = paddle.metric.Accuracy()
ckpt_dir = "./ckpts" #模型保存路径

#加载数据
batchify_fn = lambda samples, fn=Tuple(
    PadChunk(), #padding to max_len
    Stack()  # labels
): [data for data in fn(samples)]

test_loader = lm_dataloader(test_tokenizer, batch_size=1, train_flag=2, batchify_func=batchify_fn)

@paddle.no_grad()
def evaluate(model,data_loader):
    model.eval()
    losses = []
    acc_s = []
    for batch in data_loader:
        input_ids, label_ids = batch
        output = model(input_ids)
        # output = logits[0]
        loss = criterion(output.reshape((-1, output.shape[-1])), label_ids.flatten().astype("int64"))
        losses.append(loss.numpy())

        output = F.softmax(output, axis=-1)

        x1 = paddle.argmax(output, axis=-1).flatten()
        acc =  (x1 == label_ids.flatten()).astype("float32").mean()

        acc_s .append(acc)

    print("eval loss: %.5f, accu: %.5f" % (np.mean(losses),  np.mean(acc_s)))
    model.train()
    return np.mean(acc_s)

# jiazai
cls_num = 4
model = get_text_classifier(
    AWD_LSTM,
    tokenizer.vocab_size,
    cls_num, seq_len=72,
    drop_mult=0.5,
    max_len=72 * 20,
    param_path="converted_fwd.pdparams")

evaluate(model, test_loader)
