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
cls_num = 4
is_backwards = True
train_texts, train_labels = get_data("train.csv")
tokenizer = CLSTokenizer(train_texts, train_labels, bs=BATCH_SIZE, backwards=is_backwards)
test_texts, test_labels =  get_data("test.csv")
test_tokenizer = TestCLSTokenizer(test_texts, test_labels, tokenizer.vocab, tokenizer.vocab_size,tokenizer.o2i, bs=1,  backwards=is_backwards)


criterion = paddle.nn.loss.CrossEntropyLoss()
metric = paddle.metric.Accuracy()
ckpt_dir = "./ckpts" #模型保存路径

#加载数据
batchify_fn = lambda samples, fn=Tuple(
    PadChunk(), #padding to max_len
    Stack()  # labels
): [data for data in fn(samples)]

test_loader = lm_dataloader(test_tokenizer, batch_size=1, train_flag=2, batchify_func=batchify_fn)

# 加载模型

model_bwd = get_text_classifier(
    AWD_LSTM,
    tokenizer.vocab_size,
    cls_num, seq_len=72,
    drop_mult=0.5,
    max_len=72 * 20,
    param_path="cls_unfreeze_bwd.pdparams")
model_bwd.eval()

acc_s = []
res = []
labels = []
for batch in test_loader:
    input_ids, label_ids = batch
    labels.append(label_ids.numpy()[0])

    #bwd
    output_bwd = model_bwd(input_ids)
    output_bwd = F.softmax(output_bwd, axis=-1)
    res.append(output_bwd.numpy()[0])
    x1_bwd = paddle.argmax(output_bwd, axis=-1).flatten()
    acc =  (x1_bwd == label_ids.flatten()).astype("float32").mean()
    acc_s .append(acc)


final_res = np.asarray(res)
print("final_res shape", final_res.shape)
print(" accuracy: ", np.mean(acc_s))
np.save("bwd_res",final_res)
np.save("labels_res", np.asarray(labels))