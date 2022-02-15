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
from logger import get_logger
import argparse
from paddlenlp.ops.optimizer import AdamWDL

set_logger = get_logger("./log","train_cls_log")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--epochs",
    type=int,
    default=10,
    help="Setting training epochs.")
parser.add_argument(
    "--lr",
    type=float,
    default=1e-3,
    help="Setting learning rate.")
parser.add_argument(
    "--freezing",
    type=int,
    default=None,
    help="Setting wether the model need be freezed.")
parser.add_argument(
    "--save_model_name",
    type=str,
    default="unfreeze_lm2.pdparams",
    help="Setting the model weights saving path.")
parser.add_argument(
    "--pre_train",
    type=str,
    default="converted_fwd.pdparams",
    help="Getting the pretrained weights file.")
args = parser.parse_args()

BATCH_SIZE = 64
train_texts, train_labels = get_data("train.csv")
tokenizer = CLSTokenizer(train_texts, train_labels, bs=BATCH_SIZE, backwards=False)
test_texts, test_labels =  get_data("test.csv")
test_tokenizer = TestCLSTokenizer(test_texts, test_labels, tokenizer.vocab, tokenizer.vocab_size,tokenizer.o2i, bs=1, backwards=False)

criterion = paddle.nn.loss.CrossEntropyLoss()
metric = paddle.metric.Accuracy()
ckpt_dir = "./ckpts" #模型保存路径

#加载数据
batchify_fn = lambda samples, fn=Tuple(
    PadChunk(), #padding to max_len
    Stack()  # labels
): [data for data in fn(samples)]
train_loader = lm_dataloader(tokenizer, batch_size=BATCH_SIZE, train_flag=0, batchify_func=batchify_fn)
eval_loader = lm_dataloader(tokenizer, batch_size=1, train_flag=1, batchify_func=batchify_fn)
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
        # output = F.softmax(output, axis=-1)
        loss = criterion(output.reshape((-1, output.shape[-1])), label_ids.flatten().astype("int64"))
        losses.append(loss.numpy())

        x1 = paddle.argmax(output, -1).flatten()
        acc =  (x1 == label_ids.flatten()).astype("float32").mean()
        acc_s .append(acc)

    print("eval loss: %.5f, accu: %.5f" % (np.mean(losses),  np.mean(acc_s)))
    set_logger.info("eval loss: %.5f, accu: %.5f" % (np.mean(losses),  np.mean(acc_s)))
    return np.mean(acc_s)

def train(
    epochs=1,
    lr=2.5e-3,
    freezing=None,
    pre_train="freeze_lm.pdparams",
    save_model_name = "test_cls.pdparams"):

    cls_num = 4
    model = get_text_classifier(
        AWD_LSTM,
        tokenizer.vocab_size,
        cls_num, seq_len=72,
        drop_mult=1.0,
        max_len=72 * 20,
        param_path="converted_fwd.pdparams")

    if pre_train is not None:
        pre_train_wgts = paddle.load(pre_train)

        if pre_train == "unfreeze_lm2.pdparams":
            new_pre_train_wgts = {}
            for k in pre_train_wgts:
                new_k = k.replace(".rnns", ".module.rnns")
                new_k = new_k.replace(".encoder", ".module.encoder")
                new_pre_train_wgts[new_k] = pre_train_wgts[k]

            pre_train_wgts = new_pre_train_wgts

        model.set_state_dict(pre_train_wgts)

        if freezing is not None:
            started_train = False

            if freezing == -1:
                print("freezing to last layer")
                set_logger.info("freezing to last layer")
                for param in model.parameters():
                    if param.name.startswith("batch_norm1d"):
                        started_train = True

                    if started_train:
                        param.trainable = True
                        print("training...", param.name)
                    else:
                        param.trainable = False
                        print("freezing...", param.name)

            elif freezing == -2:
                print("freezing to -2 layer")
                set_logger.info("freezing to -2 layer")
                for param in model.parameters():
                    if param.name.startswith("lstm_cell_2"):
                        started_train = True
                        
                    if started_train:
                        print("training...", param.name)
                        param.trainable = True
                    else:
                        print("freezing...", param.name)
                        param.trainable = False
                        
            elif freezing == -3:
                print("freezing to -3 layer")
                set_logger.info("freezing to -3 layer")
                for param in model.parameters():
                    if param.name.startswith("lstm_cell_1"):
                        started_train = True
                    if started_train:
                        print("training...", param.name)
                        param.trainable = True
                    else:
                        print("freezing...", param.name)
                        param.trainable = False

    #学习率
    num_training_steps = len(train_loader) * epochs
    print("lr scheduler: ", lr, num_training_steps, epochs)
    lr_scheduler = SlantedTriangularLR(lr, num_training_steps)

    ##定义优化器
    def decay_lr_setting(decay_rate, name_dict, n_layers, param):
        ratio = 1.0
        static_name = name_dict[param.name]
        if "0.module.rnns" in static_name:
            layer = int(static_name.split(".")[3])
            ratio = decay_rate ** (3 - layer)
        param.optimize_attr["learning_rate" ] *= ratio

    alpha = 1 / 2.6
    name_dict = dict()
    for n,p in model.named_parameters():
        name_dict[p.name] = n
        print("p name:",p.name,"n:",n)

    optimizer = AdamWDL(
        learning_rate = lr_scheduler,
        parameters = model.parameters(),
        layerwise_decay = alpha,
        set_param_lr_fun = decay_lr_setting,
        name_dict = name_dict
    )

    global_step = 0
    tic_train = time.time()
    best_accu = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for step, batch in enumerate(train_loader, start=1):
            model.reset()
            input_ids, label_ids = batch
            # 喂数据给model
            output = model(input_ids)
            # output = logits[0]
            # 计算损失函数值
            # output = F.softmax(output,  axis=-1)
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
                set_logger.info(
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

        test_accu = evaluate(model, test_loader)
        print("tested : ", test_accu)
        set_logger.info("tested : %s" % test_accu)
        if test_accu > best_accu:
            best_accu = test_accu
            paddle.save(model.state_dict(), save_model_name)

    print("best accu: %s" % best_accu)
    set_logger.info("best accu: %s" % best_accu)
    return model

clas_model = train(
    args.epochs,
    args.lr,
    args.freezing,
    args.pre_train,
    args.save_model_name
)