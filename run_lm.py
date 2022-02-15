from utils import get_language_model, SlantedTriangularLR
from model import *
from tokenizer import *
import numpy as np
import paddle
from dataset import lm_dataloader, get_data
import time
from functools import partial
from logger import get_logger
import argparse

set_logger = get_logger("./log","train_lm_log")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--epochs",
    type=int,
    default=100,
    help="Setting training epochs.")
parser.add_argument(
    "--lr",
    type=float,
    default=1e-3,
    help="Setting learning rate.")
parser.add_argument(
    "--freezing",
    type=bool,
    default=False,
    help="Setting wether the model need be freezed.")
parser.add_argument(
    "--save_model_name",
    type=str,
    default="unfreeze_lm2.pdparams",
    help="Seeting the model weights saving path.")
parser.add_argument(
    "--pre_train",
    type=str,
    default="converted_fwd.pdparams",
    help="Getting the pretrained weights file.")
args = parser.parse_args()

BATCH_SIZE = 64
train_texts, train_labels = get_data("train.csv")
tokenizer = LMTokenizer(train_texts, bs=BATCH_SIZE, backwards=False)

criterion = paddle.nn.loss.CrossEntropyLoss()
metric = paddle.metric.Accuracy()
ckpt_dir = "./ckpts" #模型保存路径

@paddle.no_grad()
def evaluate(model, data_loader):
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
        acc = (x1 == label_ids.flatten()).astype("float32").mean()
        acc_s.append(acc)

    print("eval loss: %.5f, accu: %.5f" % (np.mean(losses), np.mean(acc_s)))
    set_logger.info("eval loss: %.5f, accu: %.5f" % (np.mean(losses), np.mean(acc_s)))
    model.train()
    return np.mean(acc_s)


# 开启训练
def train(
    epochs=1,
    lr=0.001,
    freezing=False,
    save_model_name="model.pdparams",
    pre_train=None):

    model = get_language_model(
        AWD_LSTM,
        tokenizer.vocab_size,
        config=awd_lstm_lm_config,
        param_path="converted_fwd.pdparams")

    train_loader = lm_dataloader(tokenizer, batch_size=BATCH_SIZE, train_flag=0)
    eval_loader = lm_dataloader(tokenizer, batch_size=BATCH_SIZE, train_flag=1)

    if pre_train is not None:
        print("loading model:...", pre_train)
        state_dict = paddle.load(pre_train)
        model.set_state_dict(state_dict)

        accu = evaluate(model, eval_loader)
        print("validated : ", accu)


    if freezing == True:
        print("freezing to last layer")
        for param in model.parameters():
            if param.name.startswith("linear_0"):
                print("param name: ", param.name)
                param.trainable = True
            else:
                param.trainable = False

    #学习率
    num_training_steps = len(train_loader) * epochs
    print("lr scheduler: ", lr, num_training_steps, epochs)
    lr_scheduler = SlantedTriangularLR(lr, num_training_steps)

    # #定义优化器
    optimizer = paddle.optimizer.Adam(
        learning_rate=lr_scheduler,
        beta1 = 0.9,
        beta2 = 0.99,
        parameters=model.parameters())

    global_step = 0
    tic_train = time.time()
    best_accu = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for step, batch in enumerate(train_loader, start=1):
            model.reset()
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
                set_logger.info(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f,speed: %.2f step/s, lr: %.7f, acc: %.5f"
                    % (global_step, epoch, step, loss,
                    10 / (time.time() - tic_train), lr_scheduler.get_lr(), acc))

            # 反向梯度回传，更新参数
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            lr_scheduler.step()

        # 评估当前训练的模型
        accu = evaluate(model, eval_loader)
        print("validated : ", accu)
        set_logger.info("validated : %s" % accu)
        if accu > best_accu:
            best_accu = accu
            paddle.save(model.state_dict(), save_model_name)

    return model


#unfreeze an save encoder
train_model = train(
    args.epochs,
    args.lr,
    args.freezing,
    args.save_model_name,
    args.pre_train)

# encoder_path = "encoder.pdparams"
# save_encoder(train_model,encoder_path)



