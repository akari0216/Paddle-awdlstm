import paddle
import paddle.nn as nn
from model import *

__all__ = ['LinearDecoder', 'SequentialRNN','_rm_module','load_keys', 'get_language_model', 'SentenceEncoder', 'masked_concat_pool',
           'PoolingLinearClassifier', 'get_text_classifier']

def noop (x=None, *args, **kwargs):
    "Do nothing"
    return x

# Cell
_model_meta = {AWD_LSTM: {'hid_name':'emb_sz', 'url':"", 'url_bwd':"",
                          'config_lm':awd_lstm_lm_config, 'split_lm': awd_lstm_lm_split,
                          'config_clas':awd_lstm_clas_config, 'split_clas': awd_lstm_clas_split}}

# Cell
class LinearDecoder(nn.Layer):
    "To go on top of a RNNCore module and create a Language Model."
    initrange=0.1

    def __init__(self, n_out, n_hid, output_p=0.1, tie_encoder=None, bias=True):
        super(LinearDecoder,self).__init__()

        self.decoder = nn.Linear(n_hid, n_out, bias_attr=bias)
        # print("LinearDecoder dims: ", n_hid, n_out, self.decoder.weight.shape, tie_encoder.weight.shape)
        self.decoder.weight.set_value(paddle.uniform(shape=[n_hid,n_out],min=-self.initrange,max=self.initrange))
        self.output_dp = RNNDropout(output_p)
        if bias: self.decoder.bias.set_value(paddle.zeros(shape=[n_out]))
        if tie_encoder: self.decoder.weight.set_value(paddle.transpose(tie_encoder.weight, perm=[1, 0]))



    def forward(self, input):
        dp_inp = self.output_dp(input)
        # print('---->', dp_inp.shape)
        dec = self.decoder(dp_inp)
        # print('====>', dec.shape)
        return dec, input, dp_inp
        # return input

# Cell
class SequentialRNN(nn.Sequential):
    # def __init__(self):
    #     super(SequentialRNN,self).__init__()
        # self.decoder = decoder
        # self.encoder = encoder
    "A sequential module that passes the reset call to its children."
    def reset(self):
        print("calling the sequential rnn reset")
        for c in self.children():getattr(c,"reset", noop)()

# Cell
def _rm_module(n):
    t = n.split('.')
    for i in range(len(t)-1, -1, -1):
        if t[i] == 'module':
            t.pop(i)
            break
    return '.'.join(t)

#Cell load param,包含原代码里的clean_raw_keys和load_ignore_keys
def load_keys(model,wgts):
    print("loading pdparams...")
    keys = list(wgts.keys())
    for k in keys:
        t = k.split('.module')
        if f'{_rm_module(k)}_raw' in keys: del wgts[k]
    sd = wgts.copy()
    # print("sd:",sd)
    # for k1,k2 in zip(sd.keys(), wgts.keys()): 
    #     sd[k1] = paddle.to_tensor(data=wgts[k2].detach().clone())
    model.set_state_dict(sd)
    return model

# Cell
def get_language_model(arch, vocab_sz, config=None, drop_mult=1.,param_path = None):
    "Create a language model from `arch` and its `config`."
    meta = _model_meta[arch]
    if config is None:
        config = meta["config_lm"].copy()
    for k in config.keys():
        if k.endswith("_p"):config[k] *= drop_mult

    print(config)

    tie_weights,output_p,out_bias = map(config.pop,["tie_weights","output_p","out_bias"])
    init = config.pop("init") if "init" in config else None
    encoder = arch(vocab_sz, **config)
    enc = encoder.encoder if tie_weights else None
    decoder = LinearDecoder(vocab_sz,config[meta["hid_name"]],output_p,tie_encoder=enc,bias=out_bias)
    
    model = SequentialRNN(encoder,decoder)
    wgts = paddle.load(param_path)
    if "model" in wgts:wgts = wgts["model"]
    model = load_keys(model,wgts)
    # print("after load pdparams:",model)
    return model if init is None else model.apply("init")

# Cell
def _pad_tensor(t, bs):
    if t.shape[0] < bs:
        s = paddle.zeros(shape=[bs-t.shape[0],*t.shape[1:]])
        return paddle.concat([t,s])
    return t

# Cell
class SentenceEncoder(nn.Layer):
    "Create an encoder over `module` that can process a full sentence."
    def __init__(self,bptt,module,pad_idx=1,max_len=None):
        super(SentenceEncoder, self).__init__()
        self.bptt = bptt
        self.module = module
        self.pad_idx = pad_idx
        self.max_len = max_len


    def reset(self):getattr(self.module,"reset",noop)()

    def forward(self,input):
        bs,sl = input.shape
        self.reset()
        mask = (input == self.pad_idx).astype("int64")
        outs,masks = [],[]
        for i in range(0,sl,self.bptt):
            real_bs = (input[:,i] != self.pad_idx).astype("int64").sum()
            o = self.module(input[:real_bs,i:min(i+self.bptt,sl)])
            if self.max_len is None or sl - i <= self.max_len:
                outs.append(o)

                masks.append(mask[:,i:min(i+self.bptt,sl)].astype("bool"))
        outs = paddle.concat([_pad_tensor(o,bs) for o in outs],axis = 1)
        mask = paddle.concat(masks,axis = 1)
        return outs,mask

# Cell
def masked_concat_pool(output, mask, bptt):
    "Pool `MultiBatchEncoder` outputs into one vector [last_hidden, max_pool, avg_pool]"
    mask = mask.astype("int64")
    lens = output.shape[1] - mask.sum(axis=1)
    last_lens = mask[:,-bptt:].sum(axis=1)
    # print("okok: ", last_lens, bptt)

    ant_mask = (mask - 1) * (-1)#模拟一个取反操作


    # avg_pool = output.masked_fill(mask[:, :, None], 0).sum(axis=1)
    avg_pool = (output * ant_mask.reshape((ant_mask.shape[0], ant_mask.shape[1], 1))).sum(axis=1)

    # print("avg_pool shape: ", avg_pool.shape, avg_pool.dtype)
    lens = lens.astype(avg_pool.dtype).reshape((lens.shape[0], 1))
    avg_pool = paddle.divide(avg_pool, lens)


    # max_pool = output.masked_fill(mask[:,:,None], -float('inf')).max(axis=1)[0]

    inf_matrix = paddle.ones_like(output) * (-float("inf"))
    # print(inf_matrix.shape, output.shape, mask.shape, mask.unsqueeze(-1).expand(shape=inf_matrix.shape).shape)
    max_pool = paddle.where(mask.unsqueeze(-1).expand(shape=inf_matrix.shape).astype("bool"), inf_matrix, output).max(axis=1)

    # print("===>", avg_pool.shape, max_pool.shape, output.shape)
    # print(output[:output.shape[0], -last_lens - 1].shape)

    new_lens = -last_lens-1 + output.shape[1]
    index = paddle.concat((paddle.arange(0, output.shape[0]).unsqueeze(0), new_lens.unsqueeze(0))).transpose((1, 0))
    # print(index)
    # print(paddle.gather_nd(output, index).shape)
    x = paddle.concat([paddle.gather_nd(output, index), max_pool, avg_pool], axis = 1) #Concat pooling.
    # print(x)
    # exit()
    return x


# Cell
class LinBnDrop(nn.Sequential):
    "Module grouping `BatchNorm1d`, `Dropout` and `Linear` layers"
    def __init__(self, n_in, n_out, bn=True, p=0., act=None, lin_first=False):
        layers = [nn.BatchNorm1D(n_out if lin_first else n_in)] if bn else []
        if p != 0: layers.append(nn.Dropout(p))
        lin = [nn.Linear(n_in, n_out, bias_attr =not bn)]
        if act is not None: lin.append(act)
        layers = lin+layers if lin_first else layers+lin
        super().__init__(*layers)


# Cell
class PoolingLinearClassifier(nn.Layer):
    "Create a linear classifier with pooling"
    def __init__(self,dims,ps,bptt,y_range=None):
        super(PoolingLinearClassifier, self).__init__()
        if len(ps) != len(dims) - 1:raise ValueError("Number of layers and dropout values do not match.")
        acts = [nn.ReLU()] * (len(dims) - 2)  + [None]
        # LinBnDrop,SigmoidRange具体实现？
        layers = [LinBnDrop(i,o,p=p,act=a) for i,o,p,a in zip(dims[:-1],dims[1:],ps,acts)]
        # if y_range is not None:layers.append(SigmoidRange(*y_range)) #原位y_range没啥用
        self.layers = nn.Sequential(*layers)
        self.bptt = bptt

    def forward(self,input):
        out,mask = input
        x = masked_concat_pool(out,mask, self.bptt)
        x = self.layers(x)
        return x,out,out


# Cell
def get_text_classifier(arch, vocab_sz, n_class, seq_len=72, config=None, drop_mult=1., lin_ftrs=None,
                        ps=None, pad_idx=1, max_len=72*20, y_range=None,param_path=None, freeze_last=False):
    "Create a text classifier from `arch` and its `config`, maybe `pretrained`"
    meta = _model_meta[arch]
    if config is None:
        config = meta["config_clas"].copy()
    for k in config.keys():
        if k.endswith("_p"):config[k] *= drop_mult
    if lin_ftrs is None:lin_ftrs = [50]
    if ps is None:ps = [0.1] * len(lin_ftrs)
    layers = [config[meta["hid_name"]] * 3] + lin_ftrs + [n_class]
    ps = [config.pop("output_p")] + ps
    init = config.pop("init") if "init" in config else None
    encoder = SentenceEncoder(seq_len,arch(vocab_sz,**config),pad_idx=pad_idx,max_len=max_len)
    plc = PoolingLinearClassifier(layers,ps,bptt=seq_len,y_range=y_range)
    if freeze_last:
        plc.freeze_last();
    model = SequentialRNN(encoder, plc)



    wgts = paddle.load(param_path)
    if "model" in wgts:wgts = wgts["model"]
    model = load_keys(model,wgts)
    return model if init is None else model.apply("init")


#自定义学习率
from paddle.optimizer.lr import LRScheduler

class SlantedTriangularLR(LRScheduler):
    def __init__(self,
                 learning_rate,
                 iters,
                 cut_frac=0.1,
                 ratio=32,
                 last_epoch=-1,
                 verbose=False):

        self.cut = int(iters * cut_frac)
        self.cut = float(self.cut)
        self.cut_frac = cut_frac
        self.ratio = ratio

        super(SlantedTriangularLR, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.cut:
            p = self.last_epoch / self.cut
        else:
            p = 1 - (self.last_epoch - self.cut) / (self.cut * (1 / self.cut_frac - 1))

        return self.base_lr * (1 + p * (self.ratio - 1)) / self.ratio
