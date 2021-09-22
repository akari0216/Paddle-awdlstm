import paddle
import paddle.nn as nn
from .model import *

__all__ = ['LinearDecoder', 'SequentialRNN', 'get_language_model', 'SentenceEncoder', 'masked_concat_pool',
           'PoolingLinearClassifier', 'get_text_classifier']

# Cell
_model_meta = {AWD_LSTM: {'hid_name':'emb_sz', 'url':URLs.WT103_FWD, 'url_bwd':URLs.WT103_BWD,
                          'config_lm':awd_lstm_lm_config, 'split_lm': awd_lstm_lm_split,
                          'config_clas':awd_lstm_clas_config, 'split_clas': awd_lstm_clas_split}}

# Cell
class LinearDecoder(nn.Layer):
    "To go on top of a RNNCore module and create a Language Model."
    initrange=0.1

    def __init__(self, n_out, n_hid, output_p=0.1, tie_encoder=None, bias=True):
        super(LinearDecoder,self).__init__()
        self.decoder = nn.Linear(n_hid, n_out, bias_attr=bias)
        self.decoder.weight.set_value(paddle.uniform(shape=[n_hid,n_out],min=-self.initrange,max=self.initrange))
        self.output_dp = RNNDropout(output_p)
        if bias: self.decoder.bias.set_value(paddle.zeros(shape=[n_out]))
        if tie_encoder: self.decoder.weight = tie_encoder.weight

    def forward(self, input):
        dp_inp = self.output_dp(input)
        return self.decoder(dp_inp), input, dp_inp

# Cell
class SequentialRNN(nn.Sequential):
    "A sequential module that passes the reset call to its children."
    def reset(self):
        for c in self.children():getattr(c,"reset",noop)()

# Cell
def get_language_model(arch, vocab_sz, config=None, drop_mult=1.):
    "Create a language model from `arch` and its `config`."
    meta = _model_meta[arch]
    if config is None:
        config = meta["config_lm"].copy()
    for k in config.keys():
        if k.endswith("_p"):config[k] * = drop_mult
    tie_weights,output,out_bias = map(config.pop,["tie_weights","output_p","out_bias"])
    init = config.pop("init") if "init" in config else None
    encoder = arch(vocab_sz,**config)
    enc = encoder.encoder if tie_weights else None
    decoder = LinearDecoder(vocab_sz,config[meta["hid_name"]],output_p,tie_encoder=enc,bias=out_bias)
    model = SequentialRNN(encoder,decoder)
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
        self.bptt = bptt
        self.module = module
        self.pad_idx = pad_idx
        self.max_len = max_len
    def reset(self):getattr(self.module,"reset",noop)()

    def forward(self,input):
        bs,sl = input.shape
        self.reset()
        mask = input == self.pad_idx
        outs,masks = [],[]
        for i in range(0,sl,self.bptt):
            real_bs = (input[:,i] != self.pad_idx).long().sum()
            o = self.module(input[:real_bs,i:min(i+self.bptt,sl)])
            if self.max_len is None or sl - i <= self.max_len:
                outs.append(o)
                masks.append(mask[:,i:min(i+self.bptt,sl)])
        outs = paddle.concat([_pad_tensor(o,bs) for o in outs],axis = 1)
        mask = paddle.concat(masks,axis = 1)
        return outs,mask

# Cell
def masked_concat_pool(output, mask, bptt):
    "Pool `MultiBatchEncoder` outputs into one vector [last_hidden, max_pool, avg_pool]"
    lens = output.shape[1] - mask.long().sum(dim=1)
    last_lens = mask[:,-bptt:].long().sum(dim=1)
    avg_pool = output.masked_fill(mask[:, :, None], 0).sum(dim=1)
    avg_pool.div_(lens.type(avg_pool.dtype)[:,None])
    max_pool = output.masked_fill(mask[:,:,None], -float('inf')).max(dim=1)[0]
    x = paddle.concat([output[paddle.arange(0, output.shape[0]),-last_lens-1], max_pool, avg_pool], axis = 1) #Concat pooling.
    return x

# Cell
class PoolingLinearClassifier(nn.Layer):
    "Create a linear classifier with pooling"
    def __init__(self,dims,ps,bptt,y_range=None):
        if len(ps) != len(dims) - 1:raise ValueError("Number of layers and dropout values do not match.")
        acts = [nn.ReLU()] * (len(dims) - 2)  + [None]
        # LinBnDrop,SigmoidRange具体实现？
        layers = [LinBnDrop(i,o,p=p,act=a) for i,o,p,a in zip(dims[:-1],dims[1:],ps,acts)]
        if y_range is not None:layers.append(SigmoidRange(*y_range))
        self.laysers = nn.Sequential(*layers)
        self.bptt = bptt

    def forward(self,input):
        out,mask = input
        x = masked_concat_pool(out,mask,self,bptt)
        x = self.layers(x)
        return x,out,out


# Cell
def get_text_classifier(arch, vocab_sz, n_class, seq_len=72, config=None, drop_mult=1., lin_ftrs=None,
                        ps=None, pad_idx=1, max_len=72*20, y_range=None):
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
    model = SequentialRNN(encoder,PoolingLinearClassifier(layers,ps,bptt=seq_len,y_range=y_range))
    return model if init is None else model.apply("init")