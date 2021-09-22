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
    

# Cell
class SentenceEncoder(Module):
    "Create an encoder over `module` that can process a full sentence."
    None

# Cell
def masked_concat_pool(output, mask, bptt):
    "Pool `MultiBatchEncoder` outputs into one vector [last_hidden, max_pool, avg_pool]"
    None

# Cell
class PoolingLinearClassifier(Module):
    "Create a linear classifier with pooling"
    None

# Cell
def get_text_classifier(arch, vocab_sz, n_class, seq_len=72, config=None, drop_mult=1., lin_ftrs=None,
                        ps=None, pad_idx=1, max_len=72*20, y_range=None):
    "Create a text classifier from `arch` and its `config`, maybe `pretrained`"
    None