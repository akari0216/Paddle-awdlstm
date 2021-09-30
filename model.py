import paddle
import paddle.nn as nn
import warnings
from collections import OrderedDict

__all__ = ['dropout_mask', 'RNNDropout', 'WeightDropout', 'EmbeddingDropout', 'AWD_LSTM', 'awd_lstm_lm_split',
           'awd_lstm_lm_config', 'awd_lstm_clas_split', 'awd_lstm_clas_config']


#Cell
def dropout_mask(x, sz, p):
    "Return a dropout mask of the same type as `x`, size `sz`, with probability `p` to cancel an element."
    x = paddle.empty(shape=sz)
    x = paddle.full(x.shape,1-p)
    x = paddle.bernoulli(x)
    x = paddle.divide(x,paddle.to_tensor(1-p))
    return x

# Cell
class RNNDropout(nn.Layer):
    "Dropout with probability `p` that is consistent on the seq_len dimension."
    def __init__(self, p=0.5):
        super(RNNDropout,self).__init__() 
        self.p=p

    def forward(self, x):
        if not self.training or self.p == 0.: return x
        return x * dropout_mask(x.detach(), [x.shape[0], 1, *x.shape[2:]], self.p)
    
# Cell
class WeightDropout(nn.Layer):
    "A module that wraps another layer in which some weights will be replaced by 0 during training."

    def __init__(self, module, weight_p, layer_names='weight_hh_l0'):
        super(WeightDropout,self).__init__()
        self.module,self.weight_p,self.layer_names = module,weight_p,[layer_names]
        for layer in self.layer_names:
            #Makes a copy of the weights of the selected layers.
            w = getattr(self.module, layer)
            delattr(self.module, layer)
            parameter = paddle.create_parameter(shape=w.shape,dtype=str(w.numpy().dtype),default_initializer=nn.initializer.Assign(w))
            self.add_parameter(f'{layer}_raw',parameter)
            setattr(self.module, layer, w.clone())
            if isinstance(self.module, (nn.RNNCellBase,nn.layer.rnn.RNNBase)):
                self.module.flatten_parameters = self._do_nothing

    def _setweights(self):
        "Apply dropout to the raw weights."
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            if self.training: w = nn.functional.dropout(raw_w,p=self.weight_p)
            else: w = raw_w.clone()
            setattr(self.module, layer, w)

    def forward(self, *args):
        self._setweights()
        with warnings.catch_warnings():
            # To avoid the warning that comes because the weights aren't flattened.
            warnings.simplefilter("ignore", category=UserWarning)
            return self.module(*args)

    def reset(self):
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            setattr(self.module, layer, raw_w.clone())
        if hasattr(self.module, 'reset'): self.module.reset()

    def _do_nothing(self): pass
    
#Cell
class EmbeddingDropout(nn.Layer):
    "Apply dropout with probability `embed_p` to an embedding layer `emb`."

    def __init__(self, emb, embed_p):
        super(EmbeddingDropout,self).__init__()
        self.emb,self.embed_p = emb,embed_p

    def forward(self, words, scale=None):
        if self.training and self.embed_p != 0:
            size = (self.emb.weight.shape[0],1)
            mask = dropout_mask(self.emb.weight.detach(), size, self.embed_p)
            masked_embed = self.emb.weight * mask
        else: masked_embed = self.emb.weight
        if scale: masked_embed.mul_(scale)
        padding_idx = self.emb._padding_idx
        if padding_idx is None:padding_idx = -1
        return nn.functional.embedding(words.astype("int64"), masked_embed, padding_idx, self.emb._sparse)

# Cell
class AWD_LSTM(nn.Layer):
    "AWD-LSTM inspired by https://arxiv.org/abs/1708.02182"
    initrange=0.1

    def __init__(
        self,
        vocab_sz,
        emb_sz,
        n_hid,
        n_layers,
        pad_token=1,
        hidden_p=0.2,
        input_p=0.6,
        embed_p=0.1,
        output_p = 0.1,
        weight_p=0.5,
        bidir=False,
        tie_weights=False,
        bias=True):
        super(AWD_LSTM,self).__init__()
        self.emb_sz = emb_sz
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.pad_token = pad_token
        self.bs = 1
        self.n_dir = 2 if bidir else 1
        self.encoder = nn.Embedding(vocab_sz, emb_sz, padding_idx=pad_token)
        self.encoder_dp = EmbeddingDropout(self.encoder, embed_p)
        self.rnns = nn.LayerList([self._one_rnn(emb_sz if l == 0 else n_hid, (n_hid if l != n_layers - 1 else emb_sz)//self.n_dir,
                                                 bidir, weight_p, l) for l in range(n_layers)])
        self.encoder.weight.set_value(paddle.uniform(shape=self.encoder._size,min=-self.initrange,max=self.initrange))
        self.input_dp = RNNDropout(input_p)
        self.hidden_dps = nn.LayerList([RNNDropout(hidden_p) for l in range(n_layers)])
        # enc = self.encoder if tie_weights else None
        # self.decoder = LinearDecoder(vocab_sz,emb_sz,output_p,tie_encoder=enc,bias=bias)
        self.reset()

    def forward(self, inp, from_embeds=False):
        bs,sl = inp.shape[:2] if from_embeds else inp.shape
        if bs!=self.bs: self._change_hidden(bs)

        output = self.input_dp(inp if from_embeds else self.encoder_dp(inp))
        new_hidden = []
        for l, (rnn,hid_dp) in enumerate(zip(self.rnns, self.hidden_dps)):
            output, new_h = rnn(output, self.hidden[l])
            new_hidden.append((new_h[0].detach(), new_h[1].detach()))
            if l != self.n_layers - 1: output = hid_dp(output)

        # for l in range(len(self.rnns)):
        #     output, new_h = self.rnns[l](output, self.hidden[l])
        #     new_hidden.append(new_h)
        #     if l != self.n_layers - 1: output = self.hidden_dps[l](output)

        self.hidden = new_hidden
        return output

    def _change_hidden(self, bs):
        self.hidden = [self._change_one_hidden(l, bs) for l in range(self.n_layers)]
        self.bs = bs

    def _one_rnn(self, n_in, n_out, bidir, weight_p, l):
        "Return one of the inner rnn"
        # direct = "bidirectional" if bidir else "forward"
        # lstm = nn.LSTM(n_in, n_out, 1, time_major=False, direction=direct)
        # rnn = nn.LayerList()
        # #lstm下paddle会比torch多出'0.cell.weight_ih', '0.cell.weight_hh', '0.cell.bias_ih', '0.cell.bias_hh'4个参数，故手动构造
        # for layer in lstm.state_dict().keys():
        #     if layer in ['weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0']:
        #         if hasattr(lstm,layer):
        #             w = getattr(lstm,layer)
        #             parameter = paddle.create_parameter(shape=w.shape,dtype=str(w.numpy().dtype),default_initializer=nn.initializer.Assign(w))
        #             rnn.add_parameter(layer,parameter)
        # return WeightDropout(rnn, weight_p)

        direct = "bidirectional" if bidir else "forward"
        rnn = nn.LSTM(n_in, n_out, 1, time_major=False, direction=direct)
        return WeightDropout(rnn, weight_p)

    def _one_hidden(self, l):
        "Return one hidden state"
        nh = (self.n_hid if l != self.n_layers - 1 else self.emb_sz) // self.n_dir
        s =paddle.zeros(shape=[self.n_dir, self.bs, nh])
        return (s,s)

    def _change_one_hidden(self, l, bs):
        if self.bs < bs:
            nh = (self.n_hid if l != self.n_layers - 1 else self.emb_sz) // self.n_dir
            s = paddle.zeros(shape=[self.n_dir, bs-self.bs, nh])
            return tuple(paddle.concat([h, s], axis=1) for h in self.hidden[l])
        if self.bs > bs: return (self.hidden[l][0][:,:bs], self.hidden[l][1][:,:bs])
        return self.hidden[l]

    def reset(self):
        "Reset the hidden states"
        [r.reset() for r in self.rnns if hasattr(r, 'reset')]  
        self.hidden = [self._one_hidden(l) for l in range(self.n_layers)]

     
# Cell
def awd_lstm_lm_split(model):
    "Split a RNN `model` in groups for differential learning rates."
    groups = [nn.Sequential(rnn, dp) for rnn, dp in zip(model[0].rnns, model[0].hidden_dps)]
    groups = [groups + [nn.Sequential(model[0].encoder, model[0].encoder_dp, model[1])]]
    return [p for p in groups.parameters()]

# Cell
awd_lstm_lm_config = dict(emb_sz=400, n_hid=1152, n_layers=3, pad_token=1, bidir=False, output_p=0.1,
                          hidden_p=0.15, input_p=0.25, embed_p=0.02, weight_p=0.2, tie_weights=True, out_bias=True)

# Cell
def awd_lstm_clas_split(model):
    "Split a RNN `model` in groups for differential learning rates."
    groups = [nn.Sequential(model[0].module.encoder, model[0].module.encoder_dp)]
    groups += [nn.Sequential(rnn, dp) for rnn, dp in zip(model[0].module.rnns, model[0].module.hidden_dps)]
    groups = [groups + [model[1]]]
    return [p for p in groups.parameters()]

# Cell
awd_lstm_clas_config = dict(emb_sz=400, n_hid=1152, n_layers=3, pad_token=1, bidir=False, output_p=0.4,
                            hidden_p=0.3, input_p=0.4, embed_p=0.05, weight_p=0.5)
