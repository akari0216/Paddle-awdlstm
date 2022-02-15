import numpy as np 
import paddle.nn.functional as F
import paddle

fwd = np.load("fwd_res.npy")
bwd = np.load("bwd_res.npy")
labels = np.load("labels_res.npy")

print(fwd.shape, bwd.shape, labels.shape)

fwd = F.softmax(paddle.to_tensor(fwd), axis=-1)
bwd = F.softmax(paddle.to_tensor(bwd), axis=-1)
fwd = fwd.numpy()
bwd = bwd.numpy()

merge = (fwd + bwd) / 2.0
merge = np.argmax(merge, axis=-1)

acc =  (merge == labels.flatten()).astype("float32").mean()
print(acc) 

fwd = np.argmax(fwd, axis=-1)
print("fwd",  (fwd == labels.flatten()).astype("float32").mean())

bwd = np.argmax(bwd, axis=-1)
print("bwd",  (bwd == labels.flatten()).astype("float32").mean())

print("fwd -> bwd",   (fwd == bwd).astype("float32").mean())