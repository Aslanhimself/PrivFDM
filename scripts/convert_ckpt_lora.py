import sys
import torch


ckpt = torch.load(sys.argv[1])
sd = ckpt["state_dict"]
keys = list(sd.keys())
for key in keys:
    if "qkv." in key:
        new_key = key.replace("qkv.", "qkv.conv.")
    elif "qkv" in key:
        new_key = key.replace("qkv", "qkvconv")
    else:
        new_key = None

    if new_key is not None:
        sd[new_key] = sd[key]
        del sd[key]

torch.save(ckpt, sys.argv[2])
