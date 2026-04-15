# SRP
Super Resolution Project

## Installation

### Python
Ideally you should use Python 3.10 or later, but `__future__` imports have been added to support at least Python 3.9 as well.

Install all packages (except PyTorch) using pip and `requirements.txt` using this command:
```
pip install -r requirements.txt
```

#### PyTorch
If you have an Nvidia GPU, you can run the `nvidia-smi` command which outputs something like this:
```    
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 581.80                 Driver Version: 581.80         CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce GTX 1060 6GB  WDDM  |   00000000:01:00.0  On |                  N/A |
| 36%   43C    P0             27W /  120W |    1549MiB /   6144MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

At the top it tells you which CUDA version your GPU supports. In this case, it's 13.0 meaning that you install PyTorch like this ([other versions are found here](https://pytorch.org/get-started/locally/)):
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

If you don't have an Nvidia GPU, you can install the CPU version of PyTorch like this:
```
pip3 install torch torchvision
```