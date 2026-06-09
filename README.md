# SRP
Super Resolution Project

## Run
To run the project, install everything form installation then run main. If you get an exception, just run it again until you have downloaded all tif files. Then it should run without problems from there.

## Installation

### Python
Ideally you should use Python 3.10 or later, but `__future__` imports have been added to support at least Python 3.8.10 as well.

Install all packages (except PyTorch) using pip and `requirements.txt` with this command:
```
pip install -r requirements.txt
```

#### PyTorch
If you have an Nvidia GPU, you can run the `nvidia-smi --query-gpu=compute_cap --format=csv` command which outputs something like this:
```    
compute_cap
6.1
```

You can then cross reference your compute capability (CC) with what CUDA SDK version supports it using the [CUDA Wikipedia page](https://en.wikipedia.org/wiki/CUDA#GPUs_supported). In this case the CC is 6.1, which means that the highest CUDA version supported is 12.0-12.6.

You can then find the install command on the [PyTorch website](https://pytorch.org/get-started/locally/). The command would in thise case:
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

If you don't have an Nvidia GPU, you can install the CPU version of PyTorch like this:
```
pip3 install torch torchvision
```

A successfull installation can be tested by running
```
import torch
print("cuda") if torch.cuda.is_available()  else print("cpu")
```
which should output `cuda`

Note: if torch and pytorch is already installed, it may be necessary to uninstall the current versions before installation. This can be done as follows:
```
pip3 uninstall torch torchvision
```

