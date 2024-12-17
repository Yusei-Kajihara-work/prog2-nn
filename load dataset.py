import matplotlib.pyplot as plt
import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms


ds_train = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
)

print(f'dataset size: {len(ds_train)}')

image, target = ds_train[1]
print(type(image),target)