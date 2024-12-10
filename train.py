import time
from unittest.main import MODULE_EXAMPLES

import matplotlib.pyplot as plt

import torch
from torchvision import datasets 
import torchvision.transforms.v2 as transforms

import models

ds_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32,scale=True)
])

ds_train = datasets.FashionMNIST(
    root='data',
    train=True,
    transform=ds_transform
)
ds_test = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ds_transform
)
batch_size =64
dataloader_train = torch.utils.data.DataLoader(
    ds_train,
    batch_size=batch_size,
    shuffle=True
)
dataloader_test = torch.utils.data.DataLoader(
    ds_test,
    batch_size=batch_size
)

for image_batch,label_bach in dataloader_test:
    print(image_batch.shape)
    print(label_bach.shape)
    break

model = models.MyModel()

acc_test = models.test_accuracy(model,dataloader_test)
print(f'test accuracy:{acc_test*100:.3f}%')

model = models.MyModel()
loss_fn = torch.nn.CrossEntropyLoss()

learning_rate = 1e-3
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

n_epochs = 5

for k in range(n_epochs):
    print(f'epoch {k+1}/{n_epochs}')

    models.train(model,dataloader_train,loss_fn,optimizer)

    acc_train = models.test_accuracy(model,dataloader_train)
    print(f'train loss:{loss_train}')
    acc_test = models.test_accuracy(model,dataloader_test)
    print(f'train loss:{loss_train}')

    loss_train = models.test_accuracy(model,dataloader_train)
    print(f'trainaccuracy:{acc_train*100:.3f}%')
    loss_test = models.test_accuracy(model,dataloader_test)
    print(f'trainaccuracy:{acc_train*100:.3f}%')