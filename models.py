import torch
from torch import nn


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.network = nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10),
        )

    def forward(self,x):
        x = self.flatten(x)
        
        logits = self.network(x)
        return logits
    
    def test(model,dataloader,loss_fn):
        loss_total = 0.0

        model.eval()
        for image_batch,label_batch in dataloader:
            with torch.no_grid():
                logits_batch = model(image_batch)
                
            predict_batch