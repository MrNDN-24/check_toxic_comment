import torch
import torch.nn as nn
from transformers import AutoModel

class Classify(nn.Module):
    def __init__(self, number_of_category):
        super(Classify, self).__init__()
        self.phobert = AutoModel.from_pretrained('vinai/phobert-base')
        for param in self.phobert.parameters():
            param.requires_grad = False
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.first_function = nn.Linear(768, 512)
        self.second_function = nn.Linear(512, 32)
        self.third_function = nn.Linear(32, number_of_category)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids):
        outputs = self.phobert(input_ids)
        x = self.first_function(outputs[1])
        x = self.relu(x)
        x = self.dropout(x)
        x = self.second_function(x)
        x = self.relu(x)
        x = self.third_function(x)
        x = self.softmax(x)
        return x
