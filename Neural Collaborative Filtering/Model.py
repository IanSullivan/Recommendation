import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, num_inputs):
        super(ResBlock, self).__init__()
        self.dense = nn.Linear(num_inputs, num_inputs)
        self.relu = nn.ReLU()
        self.batchNorm1 = nn.BatchNorm1d(num_inputs)
        self.dropout1 = nn.Dropout(0.3)

    def forward(self, x):
        residual = x.cuda()
        x = self.dense(x).cuda()
        x = self.relu(x).cuda()
        x = self.batchNorm1(x).cuda()
        x = self.dropout1(x).cuda()
        out = x + residual.cuda()
        return out


class Model(nn.Module):
    def __init__(self, n_customers, n_items):
        super(Model, self).__init__()
        n_items = 19519
        n_customers = 51527
        embedding_dim = 64
        self.embedding_item = torch.nn.Embedding(n_items, embedding_dim)
        self.embedding_customer = torch.nn.Embedding(n_customers, embedding_dim)
        self.block = ResBlock(embedding_dim)
        self.block2 = ResBlock(embedding_dim)
        self.layer = nn.Linear(embedding_dim, 1)
        self.activation = nn.Sigmoid()

    # forward propagate input
    def forward(self, customer, item, price):
        item_latent = self.embedding_item(item.cuda()).cuda()
        customer_latent = self.embedding_customer(customer.cuda()).cuda()
        vector = torch.add(item_latent, customer_latent).cuda()
        X = self.block(vector).cuda()
        X = self.block2(X).cuda()
        X = self.layer(X).cuda()
        X = self.activation(X).cuda()
        return X


