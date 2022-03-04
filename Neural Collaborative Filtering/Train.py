from Dataset import NCFDataSet
from Model import Model
from NeuralMatrixModel import MF
from torch.utils.data import DataLoader, random_split
import torch
import math
import numpy as np


dataset = NCFDataSet('indexCustomersLabeled200.csv')
train, valid = random_split(dataset, [550000, 50000])
# dataset = NCFDataSet('dummy.csv')
# train, valid = random_split(dataset, [5, 2])
print(len(dataset), ' data len')
train_loader = DataLoader(dataset=train,
                          batch_size=128,
                          shuffle=True)


valid_loader = DataLoader(dataset=valid,
                          batch_size=128,
                          shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
n_customers = 51527
n_items = 19519
ncfModel = MF(n_customers, n_items)
ncfModel.to(device)

optimizer = torch.optim.Adam(ncfModel.parameters())
loss_fn = torch.nn.BCELoss()
loss_fn.to(device)

num_epochs = 100
total_samples = len(dataset)
n_iterations = math.ceil(total_samples / 4)
print("Allocated:", round(torch.cuda.memory_allocated(0) / 1243, 1), "GB")
min_valid_loss = np.inf


def validation_set():
    target = ncfModel(customers.int(), items.int())
    target = torch.squeeze(target)
    return loss_fn(target, labels.float())


def train_step():
    optimizer.zero_grad()
    outputs = ncfModel(customers.int(), items.int())
    outputs = torch.squeeze(outputs)
    loss = loss_fn(outputs, labels.float())
    loss.backward()
    optimizer.step()
    return loss.item()


for epoch in range(num_epochs):
    running_loss = 0
    for i, (customers, items, price, labels) in enumerate(train_loader):
        if device == 'cuda':
            customers, items, price, labels = customers.cuda(), items.cuda(), price.cuda(), labels.cuda()
        running_loss += train_step()
        if i % 50 == 1:
            last_loss = running_loss / i  # loss per batch
            print('batch {} loss: {}'.format(i + 1, last_loss))

    valid_loss = 0.0
    ncfModel.eval()  # Optional when not using Model Specific layer
    for customers, items, price, labels in valid_loader:
        if device == 'cuda':
            customers, items, price, labels = customers.cuda(), items.cuda(), price.cuda(), labels.cuda()
        valid_loss += validation_set()

    print(f'Epoch {epoch + 1} \t\t Training Loss: {running_loss / len(train_loader)} \t\t Validation Loss: {valid_loss / len(valid_loader)}')
    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
        torch.save(ncfModel.state_dict(), 'saved_model.pth')


