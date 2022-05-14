import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from data import *
from aug_data import *
from unet import *
from mlp import *
from enet import *

import wandb

wandb.login()

num_epochs = 200
test_split = 0.3
batch_size = 64
SEED = 42
lr = 0.005
dataset = 'aug'
model_name = 'enet'

# MODEL
if model_name == 'unet':
    model = UNet(3,1)
elif model_name == 'mlp':
    model = MLP(786432, 262144, 1)
elif model_name == 'enet':
    model = ENet(1)

if dataset == 'aug':
    data = AugDataset()
else:
    data = Dataset()

# generate indices: instead of the actual data we pass in integers instead
train_indices, test_indices, _, _ = train_test_split(
    range(len(data)),
    data.y,
    test_size=test_split,
    random_state=SEED
)

# generate subset based on indices
train = Subset(data, train_indices)
test = Subset(data, test_indices)
print(len(train))
print(len(test))

# create batches
train_generator = DataLoader(train, batch_size=batch_size, shuffle=True)
test_generator = DataLoader(test, batch_size=batch_size, shuffle=True)
train_size = len(train)
test_size = len(test)

# LOSS
criterion = nn.BCELoss()

# OPTIMIZER
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

# WANDB
config_dict = dict(
    epochs = num_epochs,
    batch_size = batch_size,
    dataset_len = len(data),
    test_split = test_split,
    lr = lr,
    data = dataset,
    model = model_name
)
wandb.init(project="BIA final", entity="ginac",config=config_dict)

# TRAIN AND TEST LOOP
tr_loss = []
ts_loss = []
best = None
for epoch in range(num_epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_generator, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        
        # zero the parameter gradients
        optimizer.zero_grad()
    

        # forward + backward + optimize
        outputs = model(inputs)

        if model_name == 'mlp':
            labels = torch.flatten(labels,1)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    epoch_train_loss = running_loss / (train_size//batch_size+1)
    tr_loss.append(epoch_train_loss)
    print("Epoch:", epoch, "\tTrain Loss:", epoch_train_loss)
    wandb.log({'Epoch Num': epoch+1, 'Train loss': epoch_train_loss})

    if epoch%5 == 0:
        test_loss = 0.0
        for i, data in enumerate(test_generator, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # forward + backward + optimize
            outputs = model(inputs)
            
            if model_name == 'mlp':
                labels = torch.flatten(labels,1)
            
            loss = criterion(outputs, labels)

            test_loss += loss.item()
        epoch_ts_loss = test_loss / (test_size//batch_size+1)
        ts_loss.append(epoch_ts_loss)

        if not best:
            best = epoch_ts_loss
        if epoch_ts_loss < best:
            best = epoch_ts_loss

        print("Test Loss:", epoch_ts_loss)
        wandb.log({'Epoch Num': epoch+1, 'Test loss': epoch_ts_loss, 'Best Test Loss': best})
