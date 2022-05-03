import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from data import *
from unet import *

import wandb

wandb.login()

num_epochs = 100
test_split = 0.3
batch_size = 64
SEED = 42
lr = 0.005

data = UNetDataset()

print(len(data))

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

# MODEL
model = UNet(4,1)

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
    lr = lr
)
wandb.init(project="BIA final", entity="ginac",config=config_dict)

# TRAIN AND TEST LOOP
tr_loss = []
val_loss = []
for epoch in range(num_epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_generator, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    epoch_train_loss = running_loss / (train_size//batch_size+1)
    tr_loss.append(epoch_train_loss)
    print("Epoch:", epoch, "\tTrain Loss:", epoch_train_loss)
    wandb.log({'Epoch Num': epoch+1, 'Train loss': epoch_train_loss})

    test_loss = 0.0
    for i, data in enumerate(test_generator, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
    epoch_val_loss = test_loss / (test_size//batch_size+1)
    val_loss.append(epoch_val_loss)
    print("Test Loss:", epoch_val_loss)
    wandb.log({'Epoch Num': epoch+1, 'Test loss': epoch_val_loss})
