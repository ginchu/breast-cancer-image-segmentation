
import numpy as np
import sklearn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from fcn_data_aug import *
from fcn import *
import wandb
wandb.login()

num_epochs = 20
test_split = 0.3
batch_size = 8
SEED = 42
lr = 0.005

data = FCNDataset()
print(len(data))

# generate indices
train_indices, test_indices, _, _ = train_test_split(
    range(len(data)),
    data.y,
    test_size=test_split,
    random_state=SEED
)

train = Subset(data, train_indices)
test = Subset(data, test_indices)
print(len(train))
print(len(test))

train_generator = DataLoader(train, batch_size=batch_size, shuffle=True)
test_generator = DataLoader(test, batch_size=batch_size, shuffle=True)
train_size = len(train)
test_size = len(test)

# MODEL
model = FCN_res101(4,8)
#model = FCN_res50(4,8)
#model = FCN_res34(4,8)
#model = FCN_res18(4,8)

# LOSS

loss = torch.nn.CrossEntropyLoss()

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
wandb.init(project="BIA final project Augmented+ FCN", entity="ananyab",config=config_dict)

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
        #print("check target tensor: ", outputs.unsqueeze(1).type(torch.FloatTensor))
        #loss = criterion(outputs.unsqueeze(1).type(torch.FloatTensor), labels.unsqueeze(1).type(torch.FloatTensor))
        #loss.backward()
        loss_ce = loss(outputs, labels.squeeze(1).type(torch.LongTensor))
        loss_ce.backward()
        optimizer.step()

        # print statistics
        running_loss += loss_ce.item()
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
        #loss = criterion(outputs.unsqueeze(1).type(torch.FloatTensor), labels.unsqueeze(1).type(torch.FloatTensor))
        loss_ce = loss(outputs, labels.squeeze(1).type(torch.LongTensor))
        test_loss += loss_ce.item()
    epoch_val_loss = test_loss / (test_size//batch_size+1)
    val_loss.append(epoch_val_loss)
    print("Test Loss:", epoch_val_loss)
    wandb.log({'Epoch Num': epoch+1, 'Test loss': epoch_val_loss})
