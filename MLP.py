import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import torch.optim as optim
from torch.utils.data import TensorDataset
from torchvision.transforms import transforms
from torchvision.datasets import MNIST

import matplotlib.pyplot as plt
import time

train = MNIST('./data', train=True, download=True, transform=transforms.ToTensor())

test = MNIST('./data', train=False, download=True, transform=transforms.ToTensor())

cuda = torch.cuda.is_available()

dataloader_args = dict(shuffle=True, batch_size=256, num_workers=4, pin_memory=True) if cuda \
    else dict(shuffle=True, batch_size=64)
train_loader = dataloader.DataLoader(train, **dataloader_args)
test_loader = dataloader.DataLoader(test, **dataloader_args)

train_data = train.train_data
train_data = train.transform(train_data.numpy())

print('[Train]')
print('Numpy of shape ', train.train_data.cpu().numpy().shape)
print('Tensor of shape ', train.train_data.size())
print("Mean of train data", torch.mean(train_data))
print("Maximum of train data", torch.max(train_data))
print("Minimum of train data", torch.min(train_data))

#plt.imshow(train.train_data.cpu().numpy()[0], cmap='gray')
# plt.show()
#plt.imshow(train.train_data.cpu().numpy()[1], cmap='gray')

# plt.show()


class simple_mlp(nn.Module):
    def __init__(self, size_list):
        super(simple_mlp, self).__init__()
        layers = []
        self.size_list = size_list
        for i in range(len(size_list) - 2):
            layers.append(nn.Linear(size_list[i], size_list[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(size_list[-2], size_list[-1]))
        self.net = nn.Sequential(*layers)

    def forwards(self, x):
        x = x.view(-1, self.size_list[0])
        return self.net(x)


model = simple_mlp([784, 256, 10])
criterior = nn.CrossEntropyLoss()
print(model)
optimizer = optim.Adam(model.parameters())
device = torch.device("cuda" if cuda else "cpu")


def train_epoch(model, train_loader, criterior, optimizer):
    model.train()
    model.to(device)
    i = 0
    running_loss = 0.0
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        target = target.long().to(device)

        output = model.forwards(data)
        i = i + 1
        loss = criterior(output, target)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()
    end_time = time.time()
    running_loss /= len(train_loader)
    print('Training_loss:', running_loss, 'Time:', end_time - start_time, 's')
    return running_loss


train_loss = train_epoch(model, train_loader, criterior, optimizer)


def test_model(model, test_loader, criterior):
    with torch.no_grad():
        model.eval()
        model.to(device)

        running_loss = 0.0
        correct_prediction = 0.0
        total_pred = 0.0
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = target.long().to(device)

            output = model.forwards(data)

            _, predicted = torch.max(output.data, 1)
            predicted = predicted
            total_pred += target.size(0)
            correct_prediction += (predicted == target).sum().item()

            loss = criterior(output, target).detach()
            running_loss += loss.item()

    running_loss /= len(test_loader)
    acc = (correct_prediction / total_pred) * 100.0
    print('Testing Loss: ', running_loss)
    print('Testing Accuracy: ', acc, '%')
    return running_loss, acc


train_loss1 = []
test_loss1 = []
epochs = 10
test_acc1 = []
for i in range(epochs):
    train_loss = train_epoch(model, train_loader, criterior, optimizer)
    test_loss, test_acc = test_model(model, test_loader, criterior)
    train_loss1.append(train_loss)
    test_loss1.append(test_loss)
    test_acc1.append(test_acc)
    print('=' * 20)

plt.title('Training_Loss')
plt.xlabel("Epoch")
plt.ylabel("Train_loss")
plt.plot(train_loss1)
plt.show()


plt.title('Test_Loss')
plt.xlabel("Epoch")
plt.ylabel("Test_loss")
plt.plot(test_loss1)
plt.show()
