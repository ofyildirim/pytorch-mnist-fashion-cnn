import torch
import torchvision
import torchvision.transforms as transforms
from model import *
from create_plots import *
# Training set imported
train_set = torchvision.datasets.FashionMNIST(root=".", train=True, download=True, transform=transforms.ToTensor())
# Test set imported
test_set = torchvision.datasets.FashionMNIST(root=".", train=False, download=True, transform=transforms.ToTensor())
# Train loader created with 32 batch size
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=False)
# Test loader created with 32 batch size
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)
# Manual seed used will return deterministic random numbers
torch.manual_seed(0)
# If you are using CuDNN , otherwise you can just ignore
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device decision made, GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Activation function = ReLU, learning rate = 0.1, dropout rate = 0
model, train_accuracy, train_loss, test_accuracy, test_loss = train(train_loader, test_loader, 'relu', 0.1, 0, device)

create_plots(train_accuracy, train_loss, test_accuracy, test_loss)
