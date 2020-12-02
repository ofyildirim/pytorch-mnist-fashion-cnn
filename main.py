import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from model import *

train_set = torchvision.datasets.FashionMNIST(root=".", train=True, download=True, transform=transforms.ToTensor())

test_set = torchvision.datasets.FashionMNIST(root=".", train=False, download=True, transform=transforms.ToTensor())

training_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=False)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

torch.manual_seed(0)
# If you are using CuDNN , otherwise you can just ignore
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model, train_epoch_loss, train_epoch_accuracy, test_epoch_loss, test_epoch_accuracy = train(training_loader,
                                                                                            test_loader, 'relu', 0.1, 0,
                                                                                            device)

plt.plot(np.array(train_epoch_accuracy) * 100, label='Training Accuracy')
plt.plot(np.array(test_epoch_accuracy) * 100, label='Test Accuracy')
plt.xlabel('Number of Epochs')
# Set the y axis label of the current axis.
plt.ylabel('Accuracy')
# Set a title of the current axes.
plt.title('Accuracies')
# show a legend on the plot
plt.legend()
plt.savefig('ReLU_Activation_Accuracies.png')

plt.plot(train_epoch_loss, label='Training Loss')
plt.plot(test_epoch_loss, label='Test Loss')
plt.xlabel('Number of Epochs')
# Set the y axis label of the current axis.
plt.ylabel('Loss')
# Set a title of the current axes.
plt.title('Losses')
# show a legend on the plot
plt.legend()
plt.savefig('ReLU_Activation_Losses.png')
