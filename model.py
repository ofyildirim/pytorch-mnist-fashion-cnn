import torch
from torch import nn
from sklearn.metrics import accuracy_score  # Will be used while calculating accuracies for training and test sets


class Convolutional_Neural_Network(nn.Module):
    def __init__(self, activation_function, dropout_rate):
        """
        CNN Class with 2 convolutional layers (and 2 2x2 max-pooling after each layer),
        followed by a 2 fully connected linear layers with 256 hidden neurons and 10 output neurons.
        :param activation_function: Which activation function to use (ReLU, ELU, Sigmoid or Tanh)
        :param dropout_rate: Dropout rate
        """
        super(Convolutional_Neural_Network, self).__init__()  # Call the base constructor first!
        # CNN
        # We define first conv. layer with 1 input channel 32 output channels and 5 kernel size
        self.layer1 = nn.Conv2d(1, 32, 5)
        # 2x2 max pooling
        self.max1 = nn.MaxPool2d(2)
        # We define second conv. layer with 32 input channels 64 output channels and 5 kernel size
        self.layer2 = nn.Conv2d(32, 64, 5)
        # 2x2 max pooling
        self.max2 = nn.MaxPool2d(2)
        # Fully connected layer
        # Fully connected linear layer with 1024 inputs and 256 outputs (hidden layers)
        self.fc1 = nn.Linear(1024, 256)
        # Fully connected output layer with 256 inputs and 10 outputs (there are 10 classes)
        self.out = nn.Linear(256, 10)

        # Activation function selection between ReLU, ELU, Tanh and Sigmoid
        if activation_function.lower() == 'relu':
            self.act = nn.ReLU()
        elif activation_function.lower() == 'elu':
            self.act = nn.ELU()
        elif activation_function.lower() == 'tanh':
            self.act = nn.Tanh()
        elif activation_function.lower() == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            raise AttributeError

        # Dropout rate (will be assigned to 0 when not used)
        self.dropout = nn.Dropout(dropout_rate)
        # Softmax instance created to use in output layer
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward propagation
        :param x:  X is the input batch
        :return: returns outputs
        """
        # Getting the batch size to use in reshaping
        batch_size = x.shape[0]
        # Transform the tensor shape to fit
        x.view(-1, 28, 28)
        x = self.layer1(x)
        x = self.act(self.max1(x))
        x = self.layer2(x)
        x = self.act(self.max2(x))
        # Reshaping with respect to batch size
        x = x.reshape(batch_size, -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.out(x)
        x = self.softmax(x)
        return x  # Forward pass completed!


def weights_init(m):
    """
    Function for initialising weights with Xavier
    :param m: model
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        # Weights initialised with Xavier
        torch.nn.init.xavier_uniform_(m.weight)


def train(train_loader, test_loader, activation_function, learning_rate, dropout_rate, device):
    """
    Function for training and testing.
    Algorithm
        1. Initialising of CNN, loss function, optimizer and necessary lists
        2. For each epoch
            3. For each batch in training loader
                - Performing forward and backward propagations, classifying inputs, calculating accuracy and loss values
            4. For each batch in test loader
                - Performing forward propagation, classifying inputs, calculating accuracy and loss values

    :param train_loader: Includes training batches
    :param test_loader: Includes test batches
    :param activation_function: Which activation function to use (ReLU, ELU, Sigmoid or Tanh)
    :param learning_rate: Learning rate
    :param dropout_rate: Dropout rate
    :param device: Which device to use, GPU or CPU
    :return: Created model and loss and accuracy lists for training and test
    """
    # Instance of CNN created with specified activation function and dropout rate
    model = Convolutional_Neural_Network(activation_function, dropout_rate)
    # Weights initialised with Xavier
    model.apply(weights_init)
    # Telling model the device it will be run
    model.to(device)
    # Loss function defined
    my_loss_criterion = nn.CrossEntropyLoss()
    # Optimizer initialised (Simple Gradient Descent)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # Number of epochs
    epochs = 50
    # List initialisation to store training losses in each epoch
    train_epoch_loss = []
    # List initialisation to store training accuracies in each epoch
    train_epoch_accuracy = []
    # List initialisation to store test losses in each epoch
    test_epoch_loss = []
    # List initialisation to store test accuracies in each epoch
    test_epoch_accuracy = []
    for epoch in range(0, epochs):
        # Telling model to train, so parameters will become modifiable
        model.train()
        # initialising loss variable for training
        train_running_loss = 0
        # initialising accuracy variable for training
        train_accuracy = 0
        for i, (inp, targ) in enumerate(train_loader):
            inp, targ = inp.to(device), targ.to(device)
            # Reset gradients
            optimizer.zero_grad()
            # Calculate outputs
            out = model.forward(inp)
            # Calculate Loss
            loss = my_loss_criterion(out, targ)
            # Perform backpropagation
            loss.backward()
            # Apply changes
            optimizer.step()
            # Add current loss to running loss
            train_running_loss += loss.item()
            # Classify by taking the max value from softmax output
            classes = out.cpu().data.numpy().argmax(axis=1)
            # Calculate current accuracy and add it to accuracy variable
            train_accuracy += accuracy_score(targ.cpu().data.numpy(), classes)
        # Divide by number of batches to get average loss value
        train_running_loss /= (i + 1)
        print(f'Epoch {epoch + 1}, training loss: {train_running_loss}')
        # Divide by number of batches to get average accuracy value
        train_accuracy /= (i + 1)
        print(f'Epoch {epoch + 1}, training accuracy: {train_accuracy}')
        # Append loss to a list to use it in visualisation afterwards
        train_epoch_loss.append(train_running_loss)
        # Append accuracy to a list to use it in visualisation afterwards
        train_epoch_accuracy.append(train_accuracy)
        # Telling model to evaluate, so parameters will become immutable
        model.eval()
        # initialising loss variable for test
        test_running_loss = 0
        # initialising accuracy variable for test
        test_accuracy = 0
        for i, (inp, targ) in enumerate(test_loader):
            inp, targ = inp.to(device), targ.to(device)
            # Calculate outputs
            out = model.forward(inp)
            # Calculate Loss
            loss = my_loss_criterion(out, targ)
            # Add current loss to running loss
            test_running_loss += loss.item()
            # Classify by taking the max value from softmax output
            classes = out.cpu().data.numpy().argmax(axis=1)
            # Calculate current accuracy and add it to accuracy variable
            test_accuracy += accuracy_score(targ.cpu().data.numpy(), classes)
        # Divide by number of batches to get average loss value
        test_running_loss /= (i + 1)
        print(f'Epoch {epoch + 1}, test loss: {test_running_loss}')
        # Divide by number of batches to get average accuracy value
        test_accuracy /= (i + 1)
        print(f'Epoch {epoch + 1}, test accuracy: {test_accuracy}')
        # Append loss to a list to use it in visualisation afterwards
        test_epoch_loss.append(test_running_loss)
        # Append accuracy to a list to use it in visualisation afterwards
        test_epoch_accuracy.append(test_accuracy)
    # Return created model and loss and accuracy lists for training and test
    return model, train_epoch_accuracy, train_epoch_loss, test_epoch_accuracy, test_epoch_loss
