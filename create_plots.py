import numpy as np
import matplotlib.pyplot as plt
import os


def create_plots(train_accuracy, train_loss, test_accuracy, test_loss):
    """
    Function to create plots for accuracy and loss values
    :param train_accuracy: Train accuracy list
    :param train_loss: Train loss list
    :param test_accuracy: Test accuracy list
    :param test_loss: Test loss list
    :return:
    """
    plt.subplot(1, 2, 1)
    plt.plot(np.array(train_accuracy) * 100, label='Training Accuracy')
    plt.plot(np.array(test_accuracy) * 100, label='Test Accuracy')
    plt.xlabel('Number of Epochs')
    # Set the y axis label of the current axis.
    plt.ylabel('Accuracy')
    # Set a title of the current axes.
    plt.title('Accuracies')
    # show a legend on the plot
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(test_loss, label='Test Loss')
    plt.xlabel('Number of Epochs')
    # Set the y axis label of the current axis.
    plt.ylabel('Loss')
    # Set a title of the current axes.
    plt.title('Losses')
    # show a legend on the plot
    plt.legend()

    plot_filename = os.path.join(os.getcwd(), 'figures', 'Accuracy_and_Loss.png')
    plt.savefig(plot_filename)
    plt.show()
