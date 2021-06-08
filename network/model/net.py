"""Defines the neural network, losss function and metrics"""

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, params):
        """
        We define an convolutional network. The components required are:

        Args:
            params: (Params) contains num_channels
        """
        super(Net, self).__init__()

        # Input has shape (101, 2)
        # 2 fully connected layers to transform the output of the convolution layers to the final output
        self.fc1 = nn.Linear(202, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 512)
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 512)
        self.batch_norm2 = nn.BatchNorm1d(512)
        self.fc5 = nn.Linear(512, 256)
        self.batch_norm3 = nn.BatchNorm1d(256)
        self.fc6 = nn.Linear(256, 128)
        self.batch_norm4 = nn.BatchNorm1d(128)
        self.fc7 = nn.Linear(128, 1)
        self.dropout_rate = params.dropout_rate
        self.dropout_layer = nn.Dropout2d(p=self.dropout_rate)

    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 3 x 64 x 64 .

        Returns:
            out: (Variable) dimension batch_size x 6 with the log probabilities for the labels of each image.

        Note: the dimensions after each step are provided
        """
        s = s.view(-1, 202)
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        s = self.dropout_layer(s)
        residual = s
        s = F.relu(self.fc3(s))
        s = self.batch_norm1(s)
        s = F.relu(self.fc4(s))
        s = self.batch_norm2(s)
        s = F.relu(self.fc5(s))
        s = self.batch_norm3(s)
        s += residual
        s = F.relu(self.fc6(s))
        s = self.batch_norm4(s)
        s = self.fc7(s)
        # s = F.relu(self.fc1(s))
        # s = F.relu(self.fc2(s))
        # s = F.relu(self.fc3(s))
        s = s.squeeze()
        return s

# We are able to define arbitrary metrics for our models, beyond L2 loss, and
# print them out at each epoch. For an example, see the starter code:
# https://github.com/cs230-stanford/cs230-code-examples/tree/master/pytorch/vision

# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    # 'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}
