"""Defines the neural network, losss function and metrics"""

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torch.optim as optim

import sys
sys.path.insert(0, '/home/tim/Classes/CS535/PlanarGrasping/')
import sdf_network.model.net as sdfnet
import grasp_network.utils as utils

class Net(nn.Module):

    def __init__(self, params):
        """
        We define an convolutional network. The components required are:

        Args:
            params: (Params) contains num_channels
        """
        super(Net, self).__init__()

        # Initialize SDFnet
        json_path = '/home/tim/Classes/CS535/PlanarGrasping/sdf_network/experiments/sdf_cnn/params.json'
        sdfparams = utils.Params(json_path)
        self.sdf = sdfnet.Net(sdfparams)
        weight_file = '/home/tim/Classes/CS535/PlanarGrasping/sdf_network/experiments/sdf_cnn/best.pth.tar'
        optimizer = optim.Adam(self.sdf.parameters(), lr=sdfparams.learning_rate)
        utils.load_checkpoint(weight_file, self.sdf, optimizer)
        for param in self.sdf.parameters():
            param.requires_grad = False
        self.sdf.eval()

        # Create a regular grid of 2D sample points
        x = torch.linspace(0, 15, 4)
        y = torch.linspace(-5, 5, 4)
        X, Y = torch.meshgrid(x, y)
        query_pts = torch.column_stack((X.ravel(), Y.ravel())) # Put the grid in an Nx2 matrix
        self.num_query_pts = query_pts.shape[0]
        self.query_pts = torch.flatten(query_pts).cuda()

        # Input has shape (101, 2)
        # 2 fully connected layers to transform the output of the convolution layers to the final output
        # self.conv1 = nn.Conv1d(2, 16, 3, padding=1)
        # self.conv2 = nn.Conv1d(16, 2, 3, padding=1)
        # self.fc1 = nn.Linear(202, 512)
        self.fc1 = nn.Linear(202 + 64, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)
        # self.batch_norm1 = nn.BatchNorm1d(256)
        # self.fc4 = nn.Linear(256, 128)
        # self.batch_norm2 = nn.BatchNorm1d(512)
        # self.fc5 = nn.Linear(128, 1)
        # self.batch_norm3 = nn.BatchNorm1d(256)
        # self.fc6 = nn.Linear(256, 128)
        # self.batch_norm4 = nn.BatchNorm1d(128)
        # self.fc7 = nn.Linear(128, 1)
        # self.dropout_rate = params.dropout_rate
        # self.dropout_layer = nn.Dropout2d(p=self.dropout_rate)



    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 3 x 64 x 64 .

        Returns:
            out: (Variable) dimension batch_size x 6 with the log probabilities for the labels of each image.

        Note: the dimensions after each step are provided
        """
        # evaluate sdf at regularly spaced points
        scan_pts = s[:,:-1,:] # The scan includes everything but the last row

        cnn_vector, m2x2, m64x64 = self.sdf.pointnet1(scan_pts)

        # tpts = torch.tile(scan_pts, (1, 1, self.num_query_pts)) # Copy the scan points many times
        # qpts = self.query_pts.expand(s.shape[0], 1, self.num_query_pts*2)
        # mtx = torch.cat(( # Stack each repeated scan matrix on top of a 2D query pt
        #     tpts,
        #     qpts), dim=-2)

        # # Turn the extra-wide repeated matrix into a 3-d matrix that can be fed
        # # into a nn.Module
        # mtx = torch.swapaxes(mtx, -1, -2)
        # mtx = torch.reshape(mtx, (-1, self.num_query_pts, 2, mtx.shape[-1]))
        # mtx = torch.swapaxes(mtx, -1, -2).contiguous()

        # # mtx is torch.Size([256, 45, 101, 2]), need torch.Size([-1, 101, 2])
        # mtx = mtx.view(-1, 101, 2)
        # dists, matrix2x2, matrix64x64 = self.sdf(mtx)
        # distances = dists.view(-1, self.num_query_pts)

        
        # s = torch.transpose(s, 2, 1)
        # s = F.relu(self.conv1(s))
        # s = F.relu(self.conv2(s))
        s = s.view(-1, 202)
        s = torch.cat((s, cnn_vector), dim=1)

        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        # s = self.dropout_layer(s)
        # residual = s
        # s = F.relu(self.fc3(s))
        s = self.fc3(s)
        # s = self.batch_norm1(s)
        # s = F.relu(self.fc4(s))
        # s = self.batch_norm2(s)
        # s = self.fc5(s)
        # s = self.batch_norm3(s)
        # s += residual
        # s = F.relu(self.fc6(s))
        # s = self.batch_norm4(s)
        # s = self.fc7(s)
        # s = F.relu(self.fc1(s))
        # s = F.relu(self.fc2(s))
        # s = F.relu(self.fc3(s))
        s = s.squeeze()
        return s

def mse_loss(outputs, labels):
    criterion = torch.nn.MSELoss()
    return criterion(torch.Tensor(outputs), torch.Tensor(labels))

def mae_loss(outputs, labels):
    num_examples = outputs.shape[0]
    return np.sum(abs(outputs-labels))/num_examples

def rmse_loss(outputs, labels):
    num_examples = outputs.shape[0]
    return np.sqrt(np.sum((outputs - labels)**2)/num_examples)


# We are able to define arbitrary metrics for our models, beyond L2 loss, and
# print them out at each epoch. For an example, see the starter code:
# https://github.com/cs230-stanford/cs230-code-examples/tree/master/pytorch/vision

# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'rmse_loss': rmse_loss,
    'mae_loss': mae_loss,
    'mse_loss': mse_loss
    # 'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}
