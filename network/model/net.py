"""Defines the neural network, losss function and metrics"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Tnet(nn.Module):
   def __init__(self, k=2):
      super().__init__()
      self.k=k

    #   self.conv1 = nn.Conv1d(k,64,1)
    #   self.conv2 = nn.Conv1d(64,128,1)
    #   self.conv3 = nn.Conv1d(128,1024,1)

    #   self.fc1 = nn.Linear(1024,512)
    #   self.fc2 = nn.Linear(512,256)
    #   self.fc3 = nn.Linear(256,k*k)

    #   self.bn1 = nn.BatchNorm1d(64)
    #   self.bn2 = nn.BatchNorm1d(128)
    #   self.bn3 = nn.BatchNorm1d(1024)
    #   self.bn4 = nn.BatchNorm1d(512)
    #   self.bn5 = nn.BatchNorm1d(256)
      SCALE=16
      self.conv1 = nn.Conv1d(k,int(64/SCALE),1)
      self.conv2 = nn.Conv1d(int(64/SCALE),int(128/SCALE),1)
      self.conv3 = nn.Conv1d(int(128/SCALE),int(1024/SCALE),1)

      self.fc1 = nn.Linear(int(1024/SCALE),int(512/SCALE))
      self.fc2 = nn.Linear(int(512/SCALE),int(256/SCALE))
      self.fc3 = nn.Linear(int(256/SCALE),k*k)

      self.bn1 = nn.BatchNorm1d(int(64/SCALE))
      self.bn2 = nn.BatchNorm1d(int(128/SCALE))
      self.bn3 = nn.BatchNorm1d(int(1024/SCALE))
      self.bn4 = nn.BatchNorm1d(int(512/SCALE))
      self.bn5 = nn.BatchNorm1d(int(256/SCALE))
      
       

   def forward(self, input):
      # input.shape == (bs,n,3)
      bs = input.size(0)
      xb = F.relu(self.bn1(self.conv1(input)))
      xb = F.relu(self.bn2(self.conv2(xb)))
      xb = F.relu(self.bn3(self.conv3(xb)))
      pool = nn.MaxPool1d(xb.size(-1))(xb)
      flat = nn.Flatten(1)(pool)
      xb = F.relu(self.bn4(self.fc1(flat)))
      xb = F.relu(self.bn5(self.fc2(xb)))
      
      #initialize as identity
      init = torch.eye(self.k, requires_grad=True).repeat(bs,1,1)
      if xb.is_cuda:
        init=init.cuda()
      matrix = self.fc3(xb).view(-1,self.k,self.k) + init
      return matrix


class Transform(nn.Module):
   def __init__(self):
        super().__init__()
        self.input_transform = Tnet(k=2)
        # self.feature_transform = Tnet(k=64)

        # self.conv1 = nn.Conv1d(2,64,1)
        # self.conv2 = nn.Conv1d(64,128,1)
        # self.conv3 = nn.Conv1d(128,1024,1)
       
        # self.bn1 = nn.BatchNorm1d(64)
        # self.bn2 = nn.BatchNorm1d(128)
        # self.bn3 = nn.BatchNorm1d(1024)
        SCALE=2
        self.feature_transform = Tnet(k=int(64/SCALE))
        self.conv1 = nn.Conv1d(2,int(64/SCALE),1)
        self.conv2 = nn.Conv1d(int(64/SCALE),int(128/SCALE),1)
        self.conv3 = nn.Conv1d(int(128/SCALE),int(1024/SCALE),1)
       
        self.bn1 = nn.BatchNorm1d(int(64/SCALE))
        self.bn2 = nn.BatchNorm1d(int(128/SCALE))
        self.bn3 = nn.BatchNorm1d(int(1024/SCALE))
       
   def forward(self, input):
        matrix2x2 = self.input_transform(input)
        # batch matrix multiplication
        xb = torch.bmm(torch.transpose(input,1,2), matrix2x2).transpose(1,2)

        xb = F.relu(self.bn1(self.conv1(xb)))

        matrix64x64 = self.feature_transform(xb)
        xb = torch.bmm(torch.transpose(xb,1,2), matrix64x64).transpose(1,2)

        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = self.bn3(self.conv3(xb))
        xb = nn.MaxPool1d(xb.size(-1))(xb)
        output = nn.Flatten(1)(xb)
        return output, matrix2x2, matrix64x64


class PointNet(nn.Module):
    def __init__(self, classes = 1):
        super().__init__()
        self.transform = Transform()
        # self.fc1 = nn.Linear(1024, 512)
        # self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256, classes)
        

        # self.bn1 = nn.BatchNorm1d(512)
        # self.bn2 = nn.BatchNorm1d(256)

        SCALE=2
        self.fc1 = nn.Linear(int(1024/SCALE), int(512/SCALE))
        self.fc2 = nn.Linear(int(512/SCALE), int(256/SCALE))
        self.fc3 = nn.Linear(int(256/SCALE), classes)
        

        self.bn1 = nn.BatchNorm1d(int(512/SCALE))
        self.bn2 = nn.BatchNorm1d(int(256/SCALE))


        self.dropout = nn.Dropout(p=0)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 2, 101)
        xb, matrix2x2, matrix64x64 = self.transform(x)
        xb = F.relu(self.bn1(self.fc1(xb)))
        xb = F.relu(self.bn2(self.dropout(self.fc2(xb))))
        output = self.fc3(xb)
        return output, matrix2x2, matrix64x64


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
        
        self.conv1 = nn.Conv1d(2, 16, 3, padding=1)
        self.conv2 = nn.Conv1d(16, 16, 3, padding=1)
        
        self.conv3 = nn.Conv1d(8, 8, 3, padding=1)
        
        self.fc1 = nn.Linear(100, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.batch_norm2 = nn.BatchNorm1d(512)
        self.batch_norm3 = nn.BatchNorm1d(256)
        self.batch_norm4 = nn.BatchNorm1d(128)
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
        s = s.view(-1, 2, 101)
        
        s = F.relu(self.conv1(s))
        s = F.relu(self.conv2(s))
        s = F.max_pool2d(s, 2)
        
        s = F.relu(self.conv3(s))
        s = F.max_pool2d(s, 2)
        s = s.view(-1, self.num_flat_features(s))

        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        s = F.relu(self.fc3(s))
        s = self.fc4(s)
        s = s.squeeze()
        return s

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def pointnetloss(outputs, labels, m2x2, m64x64, alpha = 0.0001):
    criterion = torch.nn.NLLLoss()
    bs=outputs.size(0)
    id2x2 = torch.eye(2, requires_grad=True).repeat(bs,1,1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs,1,1)
    if outputs.is_cuda:
        id2x2=id2x2.cuda()
        id64x64=id64x64.cuda()
    diff2x2 = id2x2-torch.bmm(m2x2,m2x2.transpose(1,2))
    diff64x64 = id64x64-torch.bmm(m64x64,m64x64.transpose(1,2))
    return criterion(outputs, labels) + alpha * (torch.norm(diff2x2)+torch.norm(diff64x64)) / float(bs)


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
    'mae_loss': mae_loss
    # 'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}
