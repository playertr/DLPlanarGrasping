import sys
sys.path.insert(0, '/home/tim/Classes/CS535/PlanarGrasping/')
# sys.path.insert(0, '/home/tim/Classes/CS535/PlanarGrasping/network/')

import pickle
import matplotlib.pyplot as plt

from network.model import net, data_loader
from network import utils
import numpy as np

from IPython import display

import torch

import torch.optim as optim

json_path = '/home/tim/Classes/CS535/PlanarGrasping/network/experiments/sdf_cnn/params.json'
params = utils.Params(json_path)
params.cuda = torch.cuda.is_available()

data_dir = '/home/tim/Classes/CS535/PlanarGrasping/network/data'

# fetch dataloaders
dataloaders = data_loader.fetch_dataloader(
    ['train', 'val'],data_dir, params)
train_dl = dataloaders['train']
val_dl = dataloaders['val']

# Define the model and optimizer
model = net.Net(params).cuda()
optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

model.eval()

# reload weights from restore_file if specified
weight_file = '/home/tim/Classes/CS535/PlanarGrasping/network/experiments/sdf_cnn/best.pth.tar'
utils.load_checkpoint(weight_file, model, optimizer)
print()


for i, (train_batch, labels_batch) in enumerate(val_dl):
    scan_pts = train_batch[0][:-1,:].numpy()
    print(scan_pts[0,:])
    if i > 5:
        break