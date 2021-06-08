import os
import pickle
import numpy as np

from torch.utils.data import Dataset, DataLoader
import sys
sys.path.insert(0, '/home/tim/Classes/CS535/PlanarGrasping/')

class SDFDataset(Dataset):
    """
    A standard PyTorch definition of Dataset that defines the functions __len__ and __getitem__.
    """
    def __init__(self, data_dir):
        """
        Store all examples in memory.

        Args:
            data_dir: (string) directory containing the dataset
        """

        # find file paths in data_dir ending with ".p"
        filenames = os.listdir(data_dir)
        filenames = [os.path.join(data_dir, f) for f in filenames if f.endswith('.p')]

        if len(filenames) != 1:
            print(f"Funny number of data files in this directory. Should be 1, is {len(filenames)}")
        
        f = filenames[0]

        # Load data from 1000 shapes.
        # examples is a list of tuples, one per training shape.
        # Each tuple contains three ndarrays:
        # scan_pts      : an (N_SCAN_POINTS,2) ndarray of coordinates corresponding to the local-frame cartesian locations obtained by casting rays from the robot to the shape. The rays span the entire visible portion of the shape.
        # query_pts     : an (N_SDF_QUERIES, 2) ndarray of uniformly sampled points from the ROBOT_RANGE to sample SDF
        # dists         : the SDF value associated with each query point
        self.examples = pickle.load(open( f, "rb" ))
        self.sdf_queries_per_shape = self.examples[0][1].shape[0]

    def __len__(self):
        # return size of dataset
        
        return len(self.examples) * self.sdf_queries_per_shape

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            input: (Tensor) concatenation of scan_pts with query (x,y)
            distance: (float) corresponding distance from query to surface
        """
        
        shape_idx = int(idx / self.sdf_queries_per_shape)
        pt_idx = idx % self.sdf_queries_per_shape

        scan_pts, query_pts, dists, *_ = self.examples[shape_idx] # See description above.
        query_pt = query_pts[pt_idx]
        dist = dists[pt_idx]
        # We want:
        # INPUT: (101, 2) array of points in the local frame
        # The first 100 are the scan, the last point is the query point
        # OUTPUT: (1,) ndarray of SDF value

        data = np.vstack([
            scan_pts, query_pt
            ])
        return data, dist


def fetch_dataloader(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:
            path = os.path.join(data_dir, "{}".format(split))

            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                dl = DataLoader(SDFDataset(path), batch_size=params.batch_size, shuffle=True,
                                        num_workers=params.num_workers,
                                        pin_memory=params.cuda)
            else:
                dl = DataLoader(SDFDataset(path), batch_size=params.batch_size, shuffle=False,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda)

            dataloaders[split] = dl

    return dataloaders
