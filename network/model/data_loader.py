import os
import pickle
import numpy as np

from torch.utils.data import Dataset, DataLoader

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
            print("Funny number of data files in this directory.")
        
        f = filenames[0]

        # Load data from 100K shapes.
        # examples is a list of lists, one per training example.
        # Each list contains a 
        #   (500,2) ndarray of points           Xtrain
        #   (500,)  ndarray of function values  ytrain
        #   (100,)  ndarray of SDF zero points  r
        self.examples = pickle.load(open( f, "rb" ))

    def __len__(self):
        # return size of dataset
        return len(self.examples)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        Xtrain, ytrain, r = self.examples[idx] # See description above.

        
        # We want:
        # INPUT: (500,3) concatenation of Xtrain, ytrain for this shape
        # OUTPUT: (100,) ndarray of SDF zero points

        data = np.column_stack([Xtrain, ytrain])
        return data, r


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
            path = os.path.join(data_dir, "{}_sdf".format(split))

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
