"""
This file contains code that will kick off training and testing processes
"""
import os
import json
import numpy as np

from experiments.UNetExperiment import UNetExperiment
from data_prep.HippocampusDatasetLoader import LoadHippocampusData

class Config:
    """
    Holds configuration parameters
    """
    def __init__(self):
        self.name = "Basic_unet"
        self.root_dir = r"../../section1/out"
        self.n_epochs = 2
        self.learning_rate = 0.0002
        self.batch_size = 8
        self.patch_size = 64
        self.test_results_dir = "../out/Result"

if __name__ == "__main__":
    # Get configuration
    c = Config()

    # Load data
    print("Loading data...")

    data = LoadHippocampusData(c.root_dir, y_shape = c.patch_size, z_shape = c.patch_size)


    # Create test-train-val split

    keys = range(len(data))

    # Here, random permutation of keys array would be useful in case if we do something like 
    # a k-fold training and combining the results. 

    split = dict()

    # Create three keys in the dictionary: "train", "val" and "test". In each key, store
    # the array with indices of training volumes to be used for training, validation 
    # and testing respectively.

    datasize = len(data)
    train_size = int(datasize*0.7)
    val_size = int(datasize*0.2)
    test_size = datasize-train_size-val_size
    indexes = np.random.choice(datasize, datasize, False)

    split['train'] = indexes[:train_size]
    split['val'] = indexes[train_size:train_size+val_size]
    split['test'] = indexes[train_size+val_size:]

    # Set up and run experiment

    exp = UNetExperiment(c, split, data)

    # run training
    exp.run()

    # prep and run testing

    # Test method is not complete. Go to the method and complete it
    results_json = exp.run_test()

    results_json["config"] = vars(c)

    with open(os.path.join(exp.out_dir, "results.json"), 'w') as out_file:
        json.dump(results_json, out_file, indent=2, separators=(',', ': '))

