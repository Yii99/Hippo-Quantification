"""
Contains class that runs inferencing
"""
import torch
import numpy as np

from networks.RecursiveUNet import UNet

from utils.utils import med_reshape

class UNetInferenceAgent:
    """
    Stores model and parameters and some methods to handle inferencing
    """
    def __init__(self, parameter_file_path='', model=None, device="cpu", patch_size=64):

        self.model = model
        self.patch_size = patch_size
        self.device = device

        if model is None:
            self.model = UNet(num_classes=3)

        if parameter_file_path:
            self.model.load_state_dict(torch.load(parameter_file_path, map_location=self.device))

        self.model.to(device)

    def single_volume_inference_unpadded(self, volume):
        """
        Runs inference on a single volume of arbitrary patch size,
        padding it to the conformant size first

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        
        x, y, z = volume.shape
        new_shape= (x, self.patch_size, self.patch_size)
        new_volume = med_reshape(volume, new_shape)
        slices = np.zeros(new_volume.shape)
        for slice_idx in range(new_volume.shape[0]):
            slc = new_volume[slice_idx,:,:]
            slice = torch.from_numpy(slc.astype(np.single)/np.max(slc)).unsqueeze(0).unsqueeze(0).to(self.device)
            prediction = self.model(slice)
            prediction = np.squeeze(prediction).cpu().detach()
            slices[slice_idx,:,:] = torch.argmax(prediction, dim=0)
        return slices
        


    def single_volume_inference(self, volume):
        """
        Runs inference on a single volume of conformant patch size

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        self.model.eval()

        # Assuming volume is a numpy array of shape [X,Y,Z] and we need to slice X axis
        # slices = []

        # Create mask for each slice across the X (0th) dimension. After
        # that, put all slices into a 3D Numpy array. 

        slices = np.zeros(volume.shape)
        for slice_index in range(volume.shape[0]):
            slc = volume[slice_index,:,:]
            slc = slc.astype(np.single)/np.max(slc)
            slice = torch.from_numpy(slc).unsqueeze(0).unsqueeze(0).to(self.device)
            prediction = self.model(slice)
            prediction = np.squeeze(prediction.cpu().detach())
            slices[slice_index,:,:] = torch.argmax(prediction, dim=0)


        return slices
