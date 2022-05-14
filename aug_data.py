import numpy as np
import torch
import glob
import matplotlib.image as mpimg
class AugDataset(torch.utils.data.Dataset):
  def __init__(self, train = True):
      self.x = sorted(glob.glob('./Augmented_Data/Slide_*/*'))
      
      self.y = sorted(glob.glob('./Augmented_Data/GT_*/*'))


  def __len__(self):
      return len(self.x)
   
  def __getitem__(self, idx):
      x = np.transpose(mpimg.imread(self.x[idx])[:,:,:3],[2,1,0])
      y = np.transpose(mpimg.imread(self.y[idx]).reshape(512,512,1),[2,1,0])
      return torch.tensor(x).float(), torch.tensor(y).float()
