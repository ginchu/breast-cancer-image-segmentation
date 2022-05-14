import numpy as np
import torch
import glob
import matplotlib.image as mpimg
class Dataset(torch.utils.data.Dataset):
  def __init__(self, train = True):
      self.x = sorted(glob.glob('./TNBC_NucleiSegmentation/Slide_*/*'))
      self.x.remove('./TNBC_NucleiSegmentation/Slide_09/09_2.png')
      self.x.remove('./TNBC_NucleiSegmentation/Slide_09/09_3.png')
      self.x.remove('./TNBC_NucleiSegmentation/Slide_09/09_4.png')
      
      self.y = sorted(glob.glob('./TNBC_NucleiSegmentation/GT_*/*'))
      self.y.remove('./TNBC_NucleiSegmentation/GT_09/09_2.png')
      self.y.remove('./TNBC_NucleiSegmentation/GT_09/09_3.png')
      self.y.remove('./TNBC_NucleiSegmentation/GT_09/09_4.png')


  def __len__(self):
      return len(self.x)
   
  def __getitem__(self, idx):
      x = np.transpose(mpimg.imread(self.x[idx])[:,:,:3],[2,1,0])
      y = np.transpose(mpimg.imread(self.y[idx]).reshape(512,512,1),[2,1,0])
      return torch.tensor(x).float(), torch.tensor(y).float()
