from torch.utils.data import Dataset
import os
import torch
import scipy.io as scio
import numpy as np

import numpy as np

dy,dx = np.ogrid[-33:33+1,-33:33+1]
mask = dx**2+dy**2 <= 33**2
mask = mask.astype(float)
mask = np.concatenate((mask, mask),axis = 0)
mask = np.concatenate((mask, mask),axis = 1)



class Imgdataset(Dataset):

    def __init__(self, path):
        super(Imgdataset, self).__init__()
        self.data = []
        
        if os.path.exists(path):
            gt_path = path

            if os.path.exists(gt_path):
                gt = os.listdir(gt_path)
                self.data = [{'orig': gt_path + '/' + gt[i]} for i in range(len(gt))]
            else:
                raise FileNotFoundError('path doesnt exist!')
        else:
            raise FileNotFoundError('path doesnt exist!')

    def __getitem__(self, index,):
         data = self.data[index]["orig"]
         if data[-3]+data[-2]+data[-1] == 'mat':
            data = scio.loadmat(data)
            gt = data['Yz']       
            Xs = data['Xs']
            meas = Xs
         elif data[-3]+data[-2]+data[-1] == 'npy':
            data = np.load(data, allow_pickle=True)
            meas = data[0]*100000
            gt = data[1]
         
         meas = torch.squeeze(torch.from_numpy(meas))
         gt = torch.squeeze(torch.from_numpy(gt)) 

         return gt, meas

    def __len__(self):

        return len(self.data)
