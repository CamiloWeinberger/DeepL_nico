from Functions.loadData import Imgdataset
from torch.utils.data import DataLoader
from Models.GCVit import gc_vit_xxtiny
import torch.optim as optim
import torch
import scipy.io as scio
import time
import datetime
import os
from torch.autograd import Variable
from tqdm import tqdm



## Train
def test(test_path, epoch, result_path, model):
    test_list = os.listdir(test_path)
    loss_cnn = torch.zeros(len(test_list))
    Yest_res = None
    Ygt_res = None
    for i in range(len(test_list)):
        datamat = scio.loadmat(test_path + '/' + test_list[i])     
        Ygt = datamat['Zgt']
        Ygt = torch.from_numpy(Ygt).cuda().float()
        phaseMap = datamat['x']
        phaseMap = torch.from_numpy(phaseMap).cuda().float()
        phaseMap = torch.unsqueeze(phaseMap,0)

        with torch.no_grad():

            rmse_1 = 0
            phaseMap = torch.unsqueeze(phaseMap,0)
            Yest = model(phaseMap)
            rmse_1 = torch.sqrt(torch.mean((Ygt-Yest)**2)) 
            loss_cnn[i] = rmse_1

            a = test_list[i]
            name = result_path + '/CNN_' + a[0:len(a) - 4] + '_{}_{:.4f}'.format(epoch, loss_cnn[i]) + '.mat'
            if Yest_res is not None:
                Yest_res = torch.concat([Yest_res,Yest.cpu()],0)
                Ygt_res = torch.concat([Ygt_res,Ygt.cpu()],0)
            else:
                Yest_res = Yest.cpu()
                Ygt_res = Ygt.cpu()
            
    fname = result_path + '/CNN_Results_{}'.format(epoch) + '.mat'        
    prtname = "CNN Validation: RMSE -- {:.4f}".format(torch.mean(loss_cnn))        
    scio.savemat(fname, {'Yest': Yest_res.numpy(),'Ygt': Ygt_res.numpy(),'loss':loss_cnn.numpy()})
    print(prtname)

def validation(val_path, epoch, result_path, model):
    test_list = os.listdir(val_path)
    loss_cnn = torch.zeros(len(test_list))
    Yest_res = None
    Ygt_res = None
    for i in range(len(test_list)):
        datamat = scio.loadmat(val_path + '/' + test_list[i])     
        Ygt = datamat['Zgt']
        Ygt = torch.from_numpy(Ygt).cuda().float()
        phaseMap = datamat['x']
        phaseMap = torch.from_numpy(phaseMap).cuda().float()
        phaseMap = torch.unsqueeze(phaseMap,0)

        with torch.no_grad():

            rmse_1 = 0
            phaseMap = torch.unsqueeze(phaseMap,0)
            Yest = model(phaseMap)
            rmse_1 = torch.sqrt(torch.mean((Ygt-Yest)**2)) 
            loss_cnn[i] = rmse_1

            a = test_list[i]
            name = result_path + '/CNN_' + a[0:len(a) - 4] + '_{}_{:.4f}'.format(epoch, loss_cnn[i]) + '.mat'
            if Yest_res is not None:
                Yest_res = torch.concat([Yest_res,Yest.cpu()],0)
                Ygt_res = torch.concat([Ygt_res,Ygt.cpu()],0)
            else:
                Yest_res = Yest.cpu()
                Ygt_res = Ygt.cpu()
            
    fname = result_path + '/CNN_Results_{}'.format(epoch) + '.mat'        
    prtname = "CNN Validation: RMSE -- {:.4f}".format(torch.mean(loss_cnn))        
    scio.savemat(fname, {'Yest': Yest_res.numpy(),'Ygt': Ygt_res.numpy(),'loss':loss_cnn.numpy()})
    print(prtname)


def train(epoch, result_path, model, lr):
    epoch_loss = 0
    begin = time.time()

    optimizer_g = optim.AdamW([{'params': model.parameters()}], lr=lr)

    for iteration, batch in tqdm(enumerate(train_data_loader)):
        Ygt = Variable(batch[0])
        Ygt = Ygt.cuda().float()
        phaseMap = Variable(batch[1])
        phaseMap = phaseMap.cuda().float()
        phaseMap = torch.unsqueeze(phaseMap,1)

        optimizer_g.zero_grad()
        Yest = model(phaseMap)
        Loss1 = loss(Yest,Ygt)

        Loss1.backward()
        optimizer_g.step()

        epoch_loss += Loss1.data

    model = model.module if hasattr(model, "module") else model
    validation(val_fold, epoch, result_path, model.eval())
    end = time.time()
    print("===> Epoch {} Complete: Avg. Loss: {:.7f}".format(epoch, epoch_loss / len(train_data_loader)),
          "  time: {:.2f}".format(end - begin))
    
def checkpoint(epoch, model_path):
    model_out_path =  model_path + '/' + "PyrNet_epoch_{}.pth".format(epoch)
    torch.save(CNNModel, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))   