from Functions.loadData_numpy import Imgdataset
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
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
#import model

from Models.GCVit import gc_vit_xxtiny

date = datetime.date.today()  
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
n_gpu = torch.cuda.device_count()
print(torch.cuda.is_available())
print('The number of GPU is {}'.format(n_gpu))



inResolution = 224
numChannels = 1

def set_paths(main_fold,sub_fold):
    train_fold = main_fold + sub_fold + "/train"
    val_fold   = main_fold + sub_fold + "/val"
    test_fold  = main_fold + sub_fold + "/test" 
    model_path = "./model/nocap/" + sub_fold + "/checkpoint"
    return train_fold, val_fold, test_fold, model_path

def set_CNN(net_name, Zernikes, load_train):
    if net_name == 'gc_vit_xxtiny':
        CNNModel = gc_vit_xxtiny(num_classes=Zernikes)
    elif net_name == 'wnfet':
        CNNModel = wfnet(num_classes=Zernikes)
    CNNModel = CNNModel.cuda()

    loss = torch.nn.MSELoss()
    loss.cuda()


    if n_gpu > 1:
        CNNModel = torch.nn.DataParallel(CNNModel)
    if load_train != 0:
        CNNModel = torch.load(model_path + "/CNN_epoch_{}.pth".format(load_train))
        CNNModel = model.module if hasattr(model, "module") else model

    return CNNModel, loss


## Train
def validation(val_path, epoch, result_path, model):
    test_list = os.listdir(val_path)
    loss_cnn = torch.zeros(len(test_list))
    Yest_res = None
    Ygt_res = None
    for i in range(len(test_list)):
        datamat = val_path + '/' + test_list[i]
        if datamat[-3]+datamat[-2]+datamat[-1] == 'mat':
            datamat = scio.loadmat(test_list[i])
            gt = datamat['Y_z']    
            meas = datamat['X_s']*100000
        elif datamat[-3]+datamat[-2]+datamat[-1] == 'npy':
            datamat = np.load(datamat, allow_pickle=True)
            meas = datamat[0]*100000
            gt = datamat[1]
        
        phaseMap = torch.squeeze(torch.from_numpy(meas).cuda()).float()
        Ygt = torch.squeeze(torch.from_numpy(gt).cuda()).float()
       

        with torch.no_grad():

            rmse_1 = 0
            phaseMap = torch.unsqueeze(phaseMap,0)
            transform = T.Resize(224)
            phaseMap = transform(phaseMap)
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

    vlp[epoch-1] = torch.mean(loss_cnn)
    display.clear_output(wait=True)
    plt.plot(vlp[0:epoch])
    display.display(plt.gcf())


def train(epoch, result_path, model, lr, train_data_loader,loss, val_fold, model_path):
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
        transform = T.Resize(224)
        phaseMap = transform(phaseMap)
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
    
    
def checkpoint(epoch, CNNModel, model_path):
    model_out_path =  model_path + '/' + "PyrNet_epoch_{}.pth".format(epoch)
    torch.save(CNNModel, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))






## Main
def check_fold_output(result_path,test_result_path,model_path):
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(test_result_path):
        os.makedirs(test_result_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
            
def train_NN(load_train, nEpochs, result_path, CNNModel, lr, train_data_loader,loss, val_fold, model_path):
    for epoch in range(load_train + 1, load_train + nEpochs + 1):
        if epoch == 0:
            vlp = np.zeros(nEpochs)
        train(epoch, result_path, CNNModel, lr, train_data_loader,loss, val_fold, model_path)
        if (epoch % 5 == 0) and (epoch < 100):
            lr = lr * 0.95
            print(lr)
        if (epoch % 1 == 0 or epoch > 50):
            CNNModel = CNNModel.module if hasattr(CNNModel, "module") else CNNModel
            checkpoint(epoch, CNNModel, model_path)
        if n_gpu > 1:
            CNNModel = torch.nn.DataParallel(CNNModel)    