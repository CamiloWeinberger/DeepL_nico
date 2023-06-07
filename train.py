
# %% Check the GPU available
import argparse

parser = argparse.ArgumentParser(description="Training one network see the help")
parser.add_argument("--model",            default = 'gcvit',   type=str, help = 'input for the test file')
parser.add_argument("--epochs",         default = 30,            type=int, help = 'number of epochs, defaulkt = 1')
parser.add_argument("--loss_f",         default = 'MSE',        type=str, help = 'loss function MSE or L2, default = MSE')
parser.add_argument("--datavariant",    default = 209,           type=int, help = 'number of outputs, defalut = 54')
parser.add_argument("--device",    default = '0,1,2,3,4,5,6,7', type=str, help = 'number of GPUs')
parser.add_argument("--batch",          default = 60,           type=int, help = 'batch size')
parser.add_argument("--fulltest",        default = 0,           type=int, help = "ask for full prediction and save")
parser.add_argument("--load_train",     default = 0,           type=int, help = "load the training data")
parser.add_argument("--lr",             default = 0.0001,       type=float, help = "learning rate")
parser.add_argument("--mod",    default = '0', type=str, help = 'modulation')
args = parser.parse_args()

net_name    = args.model
nEpochs     = args.epochs
loss_function= args.loss_f
Zernikes    = args.datavariant
batch_size  = args.batch
GPUdevices  = args.device
fulltest    = args.fulltest
load_train  = args.load_train
lr          = args.lr # lr         = 1e-4
mod         = args.mod
print(' ')
print('Start training ' + net_name + ' for values of Zn=' + str(Zernikes) + ' under loss function ' + loss_function)
print(' ')

import torch




use_cuda = torch.cuda.is_available()
if use_cuda:
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__CUDA Device Name:',torch.cuda.get_device_name(0))
    print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)

##%%


from Functions.loadData_numpy import Imgdataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import scipy.io as scio
import time
import datetime
import os
from torch.autograd import Variable
from tqdm import tqdm
#import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
#import model


vlp = np.zeros(1000)
dy,dx = np.ogrid[-33:33+1,-33:33+1]
mask = dx**2+dy**2 <= 33**2
mask = mask.astype(float)
mask = np.concatenate((mask, mask),axis = 0)
mask = np.concatenate((mask, mask),axis = 1)

def set_paths(main_fold,sub_fold):
    train_fold = main_fold + sub_fold + "/train"
    val_fold   = main_fold + sub_fold + "/val"
    test_fold  = main_fold + sub_fold + "/test" 
    model_path = main_fold + '/DL_pyr_results' + "/model/nocap/" + sub_fold + "/checkpoint"
    return train_fold, val_fold, test_fold, model_path

def set_CNN(loss_function,net_name, Zernikes, load_train):
    if net_name == 'gcvit':
        # https://github.com/NVlabs/GCVit
        from Models.GCVit import gc_vit_xxtiny
        CNNModel = gc_vit_xxtiny(num_classes=Zernikes)
    elif net_name == 'wfnet':
        # Ours!
        from Models.wfnet import WFNet
        CNNModel = WFNet(num_classes=Zernikes)
    elif net_name == 'xception':
        # https://github.com/tstandley/Xception-PyTorch
        from Models.xception_nosize import Xception
        CNNModel = Xception(num_classes=Zernikes)
    elif net_name == 'vit':
        # https://github.com/lucidrains/vit-pytorch
        from Models.Vit import ViT
        CNNModel = ViT(
            image_size = 256, patch_size = 32, num_classes = Zernikes, dim = 1024,
            depth = 6, heads = 16, mlp_dim = 2048, dropout = 0.1, emb_dropout = 0.1)
    elif net_name == 'unet':
        # https://github.com/milesial/Pytorch-UNet
        from Models.unet_model import UNet
        CNNModel = UNet(num_classes=Zernikes, n_channels=1)
    elif net_name == 'bit':
        # https://github.com/google-research/big_transfer
        import Models.bit_pytorch.models as models
        CNNModel = models.KNOWN_MODELS['BiT-M-R50x1'](head_size = Zernikes)
    elif net_name == 'maxvit':
        # https://github.com/google-research/maxvit
        from maxvit.models.maxvit import MaxViT
        CNNModel = MaxViT(num_classes = Zernikes)
    elif net_name == 'convnext':
        # https://github.com/facebookresearch/ConvNeXt
        from Models.convnext import ConvNeXt
        CNNModel = model = ConvNeXt(num_classes = Zernikes, depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
    elif net_name == 'convnext2':
        # https://github.com/facebookresearch/ConvNeXt-V2
        from Models.convnextv2 import ConvNeXtV2
        CNNModel = ConvNeXtV2(num_classes = Zernikes)
        
    CNNModel = CNNModel.cuda()

    # define loss_function
    if loss_function == 'MSE':
        loss = torch.nn.MSELoss()
    elif loss_function == 'L1' or loss_function == 'MAE':
        loss = torch.nn.L1Loss()
    loss.cuda()

    
    if n_gpu > 1:
        CNNModel = torch.nn.DataParallel(CNNModel)
    if load_train != 0:
        CNNModel = torch.load(model_path + "/" + net_name + "_loss_" + loss_function + "_Zn" + str(Zernikes) + "_epoch_{}.pth".format(load_train))
        #CNNModel = model.module if hasattr(model, "module") else model

    return CNNModel, loss


def check_fold_output(result_path,test_result_path,model_path):
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(test_result_path):
        os.makedirs(test_result_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

## Train test and validation functions
def replace(x,y):
    if x == None:
        x=y[0:10,0:10]*0
    return x

def validation(val_path, epoch, result_path, model, net_name, loss_function, Zernikes,fulltest):
    test_list = os.listdir(val_path)
    loss_cnn = torch.zeros(len(test_list))
    if fulltest == 1:
        pyr2zern = scio.loadmat('../Dataset_pyramid/iMat_268_M0.mat')['pyr2zern']  #pyramidal prediction
        I_0 = scio.loadmat('../Dataset_pyramid/I0_raw_M0.mat')['I_0']
        nPhotons = np.sum(I_0)
        Io = I_0/nPhotons

    Yest_res = None
    Ygt_res = None
    Ypyr = None
    
    for i in range(len(test_list)):
        datamat = val_path + '/' + test_list[i]
        if datamat[-3]+datamat[-2]+datamat[-1] == 'mat':
            datamat = scio.loadmat(datamat)
            gt = datamat['Yz']       
            meas1 = datamat['Xs']
            if fulltest == 1:
                xs = meas1-Io
                pyr = np.matmul(pyr2zern,np.expand_dims(xs.T.flatten(),axis = 1))
        elif datamat[-3]+datamat[-2]+datamat[-1] == 'npy':
            datamat = np.load(datamat, allow_pickle=True)
            meas1 = datamat[0]
            gt = datamat[1]
        
        phaseMap = torch.squeeze(torch.from_numpy(meas1).cuda()).float()
        Ygt = torch.squeeze(torch.from_numpy(gt).cuda()).float()
        if fulltest == 1:
            Y_pyr = torch.squeeze(torch.from_numpy(pyr)).float()


        with torch.no_grad():

            rmse_1 = 0
            phaseMap = torch.unsqueeze(phaseMap,0)
            phaseMap = torch.unsqueeze(phaseMap,0)
            Yest = model(phaseMap)
            rmse_1 = torch.sqrt(torch.mean((Ygt[0:Zernikes]-torch.squeeze(Yest[0,0:Zernikes]))**2)) 
            loss_cnn[i] = rmse_1

            a = test_list[i]
            name = result_path + '/CNN_' + a[0:len(a) - 4] + '_{}_{:.4f}'.format(epoch, loss_cnn[i]) + '.mat'
            if Yest_res is not None:
                Yest_res = torch.concat([Yest_res,Yest.cpu()],0)
                Ygt_res = torch.concat([Ygt_res,Ygt.cpu().unsqueeze(0)],0)
                if fulltest == 1:
                    #X_s  = torch.concat([X_s,phaseMap.cpu().squeeze().unsqueeze(-1)],2)
                    Ypyr = torch.concat([Ypyr,Y_pyr.squeeze().unsqueeze(0)],0)
            else:
                Yest_res = Yest.cpu()
                Ygt_res = Ygt.cpu().unsqueeze(0)
                if fulltest == 1:
                    #X_s = phaseMap.cpu().squeeze().unsqueeze(-1)
                    Ypyr = Y_pyr.squeeze().unsqueeze(0)

            
    fname = result_path + '/' + net_name + "_loss_" + loss_function + '_Zn' + str(Zernikes) + '_Results_{}'.format(epoch) + '.mat'        
    prtname = net_name + " Validation: RMSE -- {:.4f}".format(torch.mean(loss_cnn))
    if Ypyr == None:
        scio.savemat(fname, {'Yest': Yest_res.numpy(),'Ygt': Ygt_res.numpy(),'loss':loss_cnn.numpy(),'Ypyr': Ygt_res.numpy(),'test_list': test_list,'val_path': val_path})
    else:
        scio.savemat(fname, {'Yest': Yest_res.numpy(),'Ygt': Ygt_res.numpy(),'loss':loss_cnn.numpy(),'Ypyr': Ypyr.numpy(),'test_list': test_list,'val_path': val_path})
    #print(prtname)


def train(epoch, result_path, model, lr, net_name, loss_function, Zernikes,fulltest, ):
    epoch_loss = 0
    begin = time.time()

    optimizer_g = optim.AdamW([{'params': model.parameters()}], lr=lr)

    for iteration, batch in tqdm(enumerate(train_data_loader)):
        Ygt = Variable(batch[0])
        Ygt = Ygt.cuda().double()
        Ygt = Ygt[:,0:Zernikes]
        phaseMap = Variable(batch[1])
        phaseMap = phaseMap.cuda().float()
        phaseMap = torch.unsqueeze(phaseMap,1)

        optimizer_g.zero_grad()

        Yest = model(phaseMap).double()
        Loss1 = loss(Yest,Ygt)

        Loss1.backward()
        optimizer_g.step()

        epoch_loss += Loss1.data

    model = model.module if hasattr(model, "module") else model
    validation(val_fold, epoch, result_path, model.eval(), net_name, loss_function, Zernikes, fulltest)
    end = time.time()
    print("===> Epoch {} Complete: Avg. Loss: {:.7f}".format(epoch, epoch_loss / len(train_data_loader)),
          "  time: {:.2f}".format(end - begin))


    model = model.module if hasattr(model, "module") else model
    validation(val_fold, epoch, result_path, model.eval(), net_name, loss_function, Zernikes, nEpochs)
    end = time.time()
    print("===> Epoch {} Complete: Avg. Loss: {:.7f}".format(epoch, epoch_loss / len(train_data_loader)),
          "  time: {:.2f}".format(end - begin))
    
    
def checkpoint(epoch, model_path, net_name, loss_function, Zernikes):
    model_out_path =  model_path + '/' + net_name + "_loss_" + loss_function + '_Zn' + str(Zernikes) +"_epoch_{}.pth".format(epoch)
    torch.save(CNNModel, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
    
def train_NN(load_train, nEpochs, result_path, CNNModel, lr, net_name, loss_function, Zernikes,fulltest):
    for epoch in range(load_train + 1, load_train + nEpochs + 1):
        train(epoch, result_path, CNNModel, lr, net_name, loss_function, Zernikes,fulltest)
        if (epoch % 5 == 0) and (epoch < 100):
            lr = lr * 0.75
            print(lr)
        if (epoch % 1 == 0 or epoch > 50):
            CNNModel = CNNModel.module if hasattr(CNNModel, "module") else CNNModel
            checkpoint(epoch, model_path, net_name, loss_function, Zernikes)
        if n_gpu > 1:
            CNNModel = torch.nn.DataParallel(CNNModel)  


def pot_z(nnn):
    z = nnn*(nnn+1)/2
    return int(z)


def load_test(main_fold, name):
    data = scio.loadmat(main_fold + '/DL_pyr_results/results/' + name)
    Ygt = data['Ygt']
    Yest = data['Yest']
    return Yest, Ygt

#%%


## Select the number of GPUs to use in the server

date = datetime.date.today()  
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=GPUdevices
n_gpu = torch.cuda.device_count()
print(torch.cuda.is_available())
print('The number of GPU is {}'.format(n_gpu))

##%%

## define parameters for the network
#Zernikes   = 54
#load_train = 0
#lr         = 1e-4
#batch_size = 60
#nEpochs    = 1

## Network parameters
main_fold = "../"
sub_fold = "Dataset_pyramid/Data_D1.5_M" + mod + "_splitted"


result_path = "../DL_pyr_results/results"
test_result_path = "../DL_pyr_results/results"
train_fold, val_fold, test_fold, model_path = set_paths(main_fold, sub_fold)
check_fold_output(result_path,test_result_path,model_path)
dataset = Imgdataset(train_fold)
train_data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

#%% Train function

CNNModel,loss   = set_CNN(loss_function,net_name, Zernikes, load_train)
#Train network
train_NN(load_train, nEpochs, result_path, CNNModel, lr, net_name, loss_function, Zernikes,fulltest)

# GIT_SSH_COMMAND='ssh -i ../.ssh/id_rsa -o IdentitiesOnly=yes' git pull
# GIT_SSH_COMMAND='ssh -i ../.ssh/id_ed25519 -o IdentitiesOnly=yes' git clone git@github.com:CamiloWeinberger/Pyramidal_DL_lightning.git