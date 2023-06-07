
# %% Check the GPU available
import argparse

parser = argparse.ArgumentParser(description="Training one network see the help")
parser.add_argument("--model",            default = 'gcvit',   type=str, help = 'input for the test file')
parser.add_argument("--epochs",         default = 30,            type=int, help = 'number of epochs, defaulkt = 1')
parser.add_argument("--loss_f",         default = 'MAE',        type=str, help = 'loss function MSE or L2, default = MSE')
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
#%%
#net_name    = 'gcvit'
#nEpochs     = 10
#loss_function= 'MAE'
#Zernikes    = 209
#batch_size  = 60
#GPUdevices  = '0,1,2,3,4,5,6,7'
#fulltest    = 0
#load_train  = 40
#lr          = 0.000001 # lr         = 1e-4
#mod         = '0'

print(' ')
print('Start training ' + net_name + ' for values of Zn=' + str(Zernikes) + ' under loss function ' + loss_function)
print(' ')



#%%
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


# Pyramid propagator
from Functions.Propagators import *
from Functions.phaseGenerators import *
from Functions.oomao_functions import *
from argparse import Namespace
from math import sqrt, pi

nPxPup = 268

wfs = Namespace(modulation = 0, samp =2, D=3, rooftop =[0,0], alpha =pi/2, zModes =[2,210], nPxPup = [],
                batchSize =1, ReadoutNoise =0, PhotonNoise =0, nPhotonBackground =0, quantumEfficiency =1,
                fovInPixel    = [], pupil = [], pyrMask = [], jModes= [], 
                pupilLogical = [], modes = [], amplitude = [], ModPhasor= [])
wfs.nPxPup = nPxPup
wfs.fovInPixel    = wfs.nPxPup*2*wfs.samp
wfs.pupil = CreateTelescopePupil(nPxPup,"disc")
wfs.pyrMask = createPyrMask(wfs)
wfs.jModes = torch.arange(wfs.zModes[0], wfs.zModes[1]+1)
wfs.pupilLogical = wfs.pupil!=0
wfs.modes = CreateZernikePolynomials(wfs)
wfs.amplitude = 0.2 #small for low noise systems
wfs.ModPhasor = CreateModulationPhasor(wfs)

def reconstruction_batch(Y,zernRec):
    Y2 = Y.T
    out = torch.reshape(torch.matmul(zernRec[:,1:Y2.shape[0]+1], Y2),[268, 268, Y2.shape[1]])/550*2*torch.pi
    return out.permute(2,1,0)



def validation(val_path, epoch, result_path, model, net_name, loss_function, Zernikes,fulltest):
    test_list = os.listdir(val_path)
    loss_cnn = torch.zeros(len(test_list))
    zernRec = torch.from_numpy(scio.loadmat('../Dataset_pyramid/zernRec.mat')['zernRec']).cuda()
    if fulltest == 1:
        pyr2zern = scio.loadmat('../Dataset_pyramid/iMat_268_M0.mat')['pyr2zern']  #pyramidal prediction
        I_0 = scio.loadmat('../Dataset_pyramid/I0_raw_M0.mat')['I_0']
        nPhotons = np.sum(I_0)
        Io = I_0/nPhotons

    Yest_res = None
    Ygt_res = None
    Ypyr = None
    phaseM = None
    
    for i in range(len(test_list)):
        datamat = val_path + '/' + test_list[i]
        datamat = scio.loadmat(datamat)
        gt = datamat['Yz']       
        phase = datamat['Xs']
        x_ph = torch.from_numpy(phase).unsqueeze(0)#.permute(2,0,1)
        meas1 = Prop2VanillaPyrWFS_torch(x_ph,wfs)
        #nPhotons = torch.sum(meas1[0,0,:,:])
        #meas1 = meas1/nPhotons
        if fulltest == 1:
            xs = meas1-Io
            pyr = np.matmul(pyr2zern,np.expand_dims(xs.T.flatten(),axis = 1))

        
        phaseMap = torch.squeeze(meas1.cuda()).float()
        Ygt = torch.squeeze(torch.from_numpy(gt).cuda()).float()
        if fulltest == 1:
            Y_pyr = torch.squeeze(torch.from_numpy(pyr)).float()


        with torch.no_grad():

            rmse_1 = 0
            phaseMap = torch.unsqueeze(phaseMap,0)
            phaseMap = torch.unsqueeze(phaseMap,0)

            #for n_cl in range(epoch-load_train):
            Y0 = model(phaseMap).double()
            Ygt = Ygt[0:Zernikes]-Y0
            Ph_res  = reconstruction_batch(Ygt,zernRec)
            phaseMap = Prop2VanillaPyrWFS_torch(Ph_res.cpu(),wfs).cuda().float()
            
            nPhotons = torch.sum(phaseMap[0,0,:,:])
            phaseMap = phaseMap/nPhotons

            Yest = model(phaseMap)

            
            a = test_list[i]
            rmse_1 = torch.sqrt(torch.mean((Ygt[0:Zernikes]-torch.squeeze(Yest[0,0:Zernikes]))**2)) 
            loss_cnn[i] = rmse_1
            name = result_path + '/CNN_' + a[0:len(a) - 4] + '_{}_{:.4f}'.format(epoch, loss_cnn[i]) + '.mat'
            a = test_list[i]
            if phaseM is not None:
                Ygt_res = torch.concat([Ygt_res,Ygt.cpu().unsqueeze(0)],0)
                Yest_res =  torch.concat([Yest_res, Yest.cpu().unsqueeze(0)],0)
            else:
                Ygt_res = Ygt.cpu().unsqueeze(0)
                Yest_res = Yest.cpu().unsqueeze(0)

    #Y0 = model(phaseM).double()
    #Ygt_res = Ygt_res[:,0:Zernikes].cuda()-Y0
    #Ph_res  = reconstruction_batch(Ygt_res,zernRec)
    #phaseMap = Prop2VanillaPyrWFS_torch(Ph_res.cpu(),wfs).cuda().float()
    #nPhotons = torch.sum(phaseMap[0,0,:,:])
    #phaseMap = phaseMap/nPhotons
    #Yest = model(phaseMap)
    #rmse_1 = torch.sqrt(torch.mean((Ygt_res[:,0:Zernikes]-Yest[:,0:Zernikes])**2,1)) 


            
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
    
    zernRec = torch.from_numpy(scio.loadmat('../Dataset_pyramid/zernRec.mat')['zernRec']).cuda()
    for iteration, batch in tqdm(enumerate(train_data_loader)):
        Ygt = Variable(batch[0])
        Ygt = Ygt.cuda().double()
        Ygt = Ygt[:,0:Zernikes]
        phase = Variable(batch[1])
        phaseMap = Prop2VanillaPyrWFS_torch(phase,wfs)
        nPhotons = torch.sum(phaseMap[0,0,:,:])
        phaseMap = phaseMap/nPhotons

        phaseMap = phaseMap.cuda().float()

        optimizer_g.zero_grad()

        #for n_cl in range(epoch-load_train):
        Y0 = model(phaseMap).double()
        Ygt = Ygt-Y0
        Ph_res  = reconstruction_batch(Ygt,zernRec)
        phaseMap = Prop2VanillaPyrWFS_torch(Ph_res.cpu(),wfs).cuda().float()
        phaseMap = phaseMap/nPhotons



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
sub_fold = "Dataset_pyramid/Datasets_phasemap_D1.5_Nov_2022"


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
