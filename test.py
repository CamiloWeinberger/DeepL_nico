import argparse

parser = argparse.ArgumentParser(description="Test the neural network from the test dolfer")
parser.add_argument("--model",        default = 'convnext', type=str, help = 'select the neural network')
parser.add_argument("--loss_f",     default = 'MAE', type=str,  help = "loss function")
parser.add_argument("--datavariant",default = 209, type=int, help = "number of zernikes")
parser.add_argument("--epochs",     default = 40, type=int, help = "number of epochs that was trained")
parser.add_argument("--pyr",   default = 1,      type=int, help = 'include pyr prediction? (1/0 = yes/no)')
parser.add_argument("--device",    default = '0,1,2,3,4,5,6,7', type=str, help = 'number of GPUs')
parser.add_argument("--mod",    default = '0', type=str, help = 'modulation')
parser.add_argument("--Nfolder",    default = 'Data_D1.5_M0_splitted', type=str, help = 'Variation of NN [empty or lowr_]')
args = parser.parse_args()

net_name        = args.model
epoch           = args.epochs
loss_function   = args.loss_f
Zernikes        = args.datavariant
pyr_include     = args.pyr
GPUdevices      = args.device
mod             = args.mod
NNFolder        = args.Nfolder

import torch
import numpy as np
import os
import scipy.io as scio
import datetime
print(' ')
print('Start testing ' + net_name + ' for values of Zn=' + str(Zernikes) + ' under loss function ' + loss_function)
print(' ')

#%% Basic functions

dy,dx = np.ogrid[-33:33+1,-33:33+1]
mask = dx**2+dy**2 <= 33**2
mask = mask.astype(float)
mask = np.concatenate((mask, mask),axis = 0)
mask = np.concatenate((mask, mask),axis = 1)


pyr2zern = scio.loadmat('../Dataset_pyramid/iMat_268_M' + mod +'.mat')['pyr2zern']  #pyramidal prediction
I_0 = scio.loadmat('../Dataset_pyramid/I0_raw_M' + mod + '.mat')['I_0']


nPhotons = np.sum(I_0)
Io = I_0/nPhotons

use_cuda = torch.cuda.is_available()

def set_paths(main_fold,sub_fold):
    train_fold = main_fold + sub_fold + "/train"
    val_fold   = main_fold + sub_fold + "/val"
    test_fold  = main_fold + sub_fold + "/test" 
    model_path = main_fold + '/DL_pyr_results' + "/model/nocap/" + sub_fold + "/checkpoint"
    return train_fold, val_fold, test_fold, model_path

def set_CNN(model_out_path):
    if n_gpu > 1:
        CNNModel = torch.nn.DataParallel(CNNModel)
    CNNModel = torch.load(model_out_path)
    #CNNModel = model.module if hasattr(model, "module") else model

## Main
def check_fold_output(test_result_path):
    if not os.path.exists(test_result_path):
        os.makedirs(test_result_path)

def RMSE(x,y):
    out     = torch.sqrt(torch.mean((x-y)**2,1))
    return out

def test(model, test_path, test_result_path, mod, model_path):
    test_list   = os.listdir(test_path)
    Yest_res    = None
    Ygt_res     = None
    rmse_res    = None
    std_res     = None
    r0_res      = None
    Yest_pyr    = None
    rmsepyr_res = None
    stdpyr_res  = None
    
    for i in range(len(test_list)):
        r0  = torch.from_numpy(np.array(float(test_list[i][str.find(test_list[i],'r0_')+3:str.find(test_list[i],'_part')]))).unsqueeze(0)
        if r0_res is not None:
            r0_res = torch.concat([r0_res,r0],0)
        else:
            r0_res = r0

    r0_res, index = torch.sort(r0_res)

    for ind in range(len(test_list)):
        i   = index[ind]
        Ypyr_var = None
        print('Testing: ' + test_list[i])
        datamat = scio.loadmat(test_path + '/' + test_list[i])
        #datamat_phase = scio.loadmat(phase_path + '/Phase_' + test_list[i][4:])
        Xs  = datamat['X_s']
        Ygt = datamat['Y_z']
        X_ph = datamat['X_phase']

        X_s = torch.from_numpy(Xs).cuda().float()
        X_s = torch.unsqueeze(torch.permute(X_s,(2,0,1)),1)
        Ygt = torch.from_numpy(Ygt).cuda().float()
        X_ph = torch.from_numpy(X_ph).float()

        with torch.no_grad():
            Yest = model(X_s)#*550/2/np.pi
            
            rmse_1  = torch.mean(RMSE(Ygt[:,0:Zernikes],Yest)).cpu().unsqueeze(0)
            std_1   = torch.std(RMSE(Ygt[:,0:Zernikes],Yest)).cpu().unsqueeze(0)
            if Yest_res is not None:
                Yest_res    = torch.concat([Yest_res,Yest.cpu()],0)
                Ygt_res     = torch.concat([Ygt_res,Ygt.cpu()],0)
                rmse_res    = torch.concat([rmse_res,rmse_1],0)
                std_res     = torch.concat([std_res,std_1],0)
                X_phase     = torch.concat([X_phase,X_ph],2)
            else:
                Yest_res    = Yest.cpu()
                Ygt_res     = Ygt.cpu()
                rmse_res    = rmse_1
                std_res     = std_1
                X_phase     = X_ph
            
            if pyr_include == 1:
                for idp in range(X_s.shape[0]):
                    meas1   = Xs[:,:,idp]#*nPhotons#(Xs-Io)*mask#*nPhotons
                    xs      = meas1-Io
                    pyr     = np.matmul(pyr2zern,np.expand_dims(xs.T.flatten(),axis = 1))*550/2/np.pi
                    Y_pyr   = torch.squeeze(torch.from_numpy(pyr)).float().unsqueeze(0)
                    if Ypyr_var is not None:
                        Ypyr_var    = torch.concat([Ypyr_var,Y_pyr],0)
                    else:
                        Ypyr_var    = Y_pyr.cpu()
                rpyr_1      = torch.mean(RMSE(Ygt[:,0:Zernikes].cpu(),Ypyr_var[:,0:Zernikes])).cpu().unsqueeze(0)
                spyr_1      = torch.std(RMSE(Ygt[:,0:Zernikes].cpu(),Ypyr_var[:,0:Zernikes])).cpu().unsqueeze(0)
                if Yest_pyr is not None:
                    Yest_pyr        = torch.concat([Yest_pyr,Ypyr_var],0)
                    rmsepyr_res     = torch.concat([rmsepyr_res,rpyr_1],0)
                    stdpyr_res      = torch.concat([stdpyr_res,spyr_1],0)
                else:
                    rmsepyr_res     = rpyr_1
                    stdpyr_res      = spyr_1
                    Yest_pyr        = Ypyr_var
    
    if model_path.find('lowr')+1:
        fname = test_result_path + '/' + net_name + "_loss_" + loss_function + '_test_Zn' + str(Zernikes) + '_M' + mod + '_Results_{}'.format(epoch) + '_lowr.mat' 
    else:
        fname = test_result_path + '/' + net_name + "_loss_" + loss_function + '_test_Zn' + str(Zernikes) + '_M' + mod + '_Results_{}'.format(epoch) + '.mat'   

    scio.savemat(fname, {'Yest': Yest_res.numpy(),'Ygt': Ygt_res.numpy(),'NN_rmse':rmse_res.numpy(),
                         'NN_std':std_res.numpy(),'r0_vals':r0_res.numpy(),'Ypyr': Yest_pyr.numpy(),
                         'Pyr_rmse': rmsepyr_res.numpy(),'Pyr_std': stdpyr_res.numpy(), 'X_phase': X_phase.numpy()})
    print('saved')


date = datetime.date.today()  
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=GPUdevices
n_gpu = torch.cuda.device_count()
print(torch.cuda.is_available())
print('The number of GPU is {}'.format(n_gpu))


#%% Test code (Modify from here)

main_fold           = "../"
model_path          = '../DL_pyr_results/model/nocap/Dataset_pyramid/' + NNFolder + '/checkpoint'
#model_path          = '../DL_pyr_results/model/nocap/Dataset_pyramid/Data_D1.5_M0_lowr_splitted/checkpoint'
test_path           = '../Dataset_pyramid/Datasets_phase2pyr_D1.5_Nov_2022_M' + mod + '_test/'
#phase_path           = "../Dataset_Phasemap/Datasets_phasemap_D1.5_Nov_2022_test/"
test_result_path    = '../DL_pyr_results/test/' + NNFolder

check_fold_output(test_result_path)

model_out_path      =  model_path + '/' + net_name + "_loss_" + loss_function + '_Zn' + str(Zernikes) + "_epoch_{}.pth".format(epoch)
CNNModel            = torch.load(model_out_path)

test(CNNModel, test_path, test_result_path, mod,model_path)

