import os
import torch
import scipy.io as scio
import numpy as np
import time
from tqdm import tqdm
from IPython import display
from torch.autograd import Variable
import torch.optim as optim


dy,dx = np.ogrid[-33:33+1,-33:33+1]
mask = dx**2+dy**2 <= 33**2
mask = mask.astype(float)
mask = np.concatenate((mask, mask),axis = 0)
mask = np.concatenate((mask, mask),axis = 1)


def validation(val_path, epoch, result_path, model, net_name, loss_function, Zernikes,fulltest):
    test_list = os.listdir(val_path)
    loss_cnn = torch.zeros(len(test_list))
    if fulltest == 1:
        pyr2zern = scio.loadmat('../Dataset_pyramid/iMat_268.mat')['pyr2zern']  #pyramidal prediction
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
                pyr = np.matmul(pyr2zern,np.expand_dims(xs.flatten(),axis = 1))
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
            #transform = T.Resize(224)
            #phaseMap = transform(phaseMap)
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




def train(train_data_loader, epoch, result_path, model, lr, net_name, loss_function, Zernikes,fulltest):
    epoch_loss = 0
    begin = time.time()

    optimizer_g = optim.AdamW([{'params': model.parameters()}], lr=lr)

    for iteration, batch in tqdm(enumerate(train_data_loader)):
        Ygt = Variable(batch[0])
        Ygt = Ygt.cuda().float()
        Ygt = Ygt[:,0:Zernikes]
        phaseMap = Variable(batch[1])
        phaseMap = phaseMap.cuda().float()
        phaseMap = torch.unsqueeze(phaseMap,1)

        optimizer_g.zero_grad()
        #transform = T.Resize(224)
        #phaseMap = transform(phaseMap)
        Yest = model(phaseMap)
        Loss1 = loss(Yest,Ygt)

        Loss1.backward()
        optimizer_g.step()

        epoch_loss += Loss1.data

    model = model.module if hasattr(model, "module") else model
    validation(val_fold, epoch, result_path, model.eval(), net_name, loss_function, Zernikes, fulltest)
    end = time.time()
    print("===> Epoch {} Complete: Avg. Loss: {:.7f}".format(epoch, epoch_loss / len(train_data_loader)),
          "  time: {:.2f}".format(end - begin))

def train_gradual(train_data_loader, epoch, result_path, model, lr, net_name, loss_function, Zernikes):
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
        #transform = T.Resize(224)
        #phaseMap = transform(phaseMap)
        Yest = model(phaseMap)
        Loss1 = loss(Yest[:,0:Zernikes],Ygt[:,0:Zernikes])

        Loss1.backward()
        optimizer_g.step()

        epoch_loss += Loss1.data

    model = model.module if hasattr(model, "module") else model
    validation(val_fold, epoch, result_path, model.eval(), net_name, loss_function, Zernikes, nEpochs)
    end = time.time()
    print("===> Epoch {} Complete: Avg. Loss: {:.7f}".format(epoch, epoch_loss / len(train_data_loader)),
          "  time: {:.2f}".format(end - begin))
    
    
def checkpoint(epoch, model_path, net_name, loss_function, Zernikes):
    model_out_path =  model_path + '/' + net_name + "_loss_" + loss_function + '_Zn' + str(Zernikes) +"_epoch_{}.pth".format(epoch)
    torch.save(CNNModel, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
    
def train_NN(load_train, train_data_loader, nEpochs, result_path, CNNModel, lr, net_name, loss_function, Zernikes,fulltest):
    for epoch in range(load_train + 1, load_train + nEpochs + 1):
        train(train_data_loader, epoch, result_path, CNNModel, lr, net_name, loss_function, Zernikes,fulltest)
        if (epoch % 5 == 0) and (epoch < 100):
            lr = lr * 0.95
            print(lr)
        if (epoch % 1 == 0 or epoch > 50):
            CNNModel = CNNModel.module if hasattr(CNNModel, "module") else CNNModel
            checkpoint(epoch, model_path, net_name, loss_function, Zernikes)
        if n_gpu > 1:
            CNNModel = torch.nn.DataParallel(CNNModel)  

