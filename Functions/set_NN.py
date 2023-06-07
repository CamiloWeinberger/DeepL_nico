import torch
import os
import datetime
## Select the number of GPUs to use in the server

n_gpu = torch.cuda.device_count()
print(torch.cuda.is_available())
print('The number of GPU is {}'.format(n_gpu))

##%%

def set_CNN(loss_function,net_name, Zernikes, load_train):

    if net_name == 'gcvit':
        from Models.GCVit import gc_vit_tiny
        CNNModel = gc_vit_tiny(num_classes=Zernikes)

    elif net_name == 'wfnet':
        from Models.wfnet import WFNet
        CNNModel = WFNet(num_classes=Zernikes)

    elif net_name == 'xception':
        from Models.xception_nosize import Xception
        CNNModel = Xception(num_classes=Zernikes)
    elif net_name == 'vit':

        from Models.Vit import ViT
        CNNModel = ViT(
            image_size = 256,
            patch_size = 32,
            num_classes = Zernikes,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )

    elif net_name == 'unet':
        from Models.unet_model import UNet
        CNNModel = UNet(num_classes=Zernikes, n_channels=1)

    elif net_name == 'yolo':
        from Models.yolov6.models.yolo import Model
        CNNModel = Model(num_classes=Zernikes, n_channels=1)
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
        CNNModel = torch.load(model_path + "/CNN_epoch_{}.pth".format(load_train))
        CNNModel = model.module if hasattr(model, "module") else model

    return CNNModel, loss
