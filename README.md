# Pyramidal_DL

## Introduction
inbcludes DL models `wfnet` and `gc_vit` for wavefront estimation.

*      main.py: main file for training and testing the model.


## DL training from remote server using JupyterHub
More details can be found in [JupyterHub](./main_jupyter.ipynb)

Define the main path for runing codes. To run the Jupyther server:
* Open the JupyterHub on browser `158.251.52.231:8080` , server: `user` pass: `pass`
* To activate the `pyramidal env`, in terminal type: `python -m ipykernel install --user --name=pyramidal` (once)
* In terminal type: `jupyter server`

``` branch
    To access the server, open this file in a browser:
        file:///home/bizon/.local/share/jupyter/runtime/jpserver-4122063-open.html
    Or copy and paste one of these URLs:
        http://158.251.52.231:8888/?token=316407df12ba12ce84ac2a9323615f5c0d1276de628fee9e
     or http://127.0.0.1:8888/?token=316407df12ba12ce84ac2a9323615f5c0d1276de628fee9e
```


* copy and path the URL in the Kernel of Jupyter Notebook

![image](https://user-images.githubusercontent.com/111007682/205405377-625c7e95-6868-4886-83de-43b91f714adf.png)



### Training
Runing codes in background
``` branch
nohup bash -c "python train.py --model convnext --loss_f MSE --load_train 0 --epochs 30 --lr 1e-4; 
python train.py --model convnext --loss_f MSE --load_train 30 --epochs 10 --lr 1e-5; 
python train.py --model convnext --loss_f MAE --load_train 0 --epochs 30 --lr 1e-4; 
python train.py --model convnext --loss_f MAE --load_train 30 --epochs 10 --lr 1e-5;
python train.py --model wfnet --loss_f MSE --load_train 0 --epochs 30 --lr 1e-4; 
python train.py --model wfnet --loss_f MSE --load_train 30 --epochs 10 --lr 1e-5; 
python train.py --model wfnet --loss_f MAE --load_train 0 --epochs 30 --lr 1e-4; 
python train.py --model wfnet --loss_f MAE --load_train 30 --epochs 10 --lr 1e-5; 
python train.py --model xception --loss_f MSE --load_train 0 --epochs 30 --lr 1e-4; 
python train.py --model xception --loss_f MSE --load_train 30 --epochs 10 --lr 1e-5; 
python train.py --model xception --loss_f MAE --load_train 0 --epochs 30 --lr 1e-4; 
python train.py --model xception --loss_f MAE --load_train 30 --epochs 10 --lr 1e-5; 
python train.py --model gcvit --loss_f MSE --load_train 0 --epochs 30 --lr 1e-4; 
python train.py --model gcvit --loss_f MSE --load_train 30 --epochs 10 --lr 1e-5; 
python train.py --model gcvit --loss_f MAE --load_train 0 --epochs 30 --lr 1e-4; 
python train.py --model gcvit --loss_f MAE --load_train 30 --epochs 10 --lr 1e-5; 
python train.py --model convnext --loss_f MSE --load_train 0 --epochs 30 --lr 1e-4; 
python train.py --model convnext --loss_f MSE --load_train 30 --epochs 10 --lr 1e-5; 
python train.py --model convnext --loss_f MAE --load_train 0 --epochs 30 --lr 1e-4; 
python train.py --model convnext --loss_f MAE --load_train 30 --epochs 10 --lr 1e-5"
```

Code test simulated data ```variation --mod 9```

```branch
nohup bash -c "python test.py --model convnext --loss_f MSE --mod 0;
python test.py --model convnext --loss_f MAE --mod 0;
python test.py --model wfnet --loss_f MSE --mod 0;
python test.py --model wfnet --loss_f MAE --mod 0;
python test.py --model gcvit --loss_f MSE --mod 0;
python test.py --model gcvit --loss_f MAE --mod 0;
python test.py --model xception --loss_f MSE --mod 0;
python test.py --model xception --loss_f MAE --mod 0;"
```


Code for test experiment

```branch
nohup bash -c "python test_exp.py --model convnext --loss_f MSE --mod 0;
python test_exp.py --model convnext --loss_f MAE --mod 0;
python test_exp.py --model wfnet --loss_f MSE --mod 0;
python test_exp.py --model wfnet --loss_f MAE --mod 0;
python test_exp.py --model gcvit --loss_f MSE --mod 0;
python test_exp.py --model gcvit --loss_f MAE --mod 0;
python test_exp.py --model xception --loss_f MSE --mod 0;
python test_exp.py --model xception --loss_f MAE --mod 0;"
```

For no - nohup
```branch
nohup command >/dev/null 2>&1
```


All the codes are runing with the bizon working directory

The [result code](./test_jupyter.ipynb) predicts the ground truth values and the reconstruction.

In the results we show the estimation fronm all the systems tested
![Estimation](./figures/estimations.png)
The phasemap reconstruction
![Reconstruction](./figures/reconstruction.png)
And the comparison error from each estimation over the $r_0$ value.
![Error](./figures/error.png)

.By Optolab
