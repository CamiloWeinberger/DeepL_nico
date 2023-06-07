# Pyramidal_DL

If you want to propagate and train using a closed-Loop. Look at trian_CL.py


## Introduction
inbcludes DL models `convnext` and `gc_vit` for wavefront estimation.

*      main.py: main file for training and testing the model.


## DL training from remote server using JupyterHub
 How to train more than 1, without nohup file "retrained

``` branch
nohup bash -c "python train.py --model convnext --loss_f MSE --load_train 0 --epochs 30 --lr 1e-4; 
python train.py --model convnext --loss_f MSE --load_train 30 --epochs 10 --lr 1e-5;"  >/dev/null 2>&1
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

## Test 30-05

Models:
* Data_D1.5_M0_old
	* ep 40 (bad for low r_0)
	* gcvit, convnext, xception

* Data_D1.5_M0_lowr
	* ep 40
	* gcvit, convnext, xception 
* Data_D1.5_M0
	* ep 50 (retrained lowr)
	* gcvit, convnext
* Data_phasemap_D1.5_Nov_2022
	* ep 50 (traied in closed-loop)
	* gcvit, convnext



```branch
nohup bash -c "
python test.py --model gcvit --epochs 50;
python test.py --model convnext --epochs 50;
python test.py --model gcvit --epochs 50 --Nfold Data_D1.5_M0_lowr_splitted;
python test.py --model convnext --epochs 50 --Nfold Data_D1.5_M0_lowr_splitted;
python test_close_loop_oneIter.py --model gcvit --epochs 50;
python test_close_loop_oneIter.py --model convnext --epochs 50;
python test_close_loop_oneIter.py --model gcvit --epochs 50 --Nfold Data_D1.5_M0_lowr_splitted;
python test_close_loop_oneIter.py --model convnext --epochs 50 --Nfold Data_D1.5_M0_lowr_splitted;
python test_close_loop.py --model gcvit --epochs 50;
python test_close_loop.py --model convnext --epochs 50;
python test_close_loop.py --model gcvit --epochs 50 --Nfold Data_D1.5_M0_lowr_splitted;
python test_close_loop.py --model convnext --epochs 50 --Nfold Data_D1.5_M0_lowr_splitted;
" >/dev/null 2>&1
```


.By Optolab
