# STR-Net: A Semi-Supervised Thyroid-Prior Guided Refinement Network with Multi-Mix Augmentation and Edge Regularization for Ultrasound Thyroid Nodule Segmentation

This repository is the official implementation of STR-Net. 

# Requirements:
- Python 3.8.17
- torch 2.2.0
- torchvision 0.17.0
- opencv-python 4.8.0.76
- numpy 1.22.4
- scipy 1.10.1


# Datasets
TN3K, TG3K, and DDTI can be prepared following the [TRFE-Plus](https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation).
TUNS can be downloaded through [link](https://github.com/taodeng/TNPPD-Net/)

Download and place the datasets in ```./data/```


# Training
```
python train_strnet.py -fold 0 -model_name strnet -dataset TATN -gpu 0 -lr 1e-3
```
or
```
sh train.sh
```

# Test the model
```
python test.py -model_name strnet -load_path "./models/TATN/strnet_fold0/strnet_best.pth" -fold 0 -gpu 0 -test_dataset TATN_TN3K
```
or
```
sh test.sh
```
**You can also download the pretrained weights through the Google Drive [link](https://drive.google.com/file/d/121B4R_0evFKLbnc6FpmBk1qQLsmxBSP7/view?usp=sharing) and evaluate**


# Post-processing with Edge Distance Regularization (EDR)
```
python eval_ls.py -model_name strnet -fold 0 -gpu 0 -test_dataset TATN_TN3K
```
or
```
sh test_ls.sh
```
# Evaluate the results of EDR
```
python eval_only.py -model_name strnet -fold 0 -gpu 0 -test_dataset TATN_TN3K
```
or
```
sh test_eval_only.sh
```


## Reference
* [TRFE-Plus](https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation)
* [TNUS](https://github.com/taodeng/TNPPD-Net/)





