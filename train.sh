# TN3K and TG3K, multi-task training
python train.py -fold 0 -model_name strnet -dataset TATN -gpu 0 -lr 1e-3
python train.py -fold 1 -model_name strnet -dataset TATN -gpu 0 -lr 1e-3
python train.py -fold 2 -model_name strnet -dataset TATN -gpu 0 -lr 1e-3
python train.py -fold 3 -model_name strnet -dataset TATN -gpu 0 -lr 1e-3
python train.py -fold 4 -model_name strnet -dataset TATN -gpu 0 -lr 1e-3

# TN3K and TNUS, multi-task training
python train.py -fold 0 -model_name strnet -dataset TATU -gpu 0 -lr 1e-3
python train.py -fold 1 -model_name strnet -dataset TATU -gpu 0 -lr 1e-3
python train.py -fold 2 -model_name strnet -dataset TATU -gpu 0 -lr 1e-3
python train.py -fold 3 -model_name strnet -dataset TATU -gpu 0 -lr 1e-3
python train.py -fold 4 -model_name strnet -dataset TATU -gpu 0 -lr 1e-3

