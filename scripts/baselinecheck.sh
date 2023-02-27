##! /bin/bash

# Note: Don't forget to edit the hyper-parameters for part d.

# Pretrain
python classifier.py --option pretrain --lr 1e-3 --hidden_dropout_prob 0.1
python classifier.py --option finetune --hidden_dropout_prob 0.1