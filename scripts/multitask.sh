python multitask_classifier.py --option pretrain --lr 1e-3 --hidden_dropout_prob 0.1 --batch_size 12 --use_gpu
python multitask_classifier.py --option finetune --hidden_dropout_prob 0.1 --batch_size 12 --use_gpu
