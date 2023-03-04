#!/usr/bin/env bash
python ./launch.py -data_path ./data/ -data_name assist1213 -environment env -CDM IRT1 -T 10 -ST [1,5,10] -agent Train -FA NCAT -latent_factor 50 \
-learning_rate 0.001 -training_epoch 1 -seed 145 -gpu_no 0 -inner_epoch 30 -rnn_layer 2 -gamma 0.8 -batch 128 -restore_model False
