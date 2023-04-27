#!/bin/bash

#model=("LSTM_ehrGAN" "LSTM_GcGAN" "LSTM_actGAN" "LSTM_medGAN")
model=( "LSTM_GcGAN" "LSTM_medGAN")
target_disease=("mimic" "Heart_failure" "COPD" "Kidney" "Amnesia")
seeds=(1234 2345 3456)
save_path="./saved_models/"

for SEED in ${seeds[@]}
        do
        for MODAL in ${model[@]}
        do
        for DISEASE in ${target_disease[@]}
        do
          if [ DISEASE="mimic" ]; then
              CUDA_VISIBLE_DEVICES=3, python3 GAN_runner.py --seed=$SEED --encoder=$MODAL --target_disease=$DISEASE --max_len=15 --n_epochs=30
          else
              CUDA_VISIBLE_DEVICES=3, python3 GAN_runner.py --seed=$SEED --encoder=$MODAL --target_disease=$DISEASE --max_len=50 --n_epochs=30
          fi
      done
      done
      done
