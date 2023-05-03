#!/bin/bash

model=("LSTM_GcGAN" "LSTM_medGAN" "LSTM_ehrGAN" "LSTM_actGAN" )
#target_disease=("mimic" "Heart_failure" "COPD" "Kidney" "Amnesia")
target_disease=("mimic")
seeds=(4567 5678 6789 7890 8901 9012 0123)
save_path="./saved_models/"

for SEED in ${seeds[@]}
        do
        for MODAL in ${model[@]}
        do
        for DISEASE in ${target_disease[@]}
        do
          if [ DISEASE=="mimic" ]; then
              CUDA_VISIBLE_DEVICES=0, python3 ALL_GAN_runner.py --seed=$SEED --model=$MODAL --target_disease=$DISEASE --max_len=15 --n_epochs=30 -bs=128
          else
              CUDA_VISIBLE_DEVICES=0, python3 ALL_GAN_runner.py --seed=$SEED --model=$MODAL --target_disease=$DISEASE --max_len=50 --n_epochs=30 -bs=128
          fi
      done
      done
      done
