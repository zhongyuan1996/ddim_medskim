#!/bin/bash

model=("LSTM_ehrGAN" "LSTM_GcGAN" "LSTM_actGAN" "LSTM_medGAN")
target_disease=("mimic" "Heart_failure" "COPD" "Kidney" "Amnesia")
seeds=(1234 2345 3456)
save_path="./saved_models/"

for SEED in ${seeds[@]}
        do
        for MODAL in ${model[@]}
        do
        for DISEASE in ${target_disease[@]}
        do
          path=$save_path$MODAL$SEED$DISEASE/
          if [ DISEASE="mimic" ]; then

              python3 GAN_runner.py --seed=$SEED --model=$MODAL --target_disease=$DISEASE --max_len=15 --n_epochs=30 --save_dir=$path
          else
              python3 GAN_runner.py --seed=$SEED --model=$MODAL --target_disease=$DISEASE --n_epochs=30 --save_dir=$path
          fi
      done
      done
      done
