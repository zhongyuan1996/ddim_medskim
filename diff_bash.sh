#!/bin/bash

model=("medDiff" )
#target_disease=("mimic" "Heart_failure" "COPD" "Kidney" "Amnesia")
target_disease=("ARF" "Shock" "mortality")
seeds=(1234 2345 3456)
save_path="./saved_models/"

for SEED in ${seeds[@]}
        do
        for MODAL in ${model[@]}
        do
        for DISEASE in ${target_disease[@]}
        do
          if [ DISEASE=="ARF" ]; then
              CUDA_VISIBLE_DEVICES=2, python3 ALL_GAN_runner.py --seed=$SEED --model=$MODAL --target_disease=$DISEASE --max_len=12 --max_num_codes=5132 --n_epochs=30 -bs=64
          elif [ DISEASE=="Shock" ]; then
              CUDA_VISIBLE_DEVICES=2, python3 ALL_GAN_runner.py --seed=$SEED --model=$MODAL --target_disease=$DISEASE --max_len=12 --max_num_codes=5795 --n_epochs=30 -bs=64
          elif [ DISEASE=="mortality" ]; then
              CUDA_VISIBLE_DEVICES=2, python3 ALL_GAN_runner.py --seed=$SEED --model=$MODAL --target_disease=$DISEASE --max_len=48 --max_num_codes=7727 --n_epochs=30 -bs=64
          fi
      done
      done
      done
