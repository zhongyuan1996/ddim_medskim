#!/bin/bash

model=("Dipole_ehrGAN" "Dipole_GcGAN" "Dipole_actGAN" "Dipole_medGAN" "TLSTM_ehrGAN" "TLSTM_GcGAN" "TLSTM_actGAN" "TLSTM_medGAN" "SAND_ehrGAN" "SAND_GcGAN" "SAND_actGAN" "SAND_medGAN" )
#target_disease=("mimic" "Heart_failure" "COPD" "Kidney" "Amnesia")
target_disease=("mimic" "ARF" "Shock" "mortality" "Kidney" "Amnesia")
seeds=(1234 2345 3456)
save_path="./saved_models/"

for SEED in ${seeds[@]}
        do
        for MODAL in ${model[@]}
        do
        for DISEASE in ${target_disease[@]}
        do
          if [ DISEASE=="mimic" ]; then
              CUDA_VISIBLE_DEVICES=2, python3 ALL_GAN_runner.py --seed=$SEED --model=$MODAL --target_disease=$DISEASE --max_len=15 --n_epochs=30 -bs=128
          if [ DISEASE=="ARF" ]; then
              CUDA_VISIBLE_DEVICES=2, python3 ALL_GAN_runner.py --seed=$SEED --model=$MODAL --target_disease=$DISEASE --max_len=12 --max_num_codes=5132 --n_epochs=30 -bs=128
          if [ DISEASE=="Shock" ]; then
              CUDA_VISIBLE_DEVICES=2, python3 ALL_GAN_runner.py --seed=$SEED --model=$MODAL --target_disease=$DISEASE --max_len=12 --max_num_codes=5795 --n_epochs=30 -bs=128
          if [ DISEASE=="mortality" ]; then
              CUDA_VISIBLE_DEVICES=2, python3 ALL_GAN_runner.py --seed=$SEED --model=$MODAL --target_disease=$DISEASE --max_len=48 --max_num_codes=7727 --n_epochs=30 -bs=128
          else
              CUDA_VISIBLE_DEVICES=2, python3 ALL_GAN_runner.py --seed=$SEED --model=$MODAL --target_disease=$DISEASE --max_len=50 --n_epochs=30 -bs=128
          fi
      done
      done
      done
