#!/bin/bash

model=("hita" "lsan" "lstm" "sand" "gruself" "retain" " retainex" "TLSTM")
target_disease=("EEG")
seeds=(1234 2345 3456)
save_path="./saved_models/"

for SEED in ${seeds[@]}
        do
        for MODAL in ${model[@]}
        do
        for DISEASE in ${target_disease[@]}
        do
            CUDA_VISIBLE_DEVICES=3, python3 og_baseline.py --seed=$SEED --encoder=$MODAL --target_disease=$DISEASE --max_len=200 --n_epochs=30 --max_num_codes=16
      done
      done
      done
