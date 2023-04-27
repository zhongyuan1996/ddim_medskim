#!/bin/bash

model=("hita" "lsan" "lstm" "sand" "gruself" "timeline" "retain" " retainex" "TLSTM")
target_disease=("Amnesia")
seeds=(1234 2345 3456)
save_path="./saved_models/"

for SEED in ${seeds[@]}
        do
        for MODAL in ${model[@]}
        do
        for DISEASE in ${target_disease[@]}
        do
            CUDA_VISIBLE_DEVICES=1, python3 baseline.py --seed=$SEED --encoder=$MODAL --target_disease=$DISEASE --max_len=50 --n_epochs=30
      done
      done
      done
