#!/bin/bash

#model=("hita" "lsan" "lstm" "sand" "gruself" "timeline" "retain" " retainex" "TLSTM")
model=("Adacare" "medskim")
target_disease=("Amnesia")
seeds=(1234 2345 3456)
save_path="./saved_models/"

for SEED in ${seeds[@]}
        do
        for MODAL in ${model[@]}
        do
        for DISEASE in ${target_disease[@]}
        do
          if [ MODAL=="Adacare" ]; then
            CUDA_VISIBLE_DEVICES=1, python3 adacare.py --seed=$SEED --target_disease=$DISEASE --max_len=50 --n_epochs=30 -bs=64
          else
            CUDA_VISIBLE_DEVICES=1, python3 medskim.py --seed=$SEED --target_disease=$DISEASE --max_len=50 --n_epochs=30 -bs=64


            fi
      done
      done
      done
