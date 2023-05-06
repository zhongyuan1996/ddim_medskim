#!/bin/bash
model=("LSTM_ehrGAN" "LSTM_GcGAN" "LSTM_actGAN" "LSTM_medGAN" "Dipole_ehrGAN" "Dipole_GcGAN" "Dipole_actGAN" "Dipole_medGAN" "TLSTM_ehrGAN" "TLSTM_GcGAN" "TLSTM_actGAN" "TLSTM_medGAN" "SAND_ehrGAN" "SAND_GcGAN" "SAND_actGAN" "SAND_medGAN" )
#target_disease=("mimic" "Heart_failure" "COPD" "Kidney" "Amnesia")
target_disease=("mortality")
seeds=(1234)
save_path="./saved_models/"
for SEED in ${seeds[@]}
do
for MODAL in ${model[@]}
do
for DISEASE in ${target_disease[@]}
do
python3 ALL_GAN_runner.py --seed=$SEED --model=$MODAL --target_disease=$DISEASE --max_len=48 --max_num_codes=7727 --n_epochs=30 -bs=64
done
done
done
