import pickle
import pandas as pd
from models.dataset import padMatrix,padTime


ehr, label, time_step = pickle.load(open('data/hf/hf/hf_training_new.pickle', 'rb'))
i = 0
for a, b, c in zip(ehr, label, time_step):
    print("ehr data: " + str(a))
    print("label: " + str(b))
    print("time_step: " + str(c))
    i+=1
    if i >2 :
        break

print("started data transfrom")
code2id = pickle.load(open('data/hf/hf/hf_code2idx_new.pickle', 'rb'))
pad_id = len(code2id)
ehr, _, _ = padMatrix(ehr, 20, 50, pad_id)
time_step = padTime(time_step, 50, 100000)

i = 0
for a, b, c in zip(ehr, label, time_step):
    print("ehr data: " + str(a))
    print("label: " + str(b))
    print("time_step: " + str(c))

    print("ehr data visit 1 length: " + str(len(a[1])))
    print("time_step visit 1 length: " + str(len(c))) # visit number from a few to 50, icd code length from a few to 20

    # print("unsqueeze: "  + str(c.unsqueeze(2)))
    i+=1
    if i >=1 :
        break


