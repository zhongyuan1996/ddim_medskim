import pickle
import pandas as pd
from models.dataset import padMatrix,padTime
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt

random.seed(1234)

# ehrs = pickle.load(open('data/mimic/1.4/test.seqs', 'rb'))
# labels = pickle.load(open('data/mimic/1.4/test.morts', 'rb'))
# dates = pickle.load(open('data/mimic/1.4/test.dates', 'rb'))
#
# timestep = []
# for patient in dates:
#     temp_time = [0]
#     for i in range(1, len(patient)):
#         temp_time.append((patient[i] - patient[0]).days)
#     timestep.append(list(reversed(temp_time)))
#
# indices = list(range(len(ehrs)))
#
# indices_train, indices_test = train_test_split(indices, test_size=0.25, random_state=1234)
# indices_test, indices_val = train_test_split(indices_test, test_size=0.6, random_state=1234)
#
# train_ehr, train_label, train_time_step = [], [], []
# test_ehr, test_label, test_time_step = [], [], []
# val_ehr, val_label, val_time_step = [], [], []
#
# for index in indices_train:
#     if len(ehrs[index]) <= 15:
#         train_ehr.append(ehrs[index])
#         train_label.append(labels[index])
#         train_time_step.append(timestep[index])
#     if len(ehrs[index]) != len(timestep[index]):
#         raise Exception("length of ehr and time step is not equal")
#
# for index in indices_test:
#     if len(ehrs[index]) <= 15:
#         test_ehr.append(ehrs[index])
#         test_label.append(labels[index])
#         test_time_step.append(timestep[index])
#
#     if len(ehrs[index]) != len(timestep[index]):
#         raise Exception("length of ehr and time step is not equal")
#
# for index in indices_val:
#     if len(ehrs[index]) <= 15:
#         val_ehr.append(ehrs[index])
#         val_label.append(labels[index])
#         val_time_step.append(timestep[index])
#
#     if len(ehrs[index]) != len(timestep[index]):
#         raise Exception("length of ehr and time step is not equal")
#
# pickle.dump((train_ehr, train_label, train_time_step), open('data/mimic/mimic_train.pickle', 'wb'))
# pickle.dump((test_ehr, test_label, test_time_step), open('data/mimic/mimic_test.pickle', 'wb'))
# pickle.dump((val_ehr, val_label, val_time_step), open('data/mimic/mimic_val.pickle', 'wb'))

ehr, label, time_step = pickle.load(open('data/mimic/mimic_train.pickle', 'rb'))
ehr_test, label_test, time_step_test = pickle.load(open('data/mimic/mimic_test.pickle', 'rb'))
ehr_val, label_val, time_step_val = pickle.load(open('data/mimic/mimic_val.pickle', 'rb'))

print(sum([1 for label in label if label == 1])+sum([1 for label in label_test if label == 1])+sum([1 for label in label_val if label == 1]))
print(sum([1 for label in label if label == 0])+sum([1 for label in label_test if label == 0])+sum([1 for label in label_val if label == 0]))
print(len(label)+len(label_test)+len(label_val))

a_ehr, a_label, a_timestep = pickle.load(open('data/mimic/mimic_train.pickle', 'rb'))
b_ehr, b_label, b_timestep = pickle.load(open('data/mimic/mimic_test.pickle', 'rb'))
c_ehr, c_label, c_timestep = pickle.load(open('data/mimic/mimic_val.pickle', 'rb'))

positive, negative = 0, 0
for patient in a_label:
    if patient == 1:
        positive += 1
    else:
        negative += 1
for patient in b_label:
    if patient == 1:
        positive += 1
    else:
        negative += 1
for patient in c_label:
    if patient == 1:
        positive += 1
    else:
        negative += 1
print(positive)
print(negative)


# visit_length = []
# for patient in a_ehr:
#     length = len(patient)
#     visit_length.append(length)
#
# for patient in b_ehr:
#     length = len(patient)
#     visit_length.append(length)
#
# for patient in c_ehr:
#     length = len(patient)
#     visit_length.append(length)
#
#
# print(max(visit_length))
# print(min(visit_length))
# print(sum(visit_length)/len(visit_length))
#
# from collections import Counter
# print(Counter(visit_length))
#
# plt.hist(visit_length, bins=range(0,43,1))
# plt.show()

# a_ehr, a_label, a_timestep = pickle.load(open('data/mimic/mimic_train.pickle', 'rb'))
# b_ehr, b_label, b_timestep = pickle.load(open('data/mimic/mimic_test.pickle', 'rb'))
# c_ehr, c_label, c_timestep = pickle.load(open('data/mimic/mimic_val.pickle', 'rb'))
#
# icd_codes = set()
# for patient in a_ehr:
#     for visit in patient:
#         for code in visit:
#             icd_codes.add(code)
#
# for patient in b_ehr:
#     for visit in patient:
#         for code in visit:
#             icd_codes.add(code)
#
# for patient in c_ehr:
#     for visit in patient:
#         for code in visit:
#             icd_codes.add(code)
#
# print(len(icd_codes))


# i = 0
# for a, b, c in zip(ehr, label, time_step):
#     print("ehr data: " + str(a))
#     print("label: " + str(b))
#     print("time_step: " + str(c))
#     i+=1
#     if i >2 :
#         break


# ehr, label, time_step = pickle.load(open('data/copd/copd_training_new.pickle', 'rb'))
# i = 0
# for a, b, c in zip(ehr, label, time_step):
#     print("ehr data: " + str(a))
#     print("label: " + str(b))
#     print("time_step: " + str(c))
#     i+=1
#     if i >2 :
#         break
#
# print("started data transfrom")
# code2id = pickle.load(open('data/copd/copd_code2idx_new.pickle', 'rb'))
# pad_id = len(code2id)
# ehr, _, _ = padMatrix(ehr, 20, 50, pad_id)
# time_step = padTime(time_step, 50, 100000)
#
# i = 0
# for a, b, c in zip(ehr, label, time_step):
#     print("ehr data: " + str(a))
#     print("label: " + str(b))
#     print("time_step: " + str(c))
#
#     print("ehr data visit 1 length: " + str(len(a[1])))
#     print("time_step visit 1 length: " + str(len(c))) # visit number from a few to 50, icd code length from a few to 20
#
#     # print("unsqueeze: "  + str(c.unsqueeze(2)))
#     i+=1
#     if i >=1 :
#         break


