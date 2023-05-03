import pickle

import numpy as np
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

# ehr, label, time_step = pickle.load(open('data/mimic/mimic_train.pickle', 'rb'))
# ehr_test, label_test, time_step_test = pickle.load(open('data/mimic/mimic_test.pickle', 'rb'))
# ehr_val, label_val, time_step_val = pickle.load(open('data/mimic/mimic_val.pickle', 'rb'))
#
# print(sum([1 for label in label if label == 1])+sum([1 for label in label_test if label == 1])+sum([1 for label in label_val if label == 1]))
# print(sum([1 for label in label if label == 0])+sum([1 for label in label_test if label == 0])+sum([1 for label in label_val if label == 0]))
# print(len(label)+len(label_test)+len(label_val))
#
# a_ehr, a_label, a_timestep = pickle.load(open('data/mimic/mimic_train.pickle', 'rb'))
# b_ehr, b_label, b_timestep = pickle.load(open('data/mimic/mimic_test.pickle', 'rb'))
# c_ehr, c_label, c_timestep = pickle.load(open('data/mimic/mimic_val.pickle', 'rb'))
#
# positive, negative = 0, 0
# for patient in a_label:
#     if patient == 1:
#         positive += 1
#     else:
#         negative += 1
# for patient in b_label:
#     if patient == 1:
#         positive += 1
#     else:
#         negative += 1
# for patient in c_label:
#     if patient == 1:
#         positive += 1
#     else:
#         negative += 1
# print(positive)
# print(negative)


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

# ehr, label = pickle.load(open('data/EEG/EEG_train.pickle', 'rb'))
# # ehr_test, label_test = pickle.load(open('EEG_test.pickle', 'rb'))
# # ehr_val, label_val = pickle.load(open('EEG_val.pickle', 'rb'))
# i = 0
# for a, b in zip(ehr, label):
#     print("ehr data: " + str(a))
#     print(len(a))
#     print(str(a[0]))
#     print(len(a[0]))
#     print("label: " + str(b))
#     i+=1
#     if i >2 :
#         break

# a_ehr, a_label, a_timestep = pickle.load(open('data/mimic/mimic_train.pickle', 'rb'))
# b_ehr, b_label, b_timestep = pickle.load(open('data/mimic/mimic_test.pickle', 'rb'))
# c_ehr, c_label, c_timestep = pickle.load(open('data/mimic/mimic_val.pickle', 'rb'))
# ehr, label, time_step = pickle.load(open('data/copd/copd_training_new.pickle', 'rb'))


# ehr, label = pickle.load(open('data/ARF/ARF_training_new.npz', 'rb'))
# i = 0
# for a, b, c in zip(ehr, label):
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
# # time_step = padTime(time_step, 50, 100000)
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


###################################   ARF   ############################################

# training = np.load('data/ARF/ARF_training_og.npz')
# training_x, training_y = training['x'], training['y']
# temp = np.array(list(reversed(range(0, len(training_x[0])))))
# training_timeseq = np.repeat(temp[np.newaxis,: ], len(training_x), axis=0)
#
# testing = np.load('data/ARF/ARF_testing_og.npz')
# testing_x, testing_y = testing['x'], testing['y']
# temp = np.array(list(reversed(range(0, len(testing_x[0])))))
# testing_timeseq = np.repeat(temp[np.newaxis,: ], len(testing_x), axis=0)
#
# val = np.load('data/ARF/ARF_validation_og.npz')
# val_x, val_y = val['x'], val['y']
# temp = np.array(list(reversed(range(0, len(val_x[0])))))
# val_timeseq = np.repeat(temp[np.newaxis,: ], len(val_x), axis=0)
#
# training_x, training_y, training_timeseq, testing_x, testing_y, testing_timeseq, val_x, val_y, val_timeseq = \
# training_x.tolist(), training_y.tolist(), training_timeseq.tolist(), testing_x.tolist(), testing_y.tolist(), testing_timeseq.tolist(), val_x.tolist(), val_y.tolist(), val_timeseq.tolist()
#
# pickle.dump((training_x, training_y, training_timeseq), open('data/ARF/ARF_training_new.pickle', 'wb'))
# pickle.dump((testing_x, testing_y, testing_timeseq), open('data/ARF/ARF_testing_new.pickle', 'wb'))
# pickle.dump((val_x, val_y, val_timeseq), open('data/ARF/ARF_validation_new.pickle', 'wb'))

# training = np.load('data/Shock/Shock_training_og.npz')
# training_x, training_y = training['x'], training['y']
# temp = np.array(list(reversed(range(0, len(training_x[0])))))
# training_timeseq = np.repeat(temp[np.newaxis,: ], len(training_x), axis=0)
#
# testing = np.load('data/Shock/Shock_testing_og.npz')
# testing_x, testing_y = testing['x'], testing['y']
# temp = np.array(list(reversed(range(0, len(testing_x[0])))))
# testing_timeseq = np.repeat(temp[np.newaxis,: ], len(testing_x), axis=0)
#
# val = np.load('data/Shock/Shock_validation_og.npz')
# val_x, val_y = val['x'], val['y']
# temp = np.array(list(reversed(range(0, len(val_x[0])))))
# val_timeseq = np.repeat(temp[np.newaxis,: ], len(val_x), axis=0)
#
# training_x, training_y, training_timeseq, testing_x, testing_y, testing_timeseq, val_x, val_y, val_timeseq = \
# training_x.tolist(), training_y.tolist(), training_timeseq.tolist(), testing_x.tolist(), testing_y.tolist(), testing_timeseq.tolist(), val_x.tolist(), val_y.tolist(), val_timeseq.tolist()
#
# pickle.dump((training_x, training_y, training_timeseq), open('data/Shock/Shock_training_new.pickle', 'wb'))
# pickle.dump((testing_x, testing_y, testing_timeseq), open('data/Shock/Shock_testing_new.pickle', 'wb'))
# pickle.dump((val_x, val_y, val_timeseq), open('data/Shock/Shock_validation_new.pickle', 'wb'))
#
training = np.load('data/mortality/mortality_training_og.npz')
training_x, training_y = training['x'], training['y']
temp = np.array(list(reversed(range(0, len(training_x[0])))))
training_timeseq = np.repeat(temp[np.newaxis,: ], len(training_x), axis=0)
training_x, training_y, training_timeseq = training_x.tolist(), training_y.tolist(), training_timeseq.tolist()
pickle.dump((training_x, training_y, training_timeseq), open('data/mortality/mortality_training_new.pickle', 'wb'))

# testing = np.load('data/mortality/mortality_testing_og.npz')
# testing_x, testing_y = testing['x'], testing['y']
# temp = np.array(list(reversed(range(0, len(testing_x[0])))))
# testing_timeseq = np.repeat(temp[np.newaxis,: ], len(testing_x), axis=0)
# testing_x, testing_y, testing_timeseq = testing_x.tolist(), testing_y.tolist(), testing_timeseq.tolist()
# pickle.dump((testing_x, testing_y, testing_timeseq), open('data/mortality/mortality_testing_new.pickle', 'wb'))



# val = np.load('data/mortality/mortality_validation_og.npz')
# val_x, val_y = val['x'], val['y']
# temp = np.array(list(reversed(range(0, len(val_x[0])))))
# val_timeseq = np.repeat(temp[np.newaxis,: ], len(val_x), axis=0)
# val_x, val_y, val_timeseq = val_x.tolist(), val_y.tolist(), val_timeseq.tolist()
# pickle.dump((val_x, val_y, val_timeseq), open('data/mortality/mortality_validation_new.pickle', 'wb'))
