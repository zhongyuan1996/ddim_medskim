import pickle
import ast
import pandas as pd

def compute_time_gaps(timestamps):
    all_patient = []
    for patient in timestamps:
        temp = [0]
        for i in range(1, len(patient)):
            temp.append(abs(patient[i] - patient[i-1]))
        all_patient.append(temp)
    return all_patient

def compute_code_time_gaps(ehr, timegaps):
    all_patients_code_timegaps = []

    for p_idx, patient in enumerate(ehr):
        patient_code_timegaps = []
        patient_previous_codes = {}  # This will store the last visit index where the code was observed

        for idx, visit in enumerate(patient):
            visit_timegaps = []

            for code in visit:
                if code not in patient_previous_codes:  # First appearance of the code
                    visit_timegaps.append(0)
                else:
                    # Calculate time gap since the last appearance
                    last_idx = patient_previous_codes[code]
                    timegap_since_last = sum(timegaps[p_idx][last_idx+1:idx+1])
                    visit_timegaps.append(timegap_since_last)

                patient_previous_codes[code] = idx  # Update the last appearance of the code

            patient_code_timegaps.append(visit_timegaps)

        all_patients_code_timegaps.append(patient_code_timegaps)

    return all_patients_code_timegaps




def data_loading_and_modify_delta_t(target_disease):
    if target_disease == 'Heartfailure':
        code2id = pickle.load(open('data/hf/hf_code2idx_new.pickle', 'rb'))
        pad_id = len(code2id)
        data_path = './data/hf/hf'
    elif target_disease == 'COPD':
        code2id = pickle.load(open('./data/copd/copd_code2idx_new.pickle', 'rb'))
        pad_id = len(code2id)
        data_path = './data/copd/copd'
    elif target_disease == 'Kidney':
        code2id = pickle.load(open('./data/kidney/kidney_code2idx_new.pickle', 'rb'))
        pad_id = len(code2id)
        data_path = './data/kidney/kidney'
    elif target_disease == 'Dementia':
        code2id = pickle.load(open('./data/dementia/dementia_code2idx_new.pickle', 'rb'))
        pad_id = len(code2id)
        data_path = './data/dementia/dementia'
    elif target_disease == 'Amnesia':
        code2id = pickle.load(open('./data/amnesia/amnesia_code2idx_new.pickle', 'rb'))
        pad_id = len(code2id)
        data_path = './data/amnesia/amnesia'
    elif target_disease == 'mimic':
        pad_id = 4894
        data_path = './data/mimic/mimic'
    else:
        raise ValueError('Invalid disease')

    with open(data_path + '_training_new.pickle', 'rb') as f:
        train_ehr, train_label, train_timestamp  = pickle.load(f)
    with open(data_path + '_testing_new.pickle', 'rb') as f:
        test_ehr, test_label, test_timestamp = pickle.load(f)
    with open(data_path + '_validation_new.pickle', 'rb') as f:
        val_ehr, val_label, val_timestamp = pickle.load(f)

    # Convert timestamps to timegaps
    train_visit_timegaps = compute_time_gaps(train_timestamp)
    train_code_timegaps = compute_code_time_gaps(train_ehr, train_visit_timegaps)

    test_visit_timegaps = compute_time_gaps(test_timestamp)
    test_code_timegaps = compute_code_time_gaps(test_ehr, test_visit_timegaps)

    val_visit_timegaps = compute_time_gaps(val_timestamp)
    val_code_timegaps = compute_code_time_gaps(val_ehr, val_visit_timegaps)

    # Sample Output
    for a, b, c, d, e in zip(train_ehr, train_label, train_visit_timegaps, train_timestamp, train_code_timegaps):
        print("EHR:", a)
        print("Label:", b)
        print("Time Gaps:", c)
        print("Timestamps:", d)
        print("Code Time Gaps:", e)
        break

    # Save the data
    with open(data_path + '_training_with_timegaps.pickle', 'wb') as f:
        pickle.dump((train_ehr, train_label, train_visit_timegaps, train_timestamp, train_code_timegaps), f)
    with open(data_path + '_testing_with_timegaps.pickle', 'wb') as f:
        pickle.dump((test_ehr, test_label, test_visit_timegaps, test_timestamp, test_code_timegaps), f)
    with open(data_path + '_validation_with_timegaps.pickle', 'wb') as f:
        pickle.dump((val_ehr, val_label, val_visit_timegaps, val_timestamp, val_code_timegaps), f)

if __name__ == '__main__':

    # datas = ['Heartfailure', 'COPD', 'Kidney', 'Dementia', 'Amnesia', 'mimic']
    # # for target_disease in datas:
    # #     data_loading_and_modify_delta_t(target_disease)
    #
    # with open('./data/hf/hf'+ '_training_with_timegaps.pickle', 'rb') as f:
    #     train_ehr, train_label, train_visit_timegaps, train_timestamp, train_code_timegaps = pickle.load(f)
    # print(type(train_ehr))

    dir_ehr = './data/pancreas/toy_pancreas.csv'
    data = pd.read_csv(dir_ehr)
    ehr = data['code_int'].apply(lambda x: ast.literal_eval(x)).tolist()
    time_steps = data['time_gaps'].apply(lambda x: ast.literal_eval(x)).tolist()
    visit_consecutive_timegaps = data['consecutive_time_gaps'].apply(lambda x: ast.literal_eval(x)).tolist()
    code_timegaps = data['code_time_gaps'].apply(lambda x: ast.literal_eval(x)).tolist()

    print(type(ehr))
    print(type(time_steps))
    print(type(visit_consecutive_timegaps))
    print(type(code_timegaps))

    for a, b, c, d in zip(ehr, time_steps, visit_consecutive_timegaps, code_timegaps):
        print(a)
        print(b)
        print(c)
        print(d)
        print(len(a), len(b), len(c), len(d))
        assert len(a) == len(b) == len(c) == len(d)
    exit()