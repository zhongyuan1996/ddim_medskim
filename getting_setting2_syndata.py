import pandas as pd
import ast

datapath = './data/mimic/'

task = 'arf'

invasive_event_codes = [939, 967]
mapping_codes = [66, 0]

models = ['LSTM-MLP','MedDiffGa', 'EVA', 'LSTM-medGAN', 'LSTM-TabDDPM', 'synTEG', 'promptEHR', 'TWIN']
for model in models:
    synthetic_database = pd.read_csv(datapath + model + '_synthetic_3digmimic' + '.csv')
    #add a all zero column to the dataframe
    synthetic_database['ARF_LABEL'] = 0
    synthetic_database['PROC_ITEM_int'] = synthetic_database['PROC_ITEM_int'].apply(ast.literal_eval)

    #itrate though rows
    for index, row in synthetic_database.iterrows():
        for visit in row['PROC_ITEM_int']:
            for code in visit:
                if code in mapping_codes:
                    synthetic_database.at[index, 'ARF_LABEL'] = 1
                    break

    synthetic_database.to_csv(datapath + model + '_synthetic_3digmimic_' + task + '_setting2.csv', index=False)

    #count how many patients have ARF
    print(model, 'has', synthetic_database['ARF_LABEL'].sum(), 'patients with ARF')
    #the total number of patients
    print(model, 'has', synthetic_database.shape[0], 'patients in total')






































