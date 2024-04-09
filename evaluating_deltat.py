import pandas as pd


train_data = pd.read_csv('./data/mimic/train_3digmimic.csv')
test_data = pd.read_csv('./data/mimic/test_3digmimic.csv')
val_data = pd.read_csv('./data/mimic/val_3digmimic.csv')

train_timegap = train_data['consecutive_time_gaps']
test_timegap = test_data['consecutive_time_gaps']
val_timegap = val_data['consecutive_time_gaps']




