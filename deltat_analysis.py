import pandas as pd
import ast
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
# from filterpy.kalman import KalmanFilter
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


original_data_path = 'data/mimic/train_3digmimic.csv'
meddiffga_data_path = 'data/mimic/MedDiffGa_synthetic_3dig_newdeltatmimic.csv'
val_data_path = 'data/mimic/val_3digmimic.csv'
testing_data_path = 'data/mimic/test_3digmimic.csv'

def read_and_pad_data(data_path, max_len = 20, demo_name = 'Demographic', time_name = 'time_gaps'):
    data = pd.read_csv(data_path)
    demo, time_labels, non_padded_time = [], [], []
    for idx, row in data.iterrows():
        demographics = ast.literal_eval(row[demo_name])
        time_gaps = ast.literal_eval(row[time_name])
        non_padded_time.append(time_gaps)
        if len(time_gaps) < max_len:
            time_gaps = time_gaps + [-1] * (max_len - len(time_gaps))
        else:
            time_gaps = time_gaps[:max_len]
        demo.append(demographics)
        time_labels.append(time_gaps)
    return demo, time_labels, non_padded_time

og_data = pd.read_csv(original_data_path)
meddiff_data = pd.read_csv(meddiffga_data_path)
val_data = pd.read_csv(val_data_path)
test_data = pd.read_csv(testing_data_path)

og_demo, og_time_labels, og_non_padded_time = read_and_pad_data(original_data_path, max_len = 50, demo_name = 'Demographic', time_name = 'consecutive_time_gaps')
meddiff_demo, meddiff_time_labels, meddiff_non_padded_time = read_and_pad_data(meddiffga_data_path,  max_len = 50, demo_name = 'demo', time_name = 'time_gaps')
val_demo, val_time_labels, val_non_padded_time = read_and_pad_data(val_data_path,  max_len = 50, demo_name = 'Demographic', time_name = 'consecutive_time_gaps')
test_demo, test_time_labels, test_non_padded_time = read_and_pad_data(testing_data_path,  max_len = 50, demo_name = 'Demographic', time_name = 'consecutive_time_gaps')

#ARIMA######################
# p, d, q = 2, 1, 2
# flattened_og_non_padded_time = [item for sublist in og_non_padded_time for item in sublist]
# arima_model = ARIMA(flattened_og_non_padded_time, order=(p, d, q))
# arima_model_fit = arima_model.fit()
#
# flattened_test_non_padded_time = [item for sublist in test_non_padded_time for item in sublist]
#
# n_forecast_steps = len(flattened_test_non_padded_time)
# forecast = arima_model_fit.forecast(steps=n_forecast_steps)
#
# arima_mse = mean_squared_error(flattened_test_non_padded_time, forecast)
# print(f"Mean Squared Error of ARIMA model: {arima_mse}")
#
# #SVR and GBRT style fearures######################

features, labels = [], []
for demo, gaps in zip(og_demo, og_non_padded_time):
    for i in range(len(gaps)-1):
        features.append(demo + [gaps[i]])
        # features.append([gaps[i]])
        labels.append(gaps[i+1])

test_features, test_labels = [], []
for demo, gaps in zip(test_demo, test_non_padded_time):
    for i in range(len(gaps)-1):
        test_features.append(demo + [gaps[i]])
        # test_features.append([gaps[i]])
        test_labels.append(gaps[i+1])
#
from sklearn.model_selection import GridSearchCV
#
# svr_param_grid = {
#     'C': [1, 10, 100],
#     'gamma': [0.01, 0.1, 1],
#     'epsilon': [0.1, 0.2, 0.5]
# }
# #
# # # Grid search with cross-validation for SVR
# svr_grid_search = GridSearchCV(SVR(kernel='rbf'), svr_param_grid, cv=5)
# svr_grid_search.fit(features, labels)
# # svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.5)
# # svr.fit(features, labels)
# # print(f"Best parameters for SVR: {svr_grid_search.best_params_}")
# predicted_labels = svr_grid_search.predict(test_features)
# svr_mse = mean_squared_error(test_labels, predicted_labels)
# print(f"Mean Squared Error of SVR model: {svr_mse}")
# #
# # Define parameter grid for GBRT
# gbrt_param_grid = {
#     'max_depth': [3, 5, 10],
#     'n_estimators': [100, 200, 300],
#     'learning_rate': [0.01, 0.1, 0.2]
# }
#
# # rf_param_grid = {
# #     'n_estimators': [100, 200, 300],
# #     'max_depth': [None, 10, 20, 30],
# #     'min_samples_split': [2, 5, 10]
# # }
#
# Grid search with cross-validation for GBRT
# gbrt_grid_search = GridSearchCV(GradientBoostingRegressor(), gbrt_param_grid, cv=5)
# gbrt_grid_search.fit(features, labels)
# # gbrt = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
# # gbrt.fit(features, labels)
# # print(f"Best parameters for GBRT: {gbrt_grid_search.best_params_}")
# predicted_labels = gbrt_grid_search.predict(test_features)
# gbrt_mse = mean_squared_error(test_labels, predicted_labels)
# print(f"Mean Squared Error of GBRT model: {gbrt_mse}")

#
#
# # Grid search with cross-validation for Random Forest
# # rf_grid_search = GridSearchCV(RandomForestRegressor(random_state=42), rf_param_grid, cv=5, n_jobs=-1)
# rf = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=10)
# rf.fit(features, labels)
# # print(f"Best parameters for Random Forest: {rf_grid_search.best_params_}")
#
# # Predict and evaluate the Random Forest model
# rf_predicted_labels = rf.predict(test_features)
# rf_mse = mean_squared_error(test_labels, rf_predicted_labels)
# print(f"Mean Squared Error of Random Forest model: {rf_mse}")

#MedDiffGa######################
#only calculate the non-padded time

# aligned_meddiff = []
# aligned_og = []
#
# # Aligning the data and filtering out the padding
# for meddiff_patient, og_patient in zip(meddiff_time_labels, test_time_labels):
#     for meddiff_gap, og_gap in zip(meddiff_patient, og_patient):
#         if og_gap != -1:  # Exclude padding
#             aligned_meddiff.append(meddiff_gap)
#             aligned_og.append(og_gap)
#
# # Calculate MSE
# mse = mean_squared_error(aligned_og, aligned_meddiff)
# print(f"Mean Squared Error of diffusion: {mse}")

#Kalman Filter######################
# def run_kalman_filter_for_prediction(time_series):
#     kf = KalmanFilter(dim_x=2, dim_z=1)
#
#     # Initialize the Kalman Filter as before
#     kf.x = np.array([0., 0.])
#     kf.P *= 1000
#     kf.F = np.array([[1, 1], [0, 1]])
#     kf.H = np.array([[1, 0]])
#     kf.R = np.array([[1]])
#     kf.Q = np.array([[1, 0], [0, 1]])
#
#     predictions = []
#     for i in range(len(time_series) - 1):
#         kf.update([time_series[i]])
#         kf.predict()
#         predictions.append(kf.x[0])  # Prediction for the next time step
#
#     return predictions
#
# # Flatten and run the filter
# flattened_og_non_padded_time = [item for sublist in og_non_padded_time for item in sublist if item != -1]
# flattened_test_non_padded_time = [item for sublist in test_non_padded_time for item in sublist if item != -1]
#
# # Get predictions
# train_predictions = run_kalman_filter_for_prediction(flattened_og_non_padded_time)
# test_predictions = run_kalman_filter_for_prediction(flattened_test_non_padded_time)
#
# # Compute MSE, skipping the last observation since there's no prediction for it
# train_mse = mean_squared_error(flattened_og_non_padded_time[1:], train_predictions)
# test_mse = mean_squared_error(flattened_test_non_padded_time[1:], test_predictions)
# print(f"Train Mean Squared Error: {train_mse}")
# print(f"Test Mean Squared Error: {test_mse}")

# #LSTM######################
# class lstm_time(nn.Module):
#     def __init__(self, input_embedding_dim, hidden_embedding_dim, num_layers, dropout):
#         super(lstm_time, self).__init__()
#         self.input_embedding_dim = input_embedding_dim
#         self.hidden_embedding_dim = hidden_embedding_dim
#         self.num_layers = num_layers
#         self.dropout = dropout
#         self.lstm = nn.LSTM(input_size = input_embedding_dim, hidden_size = hidden_embedding_dim, num_layers = num_layers, dropout = dropout)
#         self.linear = nn.Linear(hidden_embedding_dim, 1)
#         self.time_to_embedding = nn.Sequential(nn.Linear(1, input_embedding_dim), nn.ReLU(), nn.Dropout(dropout))
#         self.demo_to_embedding = nn.Sequential(nn.Linear(76, input_embedding_dim), nn.ReLU(), nn.Dropout(dropout))
#
#     def forward(self, time, demo, seq_lengths):
#         time_embedding = self.time_to_embedding(time)
#         demo_embedding = self.demo_to_embedding(demo).unsqueeze(1).expand(-1, time.size(1), -1)
#         combined_embedding = time_embedding + demo_embedding
#
#         h,c = self.lstm(combined_embedding)
#         linear_out = self.linear(h)
#         return linear_out
#
# lstm = lstm_time(128, 128, 1, 0.2)
# critereon = nn.MSELoss()
# optimizer = torch.optim.Adam(lstm.parameters(), lr = 0.001)
#
# og_demo_tensor = torch.tensor(og_demo, dtype=torch.float32)
# og_time_labels_tensor = torch.tensor(og_time_labels, dtype=torch.float32)
#
# seq_lengths = torch.tensor([len([t for t in seq if t != -1]) for seq in og_time_labels])
# # og_time_labels_tensor, sorted_indices = og_time_labels_tensor.sort(0, descending=True)
# # og_demo_tensor = og_demo_tensor[sorted_indices]
# # seq_lengths = seq_lengths[sorted_indices]
#
# test_demo_tensor = torch.tensor(test_demo, dtype=torch.float32)
# test_time_labels_tensor = torch.tensor(test_time_labels, dtype=torch.float32)
#
# test_seq_lengths = torch.tensor([len([t for t in seq if t != -1]) for seq in test_time_labels])
# # test_time_labels_tensor, test_sorted_indices = test_time_labels_tensor.sort(0, descending=True)
# # test_demo_tensor = test_demo_tensor[test_sorted_indices]
# # test_seq_lengths = test_seq_lengths[test_sorted_indices]
#
# from torch.utils.data import Dataset, DataLoader
#
# class TimeGapDataset(Dataset):
#     def __init__(self, demographics, time_labels, seq_lengths):
#         self.demographics = demographics
#         self.time_labels = time_labels
#         self.seq_lengths = seq_lengths
#
#     def __len__(self):
#         return len(self.demographics)
#
#     def __getitem__(self, idx):
#         return self.demographics[idx], self.time_labels[idx], self.seq_lengths[idx]

# Create the Dataset and DataLoader
# dataset = TimeGapDataset(og_demo_tensor, og_time_labels_tensor, seq_lengths)
# data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
# testset = TimeGapDataset(test_demo_tensor, test_time_labels_tensor, test_seq_lengths)
# test_data_loader = DataLoader(testset, batch_size=32, shuffle=False)
#
# num_epochs = 50  # Number of epochs for training, adjust as needed
#
# lstm.train()  # Set the model to training mode
# for epoch in range(num_epochs):
#     total_loss = 0
#     for demographics, time_labels, lengths in data_loader:
#         # Reshape data for batch processing
#         # Assuming time_labels is a 2D tensor of shape (batch_size, seq_length)
#         time_input = time_labels[:, :].unsqueeze(-1)  # Use all but the last time gap as input
#         time_target = time_labels[:,:]  # Use all but the first time gap as target
#
#         # Zero the parameter gradients
#         optimizer.zero_grad()
#
#         # Forward pass
#         outputs = lstm(time_input, demographics, lengths)
#
#         # Calculate loss; only consider the non-padded part for loss computation
#         valid_indices = time_target != -1  # Exclude padding
#         outputs = outputs.squeeze(-1)
#         loss = critereon(outputs[valid_indices], time_target[valid_indices])
#
#         # Backward pass and optimize
#         loss.backward()
#         optimizer.step()
#
#         total_loss += loss.item()
#
#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(data_loader)}")
#
# lstm.eval()  # Set the model to evaluation mode
# total_mse = 0
# with torch.no_grad():  # No need to track gradients during evaluation
#     for demographics, time_labels, lengths in test_data_loader:  # Assuming test_data_loader is defined
#         time_input = time_labels[:, :].unsqueeze(-1)
#         time_target = time_labels[:, :]
#
#         outputs = lstm(time_input, demographics, lengths)
#         valid_indices = time_target != -1
#         outputs = outputs.squeeze(-1)
#         mse = critereon(outputs[valid_indices], time_target[valid_indices])
#         total_mse += mse.item()
#
# print(f"Test Mean Squared Error of lstm: {total_mse / len(test_data_loader)}")

from models.baseline import Attention, PositionalEncoding
#simple transformer######################
class transformer_time(nn.Module):
    def __init__(self, input_embedding_dim, hidden_embedding_dim, num_layers, dropout):
        super(transformer_time, self).__init__()
        self.input_embedding_dim = input_embedding_dim
        self.hidden_embedding_dim = hidden_embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        # self.lstm = nn.LSTM(input_size = input_embedding_dim, hidden_size = hidden_embedding_dim, num_layers = num_layers, dropout = dropout)
        self.linear = nn.Linear(hidden_embedding_dim, 1)
        self.time_to_embedding = nn.Sequential(nn.Linear(1, input_embedding_dim), nn.ReLU(), nn.Dropout(dropout))
        self.demo_to_embedding = nn.Sequential(nn.Linear(76, input_embedding_dim), nn.ReLU(), nn.Dropout(dropout))
        self.transformer = Attention(hidden_embedding_dim, 2, dropout)

    def forward(self, time, demo, seq_lengths):
        time_embedding = self.time_to_embedding(time)
        demo_embedding = self.demo_to_embedding(demo).unsqueeze(1).expand(-1, time.size(1), -1)
        combined_embedding = time_embedding + demo_embedding

        h,_ = self.transformer(combined_embedding, combined_embedding, combined_embedding)

        # h,c = self.lstm(combined_embedding)
        linear_out = self.linear(h)
        return linear_out

trans = transformer_time(128, 128, 1, 0.2)
critereon = nn.MSELoss()
optimizer = torch.optim.Adam(trans.parameters(), lr = 0.001)

og_demo_tensor = torch.tensor(og_demo, dtype=torch.float32)
og_time_labels_tensor = torch.tensor(og_time_labels, dtype=torch.float32)

seq_lengths = torch.tensor([len([t for t in seq if t != -1]) for seq in og_time_labels])
# og_time_labels_tensor, sorted_indices = og_time_labels_tensor.sort(0, descending=True)
# og_demo_tensor = og_demo_tensor[sorted_indices]
# seq_lengths = seq_lengths[sorted_indices]

test_demo_tensor = torch.tensor(test_demo, dtype=torch.float32)
test_time_labels_tensor = torch.tensor(test_time_labels, dtype=torch.float32)

test_seq_lengths = torch.tensor([len([t for t in seq if t != -1]) for seq in test_time_labels])
# test_time_labels_tensor, test_sorted_indices = test_time_labels_tensor.sort(0, descending=True)
# test_demo_tensor = test_demo_tensor[test_sorted_indices]
# test_seq_lengths = test_seq_lengths[test_sorted_indices]

from torch.utils.data import Dataset, DataLoader

class TimeGapDataset(Dataset):
    def __init__(self, demographics, time_labels, seq_lengths):
        self.demographics = demographics
        self.time_labels = time_labels
        self.seq_lengths = seq_lengths

    def __len__(self):
        return len(self.demographics)

    def __getitem__(self, idx):
        return self.demographics[idx], self.time_labels[idx], self.seq_lengths[idx]

# Create the Dataset and DataLoader
dataset = TimeGapDataset(og_demo_tensor, og_time_labels_tensor, seq_lengths)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
testset = TimeGapDataset(test_demo_tensor, test_time_labels_tensor, test_seq_lengths)
test_data_loader = DataLoader(testset, batch_size=32, shuffle=False)

num_epochs = 50  # Number of epochs for training, adjust as needed

trans.train()  # Set the model to training mode
for epoch in range(num_epochs):
    total_loss = 0
    for demographics, time_labels, lengths in data_loader:
        # Reshape data for batch processing
        # Assuming time_labels is a 2D tensor of shape (batch_size, seq_length)
        time_input = time_labels[:, :].unsqueeze(-1)  # Use all but the last time gap as input
        time_target = time_labels[:,:]  # Use all but the first time gap as target

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = trans(time_input, demographics, lengths)

        # Calculate loss; only consider the non-padded part for loss computation
        valid_indices = time_target != -1  # Exclude padding
        outputs = outputs.squeeze(-1)
        loss = critereon(outputs[valid_indices], time_target[valid_indices])

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(data_loader)}")

trans.eval()  # Set the model to evaluation mode
total_mse = 0
with torch.no_grad():  # No need to track gradients during evaluation
    for demographics, time_labels, lengths in test_data_loader:  # Assuming test_data_loader is defined
        time_input = time_labels[:, :].unsqueeze(-1)
        time_target = time_labels[:, :]

        outputs = trans(time_input, demographics, lengths)
        valid_indices = time_target != -1
        outputs = outputs.squeeze(-1)
        mse = critereon(outputs[valid_indices], time_target[valid_indices])
        total_mse += mse.item()

print(f"Test Mean Squared Error of lstm: {total_mse / len(test_data_loader)}")