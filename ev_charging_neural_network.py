import numpy as np
import pandas as pd

ev_charging_reports = pd.read_csv("datasets/EV charging reports.csv")
ev_charging_reports.head()

traffic_reports = pd.read_csv("datasets/Local traffic distribution.csv")
traffic_reports.head()

ev_charging_traffic = ev_charging_reports.merge(traffic_reports, left_on='Start_plugin_hour', right_on='Date_from')
ev_charging_traffic.head()

ev_charging_traffic.info()

drop_columns = ['session_ID', 'Garage_ID', 'User_ID', 'Shared_ID', 'Plugin_category', 'Duration_category', 'Start_plugin', 'Start_plugin_hour', 'End_plugout', 'End_plugout_hour', 'Date_from', 'Date_to']

ev_charging_traffic = ev_charging_traffic.drop(columns=drop_columns, axis=1)
ev_charging_traffic.head()

for column in ev_charging_traffic.columns:
    if ev_charging_traffic[column].dtype == 'object':
        ev_charging_traffic[column] = ev_charging_traffic[column].str.replace(',', '.')

ev_charging_traffic.head()

for column in ev_charging_traffic.columns:
    ev_charging_traffic[column] = ev_charging_traffic[column].astype(float)

ev_charging_traffic.head()

numerical_features = ev_charging_traffic.drop(['El_kWh'], axis=1).columns

X = ev_charging_traffic[numerical_features]

y = ev_charging_traffic['El_kWh']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, test_size=0.20, random_state=2)

print("Training size:", X_train.shape)
print("Testing size:", X_test.shape)

from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error

linear_test_predictions = linear_model.predict(X_test)

test_mse = mean_squared_error(y_test, linear_test_predictions)

print("Linear Regression - Test Set MSE:", test_mse)

import torch
from torch import nn
from torch import optim

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float)

y_train_tensor = torch.tensor(y_train.values, dtype=torch.float).view(-1, 1)

X_test_tensor = torch.tensor(X_test.values, dtype=torch.float)

y_test_tensor = torch.tensor(y_test.values, dtype=torch.float).view(-1, 1)

torch.manual_seed(42)

model = nn.Sequential(
    nn.Linear(26, 56),
    nn.ReLU(),
    nn.Linear(56, 26),
    nn.ReLU(),
    nn.Linear(26, 1)
)

loss = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.0007)

num_epochs = 3000

for epoch in range(num_epochs):
    outputs = model(X_train_tensor)
    mse = loss(outputs, y_train_tensor)
    mse.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], MSE Loss: {mse.item()}')

torch.save(model, 'models/model.pth')

model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    test_loss = loss(predictions, y_test_tensor)
    print('Neural Network - Test Set MSE:', test_loss.item())

model4500 = torch.load('models/model4500.pth')

model4500.eval()
with torch.no_grad():
    predictions = model4500(X_test_tensor)
    test_loss = loss(predictions, y_test_tensor)
    print('Neural Network - Test Set MSE:', test_loss.item())
