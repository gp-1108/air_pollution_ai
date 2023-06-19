import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Here you must specify the cause_id you want to train on
"""
cause_id columns meaning:
  "508","Chronic respiratory diseases"
  "509","Chronic obstructive pulmonary disease"
  "510","Pneumoconiosis"
  "511","Silicosis"
  "512","Asbestosis"
  "513","Coal workers pneumoconiosis"
  "514","Other pneumoconiosis"
  "515","Asthma"
  "516","Interstitial lung disease and pulmonary sarcoidosis"
  "520","Other chronic respiratory diseases"
  "-1","All causes"
"""
cause_id = -1

torch.manual_seed(42) # Setting the seed
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available(): 
  torch.cuda.manual_seed(42)
  torch.cuda.manual_seed_all(42)
print("Current device is {}".format(device))

class LinearClassifier(nn. Module):
  def __init__(self, input_dim):
    super().__init__()
    self.linear = nn.Linear(input_dim, 1)

  def forward(self, x):
    x = x.to(self.linear.weight.dtype)
    x = self.linear(x)
    return x

class AirDataSet(data.Dataset):
  def __init__(self):
    super().__init__()
    file_path = 'training_data/cause_id_{}.csv'.format(cause_id)
    if (cause_id == -1):
      file_path = 'training_data/all_data.csv'
    training_df = pd.read_csv(file_path,
      dtype={
        'parameter_85101': 'float32',
        'parameter_88101': 'float32',
        'parameter_44201': 'float32',
        'parameter_42602': 'float32',
        'parameter_42401': 'float32',
        'parameter_42101': 'float32',
        'mortality_rate': 'float32',
        'cause_id': 'float32',
      })
    training_df = training_df.drop(columns=['fips', 'year'])

    if (cause_id != -1):
      training_df = training_df.drop(columns=['cause_id'])

    # One-hot encode cause_id column
    if cause_id == -1:
      encoder = OneHotEncoder(sparse=False, categories='auto')
      cause_id_encoded = encoder.fit_transform(training_df[['cause_id']])
      training_df = training_df.drop(columns=['cause_id'])
      training_df = pd.concat([training_df, pd.DataFrame(cause_id_encoded)], axis=1)
    
    print(training_df.head())
    training_df.columns = training_df.columns.astype(str)  # Convert column names to strings


    # Use only the first 1000 rows for training
    # training_df = training_df[:1000]

    # For each row we have the following columns corresponding to features:
    # -parameter_85101
    # -parameter_88101
    # -parameter_44201
    # -parameter_42602
    # -parameter_42401
    # -parameter_42101
    # -cause_id (if cause_id != -1)

    # And the following columns corresponding to labels:
    # -rate

    # We want to predict the rate based on the parameters
    self.data = torch.from_numpy(training_df.drop(columns=['mortality_rate']).to_numpy())
    self.label = torch.from_numpy(training_df[['mortality_rate']].to_numpy())
    # Normalize the data using StandardScaler
    scaler = StandardScaler()
    self.data = scaler.fit_transform(training_df.drop(columns=['mortality_rate']))
    self.label = torch.from_numpy(training_df[['mortality_rate']].to_numpy(dtype='float32'))


  def __len__(self):
    # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
    return self.data.shape[0]

  def __getitem__(self, idx):
    # Return the idx-th data point of the dataset
    # If we have multiple things to return (data point and label), we can return them as tuple
    data_point = self.data[idx]
    data_label = self.label[idx]
    return data_point, data_label


dataset = AirDataSet()
model = LinearClassifier(input_dim=dataset.data.shape[1])
train_data_loader = data.DataLoader(dataset, batch_size=32, shuffle=True)
loss_func = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

model.to(device)

losses = []
epochs = []
def train_model(model, optimizer, data_loader, loss_module, num_epochs=1000):
  model.train()
  with open('./plots/cause_id_{}.txt'.format(cause_id), 'w') as f:
    for epoch in range(num_epochs):
      epochs.append(epoch)
      for data_point, data_label in data_loader:
        data_point = data_point.to(device)
        data_label = data_label.to(device)
        data_label = data_label.squeeze(dim=1)

        output = model(data_point)
        output = output.squeeze(dim=1)


        loss = loss_module(output, data_label.float())
      
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      losses.append(loss.item())
      output = f"Epoch {epoch} loss: {loss.item()}"
      print(output)
      #print(output, file = f)

train_model(model, optimizer, train_data_loader, loss_func, num_epochs=10)

# Plotting the loss
plt.plot(epochs, losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f"Loss for cause_id {cause_id}")


#plt.savefig('./plots/cause_id_{}.png'.format(cause_id))