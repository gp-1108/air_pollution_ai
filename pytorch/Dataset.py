from torch.utils import data
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

class AirDataSet(data.Dataset):
  def __init__(self, cause_id, validation_size=0.2, train=True):
    super().__init__()
    file_path = 'training_data/cause_id_{}.csv'.format(cause_id)
    if (cause_id == "-1"):
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
    
    training_df.columns = training_df.columns.astype(str)  # Convert column names to strings
  
    # Split the data into training and validation sets
    if train:
      self.data, _, self.label, _ = train_test_split(
        training_df.drop(columns=['mortality_rate']).to_numpy(),
        training_df[['mortality_rate']].to_numpy(),
        test_size=validation_size,
        random_state=42
      )
    else:
      _, self.data, _, self.label = train_test_split(
        training_df.drop(columns=['mortality_rate']).to_numpy(),
        training_df[['mortality_rate']].to_numpy(),
        test_size=validation_size,
        random_state=42
      )
    
    # Normalize the data using StandardScaler
    scaler = StandardScaler()
    self.data = scaler.fit_transform(self.data)

    self.data = torch.from_numpy(self.data)
    self.label = torch.from_numpy(self.label)


  def __len__(self):
    # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
    return self.data.shape[0]

  def __getitem__(self, idx):
    # Return the idx-th data point of the dataset
    # If we have multiple things to return (data point and label), we can return them as tuple
    data_point = self.data[idx]
    data_label = self.label[idx]
    return data_point, data_label
