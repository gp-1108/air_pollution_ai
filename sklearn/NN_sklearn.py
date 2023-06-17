import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPRegressor


causes_list = ["508", "509", "510", "511", "512", "513", "514", "515", "516", "520", "-1"]

# For each cause load the dataset
for cause in causes_list:
  print(f"Loading dataset for cause_id {cause}")
  file_path = f"training_data/cause_id_{cause}.csv"
  if (cause == "-1"):
    file_path = "training_data/all_data.csv"
  training_df = pd.read_csv(file_path,
    dtype={
      "parameter_85101": "float32",
      "parameter_88101": "float32",
      "parameter_44201": "float32",
      "parameter_42602": "float32",
      "parameter_42401": "float32",
      "parameter_42101": "float32",
      "mortality_rate": "float32",
      "cause_id": "float32",
    })
  training_df = training_df.drop(columns=["fips", "year"])

  if (cause != "-1"):
    training_df = training_df.drop(columns=["cause_id"])
  
  training_df.columns = training_df.columns.astype(str)  # Convert column names to strings

  # Split the data into training and validation sets
  train_df, val_df = train_test_split(training_df, test_size=0.2, random_state=42)


  train_df = normalize(train_df)
  val_df = normalize(val_df)
  train_df = pd.DataFrame(train_df, columns=training_df.columns)
  val_df = pd.DataFrame(val_df, columns=training_df.columns)

  X_train = train_df.drop(columns=["mortality_rate"]).to_numpy()
  y_train = train_df[["mortality_rate"]].to_numpy()
  X_val = val_df.drop(columns=["mortality_rate"]).to_numpy()
  y_val = val_df[["mortality_rate"]].to_numpy()
  
  # Train Neural Network plotting the learning curve using learning_curve
  for neurons in [20]:
    mlp = MLPRegressor(hidden_layer_sizes=(20, 20, 20), max_iter=100, alpha=0.0001,
                      solver='sgd', verbose=10, random_state=42, tol=0.000000001)
    train_sizes, train_scores, test_scores = learning_curve(mlp, X_train, y_train.ravel(), scoring='neg_mean_squared_error')
    train_scores_mean = -train_scores.mean(axis = 1)
    plt.plot(train_sizes, train_scores_mean, label = 'Training error')
    plt.ylabel('MSE', fontsize = 14)
    title = 'Neural network model ({neurons},{neurons},{neurons})'
    plt.title(title, fontsize = 18, y = 1.03)
    plt.legend()
    plt.savefig(f"/home/enrico/Desktop/air_pollution_ai/sklearn/results/NN/NN_{neurons}_{neurons}_{neurons}_cause_{cause}.png")
    plt.clf()

    # Train Neural Network
    mlp.fit(X_train, y_train.ravel())
    y_pred = mlp.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100
    r2 = r2_score(y_val, y_pred)

    with open(f"/home/enrico/Desktop/air_pollution_ai/sklearn/results/NN/NN_scores.txt", "a") as f:
      f.write(f"neurons: 20/20/20, cause {cause}, MSE: {mse}, MAE: {mae}, mape: {mape}, R2: {r2}\n")