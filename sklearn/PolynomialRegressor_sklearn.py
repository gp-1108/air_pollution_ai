import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

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
  
  # Train Polynomial regressor plotting the learning curve using learning_curve
  degrees = [1, 2, 3, 4, 5]
  for degree in degrees:
    print(f"Training Polynomial regressor with degree {degree}")
    polynomial_features = PolynomialFeatures(degree=degree)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    train_sizes, train_scores, test_scores = learning_curve(pipeline, X_train, y_train, scoring='neg_mean_squared_error')
    train_scores_mean = -train_scores.mean(axis = 1)
    #plt.plot(train_sizes, train_scores_mean, label="Training error")
    #plt.title(f"Learning curve for Polynomial regressor with degree {degree}")
    #plt.xlabel("Training set size")
    #plt.ylabel("MSE")
    #plt.legend()
    #plt.savefig(f"/home/enrico/Desktop/air_pollution_ai/sklearn/results/Polynomial_regressor/PolynomialRegressor_degree_{degree}_cause_{cause}.png")
    #plt.clf()

    # Write in a txt file the test scores
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100

    # Scattering plot range 0-1
    plt.scatter(y_val, y_pred)
    plt.title(f"Scattering plot for Polynomial regressor with degree {degree}")
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(f"/home/enrico/Desktop/air_pollution_ai/sklearn/results/Polynomial_regressor/scatter_plots/Scattering_plot_PolynomialRegressor_degree_{degree}_cause_{cause}.png")
    plt.clf()

    #with open(f"/home/enrico/Desktop/air_pollution_ai/sklearn/results/Polynomial_regressor/Polynomial_regressor_scores.txt", "a") as f:
    #  f.write(f"degree: {degree}, cause {cause}, MSE: {mse}, MAE: {mae}, mape: {mape}, R2: {r2}\n")