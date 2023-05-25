import pandas as pd
import numpy as np

df_air = pd.read_csv('data_air_index/one_row_air_data.csv')
df_mortality = pd.read_csv('data_mortality_ratio/respiratory_disease_mortality_rate_usa_clean.csv')

# Order by year
df_air = df_air.sort_values(by=['fips', 'year'])

# Ffill
df_air = df_air.fillna(method='ffill')

# Drop rows with NaN
df_air = df_air.dropna()

# Drop the rows with fips that are not in the mortality dataset
df_air = df_air[df_air['fips'].isin(df_mortality['fips'])]

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
"""

final_df = df_air.merge(df_mortality, on=['fips', 'year'])

# For each cause_id create a new csv and save it
for cause_id in range(508, 521):
  df = final_df.copy()
  df = df.rename(columns={'rate': 'mortality_rate'})
  df = df[df['cause_id'] == cause_id]
  df.to_csv(f'training_data/cause_id_{cause_id}.csv', index=False)
  print(f"Saved cause_id_{cause_id}.csv with shape: {df.shape}")


# Save it to csv
final_df.to_csv('training_data/all_data.csv', index=False)

