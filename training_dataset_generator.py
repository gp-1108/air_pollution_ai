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

final_df = df_air.merge(df_mortality, on=['fips', 'year'])
