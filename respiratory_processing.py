import pandas as pd

df = pd.read_csv('data_mortality_ratio/respiratory_disease_mortality_rate_usa.csv')
# Dtypes are:
# location_id      int64
# fips           float64
# cause_id         int64
# sex_id           int64
# year_id          int64
# rate           float64
# lower          float64
# upper          float64

# Extracting just the fips, year, and rate columns
df = df[['fips', 'year_id', 'rate']]

# Filtering out any rows that have a fips value with less than 4 digits
df = df[df['fips'] > 999]

# Converting the fips column to an int
df['fips'] = df['fips'].astype(int)

# Exporting the new dataframe to a csv file
df.to_csv('data_mortality_ratio/respiratory_disease_mortality_rate_usa_clean.csv', index=False)