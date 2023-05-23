import pandas as pd
import numpy as np

df = pd.read_csv('./data_air_index/unfiltered_day_count_data.csv', dtype={"state_code": str, "county_code": str})

# Remove state_code equals to CC
df = df[df["state_code"] != "CC"]

# Create column fips
df["fips"] = (df["state_code"] + df["county_code"]).astype(int)

# Drop state_code and county_code
df = df.drop(["state_code", "county_code"], axis=1)

print("Pre sorting")
print(df.shape)

# For each fips, for each year, for each parameter, get the row with highest valid_day_count
df = df.sort_values(by=["valid_day_count"], ascending=False)
df = df.drop_duplicates(subset=["fips", "year", "parameter_code"], keep="first")

print("Post sorting")
print(df.shape)

parameter_list = [85101, 88101, 44201, 42602, 42401, 42101]

# For each fips, for each year, create a column for each parameter
final_df = pd.DataFrame(columns=["fips", "year", "parameter_85101", "parameter_88101", "parameter_44201", "parameter_42602", "parameter_42401", "parameter_42101"])
n_fips = df["fips"].nunique()
count = 1
for fips in df["fips"].unique():
    print("Processing fips {} ({}/{})".format(fips, count, n_fips))
    for year in df["year"].unique():
        row = {"fips": fips, "year": year}
        for parameter in parameter_list:
            if(df[(df["fips"] == fips) & (df["year"] == year) & (df["parameter_code"] == parameter)]["arithmetic_mean"].size == 0):
                row["parameter_{}".format(parameter)] = "null"
            else:
                row["parameter_{}".format(parameter)] = df[(df["fips"] == fips) & (df["year"] == year) & (df["parameter_code"] == parameter)]["arithmetic_mean"].values[0]
        final_df = pd.concat([final_df, pd.DataFrame([row])], ignore_index=True)
    count += 1


print("Post final_df")
print(final_df.head())
print(final_df.shape)

# Save to csv
final_df.to_csv("./data_air_index/one_row_air_data.csv", index=False)