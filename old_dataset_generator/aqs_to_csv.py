from aqs_req_fetcher import retrieve_data_for_county
import pprint
import pandas as pd

pp = pprint.PrettyPrinter(indent=4)


if __name__ == "__main__":
  mortality_data = pd.read_csv("data_mortality_ratio/respiratory_disease_mortality_rate_usa_clean.csv")
  mortality_data = mortality_data[["fips"]]
  mortality_data = mortality_data.drop_duplicates()
  mortality_data.reset_index(drop=True, inplace=True)
  
  # For each row in the mortality_data dataframe, call the retrieve_data_for_county function
  # and write the returned data to a csv file
  for index, row in mortality_data.iterrows():
    fips = (str) (row["fips"])
    # The last 3 digits of the fips code are the county code
    county = fips[-3:]
    # The remaning digits are the state code
    state = fips[:-3]
    print("Retrieving data for fips {fips}".format(fips=fips))
    ans = retrieve_data_for_county(county, state)
    df = pd.DataFrame(ans)
    df = df.drop(columns=["missing", "valid_day_count"])
    df = df[["fips", "year", "parameterID", "parameter", "arithmetic_mean", "units_of_measure"]]
    df.to_csv("data_air_index/{location}.csv".format(location=fips), index=False)
    print("Wrote data for fips {fips}".format(fips=fips))
