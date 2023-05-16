from dotenv import load_dotenv
load_dotenv()
import requests
import os
import urllib3
import ssl
import pprint
pp = pprint.PrettyPrinter(indent=4)


# Adding code snippet from https://stackoverflow.com/questions/71603314/ssl-error-unsafe-legacy-renegotiation-disabled
# to get around SSL error
class CustomHttpAdapter (requests.adapters.HTTPAdapter):
  # "Transport adapter" that allows us to use custom ssl_context.

  def __init__(self, ssl_context=None, **kwargs):
    self.ssl_context = ssl_context
    super().__init__(**kwargs)

  def init_poolmanager(self, connections, maxsize, block=False):
    self.poolmanager = urllib3.poolmanager.PoolManager(
      num_pools=connections, maxsize=maxsize,
      block=block, ssl_context=self.ssl_context)


def get_legacy_session():
  ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
  ctx.options |= 0x4  # OP_LEGACY_SERVER_CONNECT
  session = requests.session()
  session.mount('https://', CustomHttpAdapter(ctx))
  return session

def get_data(listOfparams: list, state: str, county: str, year: str):
  base_url = "https://aqs.epa.gov/data/api/annualData/byCounty"

  # Creating a string of parameters to pass into the API call
  # For each entry in the list, add it to the string separated by a comma
  params = ""
  for param in listOfparams:
    params += param + ","
  # Removing the last comma
  params = params[:-1]

  print("Retrieving the following paramers: {params}".format(params=params))
  response = get_legacy_session().get(
    base_url,
    params={
      "email": os.getenv("email"),
      "key": os.getenv("key"),
      "param": params,
      "bdate": "{}0101".format(year),
      "edate": "{}1231".format(year),
      "state": state,
      "county": county,
    }
  )

  # Iterating through the response
  responses = {}
  for param in listOfparams:
    responses[param] = []
  for item in response.json()["Data"]:
    dict = {
      "missing": False,
      "parameter": item["parameter"],
      "parameterID": item["parameter_code"],
      "valid_day_count": item["valid_day_count"],
      "units_of_measure": item["units_of_measure"],
      "arithmetic_mean": item["arithmetic_mean"],
      "fips": "{}{}".format(item["state_code"], item["county_code"]),
      "year": year,
    }
    responses[dict["parameterID"]].append(dict)

  # For each parameter, sort the list of responses by valid_day_count in descending order
  for param in listOfparams:
    responses[param].sort(key=lambda x: x["valid_day_count"], reverse=True)
  
  # For each parameter, get the top response
  res = []
  for param in listOfparams:
    if len(responses[param]) > 0:
      res.append(responses[param][0])
    else:
      res.append({
        "missing": True,
        "parameter": "null",
        "parameterID": param,
        "valid_day_count": 0,
        "units_of_measure": "null",
        "arithmetic_mean": "null",
        "fips": "{}{}".format(state, county),
        "year": year,
      })

  pp.pprint(res)
  
  return res

def retrieve_data_for_county(county:str, state:str):
  listOfparams = [
    "85101", # PM10 - Local Conditions
    "88101", # PM2.5 - Local Conditions
    "44201", # Ozone
    "42602", # Nitrogen dioxide
    "42401", # Sulfur dioxide
    "42101", # Carbon monoxide
  ]

  # get_data supports max 5 params per request
  # so we need to split the list into chunks of 5
  chunks = [listOfparams[x:x+5] for x in range(0, len(listOfparams), 5)]
  all_ans = []
  for chunk in chunks:
    for year in range(1980, 2014): # [TODO change to 1980]
      all_ans = all_ans + get_data(chunk, "06", "001", str(year))

  return all_ans
