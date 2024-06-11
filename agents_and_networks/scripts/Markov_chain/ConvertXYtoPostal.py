import geopy
from geopy.extra.rate_limiter import RateLimiter
import pandas as pd
import os
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))


def get_zipcode(reverse, lat, lon):
    try:
        location = reverse((lat, lon),exactly_one=True)
        return location.raw['address']['postcode'].replace(" ","")
    except(KeyError,AttributeError):
        return 'a'

observation_file = os.path.join(script_dir, '..','..', 'outputs','trajectories','Louise', 'output_cell_time_dependent.csv')

df = pd.read_csv(observation_file)

geolocator = geopy.Nominatim(user_agent='postal_code_converter')
reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1)


tqdm.pandas()

unique_coordinates = df[['cellinfo.wgs84.lat','cellinfo.wgs84.lon']].drop_duplicates()

unique_coordinates['cellinfo.postal_code'] = unique_coordinates.progress_apply(lambda row: get_zipcode(reverse=reverse, lat=row['cellinfo.wgs84.lat'], lon=row['cellinfo.wgs84.lon']),axis=1)

df = pd.merge(df, unique_coordinates, on=['cellinfo.wgs84.lat', 'cellinfo.wgs84.lon'], how='left')

df = df.loc[df['cellinfo.postal_code'] != 'a']

df.to_csv(observation_file)