import fiona
from shapely.geometry import Point, shape
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from bokeh.io import show, output_file
from bokeh.plotting import figure
from bokeh.palettes import PuOr, Viridis6, PuRd
from bokeh.models import (GMapPlot, GMapOptions, ColumnDataSource, Patches, \
							DataRange1d, PanTool, WheelZoomTool, BoxSelectTool,
 							LinearColorMapper, HoverTool)

def go_to_JFK(x, y):
	"""
	Determine if a trip goes to JFK
	"""
	Lng = -73.784214
	Lat = 40.645582
	R = 0.01
	return ((x - Lng)**2 + (y - Lat)**2)**0.5 < R

def go_to_LGA(x, y):
	"""
	Determine if a trip goes to LGA
	"""
	Lng1 = -73.872238
	Lat1 = 40.773140
	Lng2 = -73.864355
	Lat2 = 40.769043
	R1 = 0.002
	R2 = 0.0025
	return (((x - Lng1)**2 + (y - Lat1)**2)**0.5 < R1) or\
			(((x - Lng2)**2 + (y - Lat2)**2)**0.5 < R2)

def get_district(lng, lat, nyc):
	"""
	Return the district name given a pair of coordinates
	"""
	point = Point(lng, lat)
	for feature in nyc:
	    if shape(feature['geometry']).contains(point):
	        return feature['properties']['ntaname']
	return 'Other'

def get_district_boundaries(nyc):
	districts = []
	lng_ls = []
	lat_ls = []
	for d in nyc:
		lng = []
		lat = []
		coords = d['geometry']['coordinates']
		n = len(coords)
		if d['geometry']['type'] == 'Polygon':
			lng += [x[0] for x in coords[0]]
			lat += [x[1] for x in coords[0]]
		else:
			for i, patch in enumerate(coords): 
				lng += [x[0] for x in patch[0]]
				lat += [x[1] for x in patch[0]]
				lng.append(np.nan)
				lat.append(np.nan)
		lng_ls.append(lng)
		lat_ls.append(lat)
		districts.append(d['properties']['ntaname'].strip())
	df = pd.DataFrame({'district':districts, 'lng':lng_ls, 'lat':lat_ls})
	return df

def aggregate_df(df):
	df = df.groupby(by='district', as_index=False).aggregate({'duration':'mean', \
															'distance':'mean', \
															'pickupHour':'count'})
	df.columns = ['district', 'duration', 'distance', 'ct']
	return df

def plot_map(df, fname):
	"""
	Generate the heat map
	"""
	map_options = GMapOptions(lat=40.75, lng=-73.9, map_type='roadmap', zoom=11)
	API_KEY = 'AIzaSyC-4SnwvK3u2CR-zh-4zl7J_msCmDfq_Sg'
	palette = PuRd[7]
	palette.reverse()
	color_mapper = LinearColorMapper(palette = palette)

	plot = GMapPlot(x_range=DataRange1d(),
	                y_range=DataRange1d(),
	                map_options=map_options,
	                api_key=API_KEY,
	                plot_width=1000,
	                plot_height=1000,)

	source = ColumnDataSource(data=dict(
	                                lat = df['lat'].tolist(), 
	                                lng = df['lng'].tolist(),
	                                grp = df['district'],
	                                dur = df['duration'],
	                                durt = df['durt'],
	                                ct = df['ct'],
	    ))

	patches = Patches(xs='lng', ys='lat',
	                fill_color={'field': 'dur', 'transform': color_mapper}, 
	                fill_alpha=0.7, line_color="blue", line_width=1.0)
	plot.add_glyph(source, patches, name='patches')
	hover = HoverTool(names=['patches'],
	                  tooltips=[
	                    ("Neighborhood", "@grp"),
	                    ("Avg Trip Duration", "@durt"),
	                    ("Trip Count", "@ct")
	            ])
	plot.add_tools(PanTool(), WheelZoomTool(), BoxSelectTool(), hover)
	output_file(fname)
	return plot


if __name__ == "__main__":

	# Read csv
	raw_data = pd.read_csv('green_tripdata_2016-01.csv')
	data = raw_data.sample(frac=0.001)
	data = data.iloc[:, [1, 2, 5, 6, 7, 8, 10, 11]]
	data.columns = ['pickupDateTime', 'dropoffDateTime', 'pickupLng', \
					'pickupLat', 'dropoffLng', 'dropoffLat', 'distance', 'fare']
	data['pickupDateTime'] = pd.to_datetime(data['pickupDateTime'])
	data['dropoffDateTime'] = pd.to_datetime(data['dropoffDateTime'])
	data['duration'] = (data['dropoffDateTime'] - data['pickupDateTime']) /\
						np.timedelta64(1, 'D')
	data['fare_rate'] = data['fare'] / data['distance']
	data['pickupHour'] = data['pickupDateTime'].dt.hour

	# Clean data
	invalid = data.ix[(data['pickupLng'] < -74.3) | (data['pickupLng'] > -73.4) |\
						(data['dropoffLng'] < -74.3) | (data['dropoffLng'] > -73.4) |\
						(data['pickupLat'] < 40.5) | (data['pickupLat'] > 41.0) |\
						(data['dropoffLat'] < 40.5) | (data['dropoffLat'] > 41.0) |\
						(data['distance'] <= 0.0) | (data['fare'] <= 0.0) |\
						(data['duration'] > 300.0), :].index
	data = data.drop(invalid) 
	
	# Determine if a trip goes to airports
	data['toJFK'] = data.apply(lambda x: go_to_JFK(x['dropoffLng'], x['dropoffLat']), axis=1)
	data['toLGA'] = data.apply(lambda x: go_to_LGA(x['dropoffLng'], x['dropoffLat']), axis=1)
	df = data.ix[(data['toJFK']  == 1) | (data['toLGA'] == 1), :]
	nyc = fiona.open("Neighborhoods/nyc.shp")
	data['district'] = data.apply(lambda x: get_district(x['pickupLng'], x['pickupLat'], nyc), axis=1)
	data['rush'] = data.apply(lambda x: (x['pickupHour'] >= 8) & (x['pickupHour'] <= 19), axis=1)

	# Aggregate datasets using airports & rushhours and calculate averages
	df_jfk_r = data.ix[(data['toJFK'] == 1) & (data['rush'] == 1), :]
	df_jfk_nr = data.ix[(data['toJFK'] == 1) & (data['rush'] == 0), :]
	df_lga_r = data.ix[(data['toLGA'] == 1) & (data['rush'] == 1), :]
	df_lga_nr = data.ix[(data['toLGA'] == 1) & (data['rush'] == 0), :]
	df_jfk_r = aggregate_df(df_jfk_r)
	df_jfk_nr = aggregate_df(df_jfk_nr)
	df_lga_r = aggregate_df(df_lga_r)
	df_lga_nr = aggregate_df(df_lga_nr)

	# Get district boundaries
	df_map = get_district_boundaries(nyc)
	df_jfk_r = df_jfk_r.merge(df_map, how='left', on='district')
	df_jfk_nr = df_jfk_nr.merge(df_map, how='left', on='district')
	df_lga_r = df_lga_r.merge(df_map, how='left', on='district')
	df_lga_nr = df_lga_nr.merge(df_map, how='left', on='district')

	# Convert duration to mins
	for df in [df_jfk_r, df_jfk_nr, df_lga_r, df_lga_nr]:
		df['durt'] = ["%.1f mins" % x if not np.isnan(x) else "N/A" for x in df.duration.tolist()]	

	"""
	Heat map
	"""

	p = plot_map(df_lga_nr, "lga_nr.html")
	show(p)

