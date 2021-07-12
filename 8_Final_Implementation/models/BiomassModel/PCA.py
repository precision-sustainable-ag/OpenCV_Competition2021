import numpy as np 
import pandas as pd 
from glob import glob 
import os 
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import h5py



# #RGB_IMAGE_SIZE = (1080,1920)
# #DEPTH_IMAGE_SIZE = (720,1280)
# filename = "/home/pi/BiomassModel/complete_h5_v2.h5"
# with h5py.File(filename, "r") as f:
    # print("Keys: %s" % f.keys())
    
# with open(filename, 'rb') as outfile:
    # print(outfile)
    # df = pd.read_hdf(outfile)
# df = pd.read_hdf("/home/pi/BiomassModel/complete_h5_v2.h5") 

# df = df[df['Biomass'].notna()]
# #df['Biomass'][~df.isnull()]
# df.head()
# #print(df_withBiomass)
# print(df['Biomass'][:])

# stats = df['stats'].to_numpy()
# print(len(stats))

# features = 13
# features_vector = np.zeros((len(stats),features))

# print(features_vector.shape)
                        
# for i in range(len(stats)):
               # features_vector[i][:] = stats[i][0:features]
# df['all_exg'] = features_vector[:,0]
# df['m_exg'] = features_vector[:,1]
# df['sd_exg'] = features_vector[:,2]
# df['all_exr'] = features_vector[:,3]
# df['m_exr'] = features_vector[:,4]
# df['sd_exr'] = features_vector[:,5]
# df['all_ndi'] = features_vector[:,6]
# df['m_ndi'] = features_vector[:,7]
# df['sd_ndi'] = features_vector[:,8]
# df['all_exg_exr'] = features_vector[:,9]
# df['m_exg_exr'] = features_vector[:,10]
# df['sd_exg_exr'] = features_vector[:,11]
# df['all_sumbinary'] = features_vector[:,12]

# df["commonname"].replace({" grass": "grass"}, inplace=True)
# df_grass = df.loc[df['commonname'] == 'grass']
# df_broadl = df.loc[df['commonname'] == 'broadleaf']
# df_clover = df.loc[df['commonname'] == 'clover']

# nested_histogram_data_broadleaf = df_broadl['histograms'].to_numpy()
# NUM_BINS = 50

# # Allocate the memory for the histogram matrix
# histogram_data_broadleaf = np.zeros((len(nested_histogram_data_broadleaf),NUM_BINS))
    
# for i in range(0,len(nested_histogram_data_broadleaf)):
    # histogram_data_broadleaf[i,:] = nested_histogram_data_broadleaf[i][0][0:NUM_BINS]
# nested_histogram_data_clover = df_clover['histograms'].to_numpy()
# NUM_BINS = 50

# # Allocate the memory for the histogram matrix
# histogram_data_clover = np.zeros((len(nested_histogram_data_clover),NUM_BINS))
    
# for i in range(0,len(nested_histogram_data_clover)):
    # histogram_data_clover[i,:] = nested_histogram_data_clover[i][0][0:NUM_BINS]
# nested_histogram_data_grass = df_grass['histograms'].to_numpy()
# NUM_BINS = 50

# # Allocate the memory for the histogram matrix
# histogram_data_grass = np.zeros((len(nested_histogram_data_grass),NUM_BINS))
    
# for i in range(0,len(nested_histogram_data_grass)):
    # histogram_data_grass[i,:] = nested_histogram_data_grass[i][0][0:NUM_BINS]


# features = 20

# biomass_grass =np.zeros(len(histogram_data_grass))
# biomass_clover = np.zeros(len(histogram_data_clover))
# biomass_broadl =np.zeros(len(histogram_data_broadleaf))

# for i in range(len(histogram_data_grass)):
    # biomass_grass[i] = sum(histogram_data_grass[i][0:features])
    # #print(sum(histogram_data_grass[i][0:features]))
# #print(biomass_grass)

# for i in range(len(histogram_data_clover)):
    # biomass_clover[i] = sum(histogram_data_clover[i][0:features])
    # #print(sum(histogram_data_clover[i][0:features]))
# #print(biomass_clover)

# for i in range(len(histogram_data_grass)):
    # biomass_broadl[i] = sum(histogram_data_broadleaf[i][0:features])
    # #print(sum(histogram_data_broadleaf[i][0:features]))
# #print(biomass_broadl)

# df_grass['sum_depth'] = biomass_grass
# df_clover['sum_depth'] = biomass_clover
# df_broadl['sum_depth'] = biomass_broadl
df_broadl = pd.read_csv('broadl_dataframe.csv')
df_grass = pd.read_csv('grass_dataframe.csv')
df_clover = pd.read_csv('clover_dataframe.csv')
import sklearn.ensemble as ske

X_grass = pd.DataFrame(df_grass[['all_exg','m_exg','sd_exg', 'all_exr', 'm_exr', 'sd_exr', 'all_sumbinary','sum_depth']]) #'all_exg_exr','m_exg_exr', 'sd_exg_exr'
X_broadl = pd.DataFrame(df_broadl[['all_exg','m_exg','sd_exg', 'all_exr', 'm_exr', 'sd_exr','all_sumbinary', 'sum_depth']])
X_clover = pd.DataFrame(df_clover[['all_exg','m_exg','sd_exg', 'all_exr', 'm_exr', 'sd_exr', 'all_sumbinary', 'sum_depth']])

y_grass = pd.DataFrame(df_grass['Biomass'])
y_broadl = pd.DataFrame(df_broadl['Biomass'])
y_clover = pd.DataFrame(df_clover['Biomass'])

features = ["All_ExG", "Mean_ExG", "Std_ExG", "All_ExR", "Mean_ExR", "Std_ExR", "Area", "CHM"]#"All_ExG-ExR", "Mean_ExG-ExR", "Std_ExG-ExR", 
print(features)
#print(X_clover)
reg_grass = ske.RandomForestRegressor()
reg_broadl = ske.RandomForestRegressor()
reg_clover = ske.RandomForestRegressor()

reg_grass.fit(X_grass, y_grass)
reg_clover.fit(X_clover, y_clover)
reg_broadl.fit(X_broadl, y_broadl)

from sklearn.model_selection import train_test_split

X_train_grass, X_test_grass, y_train_grass, y_test_grass = train_test_split(X_grass, y_grass, test_size = 0.3, random_state =0)
X_train_clover, X_test_clover, y_train_clover, y_test_clover = train_test_split(X_clover, y_clover, test_size = 0.3, random_state =0)
X_train_broadl, X_test_broadl, y_train_broadl, y_test_broadl = train_test_split(X_broadl, y_broadl, test_size = 0.3, random_state =0)

import joblib

y_predic_grass = reg_grass.predict(X_test_grass)
y_predic_clover = reg_clover.predict(X_test_clover)
y_predic_broadl = reg_broadl.predict(X_test_broadl)

joblib.dump(reg_grass, "./random_forest_grass.joblib")
joblib.dump(reg_clover, "./random_forest_clover.joblib")
joblib.dump(reg_broadl, "./random_forest_broadl.joblib")

from sklearn.metrics import r2_score
r2_grass = r2_score(y_test_grass, y_predic_grass)
r2_clover = r2_score(y_test_clover, y_predic_clover)
r2_broadl = r2_score(y_test_broadl, y_predic_broadl)
print(r2_grass)
print(r2_clover)
print(r2_broadl)

print(X_test_grass)
print(y_test_grass)
