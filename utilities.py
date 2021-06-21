#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os, glob
import struct
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_samples, silhouette_score

import scipy.cluster.hierarchy as shc
from scipy.ndimage import gaussian_filter
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import itertools
#from scipy.interpolate import griddata

from yellowbrick.cluster import KElbowVisualizer
from pprint import pprint
import cv2 as cv

# ZDA Explorer modules
from trace import Tracer
from image import MovieMaker, SignalProcessor
from freq import FreqAnalyzer


############################# Data load functions ##########################
############################################################################

class DataLoader:

    def load_all_zda(self, data_dir='.'):
        ''' Loads all ZDA data in data_dir into a dictionary of dataframes and metadata '''
        all_data = {}
        n_files_loaded = 0
        for dirName, subdirList, fileList in os.walk(data_dir,topdown=True):
            for file in fileList:
                file = str(dirName + "/" + file)
                if '.zda' == file[-4:]:
                    raw_data, meta, rli = self.read_zda_to_df(file)
                    all_data[file, 'data'] = raw_data
                    all_data[file, 'meta'] = meta
                    all_data[file, 'rli'] = rli
                    n_files_loaded += 1

        print('Number of files loaded:', n_files_loaded)
        return all_data

    def read_zda_to_df(self, zda_file):
        ''' Reads ZDA file to dataframe, and returns
        metadata as a dict. 
        ZDA files are a custom PhotoZ binary format that must be interpreted byte-
        by-byte'''
        file = open(zda_file, 'rb')
        print(zda_file)
        # data type sizes in bytes
        chSize = 1
        shSize = 2
        nSize = 4
        tSize = 8
        fSize = 4

        metadata = {}
        metadata['version'] = (file.read(chSize))
        metadata['slice_number'] = (file.read(shSize))
        metadata['location_number'] = (file.read(shSize))
        metadata['record_number'] = (file.read(shSize))
        metadata['camera_program'] = (file.read(nSize))

        metadata['number_of_trials'] = (file.read(chSize))
        metadata['interval_between_trials'] = (file.read(chSize))
        metadata['acquisition_gain'] = (file.read(shSize))
        metadata['points_per_trace'] = (file.read(nSize))
        metadata['time_RecControl'] = (file.read(tSize))

        metadata['reset_onset'] = struct.unpack('f',(file.read(fSize)))[0]
        metadata['reset_duration'] = struct.unpack('f',(file.read(fSize)))[0]
        metadata['shutter_onset'] = struct.unpack('f',(file.read(fSize)))[0]
        metadata['shutter_duration'] = struct.unpack('f',(file.read(fSize)))[0]

        metadata['stimulation1_onset'] = struct.unpack('f', (file.read(fSize)))[0]
        metadata['stimulation1_duration'] = struct.unpack('f',(file.read(fSize)))[0]
        metadata['stimulation2_onset'] = struct.unpack('f',(file.read(fSize)))[0]
        metadata['stimulation2_duration'] = struct.unpack('f',(file.read(fSize)))[0]

        metadata['acquisition_onset'] = struct.unpack('f',(file.read(fSize)))[0]
        metadata['interval_between_samples'] = struct.unpack('f',(file.read(fSize)))[0]
        metadata['raw_width'] = (file.read(nSize))
        metadata['raw_height'] = (file.read(nSize))

        # Bytes to Python data type
        for k in metadata:
            if k in ['interval_between_samples',] or 'onset' in k or 'duration' in k:
                pass # any additional float processing can go here
            elif k == 'time_RecControl':
                pass # TO DO: timestamp processing
            else:
                metadata[k] = int.from_bytes(metadata[k], "little") # endianness

        num_diodes = metadata['raw_width'] * metadata['raw_height']

        file.seek(1024, 0)
        # RLI 
        rli = {}
        rli['rli_low'] = [int.from_bytes(file.read(shSize), "little") for _ in range(num_diodes)]
        rli['rli_high'] = [int.from_bytes(file.read(shSize), "little") for _ in range(num_diodes)]
        rli['rli_max'] = [int.from_bytes(file.read(shSize), "little") for _ in range(num_diodes)]

        raw_data = np.zeros((metadata['number_of_trials'],
                             metadata['raw_width'],
                             metadata['raw_height'],
                             metadata['points_per_trace'])).astype(int)

        for i in range(metadata['number_of_trials']):
            for jw in range(metadata['raw_width']):
                for jh in range(metadata['raw_height']):
                    for k in range(metadata['points_per_trace']):
                        pt = file.read(shSize)
                        if not pt:
                            print("Ran out of points.",len(raw_data))
                            file.close()
                            return metadata
                        raw_data[i,jw,jh,k] = int.from_bytes(pt, "little")

        #count = 0
        #while pt:
        #    pt = file.read(shSize)
        #    count += 1
        #print('left over:', count)

        file.close()
        return raw_data, metadata, rli


