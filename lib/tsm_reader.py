import os
import struct
import numpy as np


# RedshirtImaging website says it works with ImageJ, which supports:
#   https://imagej.nih.gov/ij/docs/guide/146-7.html#sub:Native-Formats
class TSM_Reader():

    def __init__(self):

        self.width = None
        self.height = None
        self.num_pts = None
        self.int_pts = None
        self.images = None
        self.fp_arr = None
        self.dark_frame = None
        self.metadata = {}
        
    def get_dim(self):
        return [self.num_pts, self.width, self.height]

    def get_int_pts(self):
        return self.int_pts

    def load_tsm(self, filename):
        print(filename, "to be treated as TSM file to open")

        file = open(filename, 'rb')
        header = str(file.read(2880))

        # header parsing
        header = [x.strip() for x in header.split(" ") if x != "=" and len(x) > 0]
        for i in range(len(header)):
            if header[i] == "NAXIS1":
                self.width = int(header[i+1])
            if header[i] == "NAXIS2":
                self.height = int(header[i+1])
            if header[i] == "NAXIS3":
                self.num_pts = int(header[i+1])
            if header[i] == "EXPOSURE=":
                self.int_pts = float(header[i+1]) * 1000 # ms

        self.metadata['number_of_trials'] = 1

        print("Reading file as", self.num_pts, "images of size", self.width, "x", self.height)

        self.images = np.fromfile(file,
                                dtype=np.int16,
                                count=self.num_pts * self.width * self.height).reshape(self.num_pts, self.height, self.width)
        self.images = self.images.reshape((1,)+self.images.shape)
        self.dark_frame = np.fromfile(file,
                                dtype=np.int16,
                                count=self.width * self.height).reshape(self.height, self.width)
        file.close()

        tbn_filename = filename.split(".tsm")[0] + ".tbn"
        self.load_tbn(tbn_filename, self.num_pts)

    # read NI data from .tbn file
    def load_tbn(self, filename, num_pts, trial=0):

        file = open(filename, 'rb')

        num_channels = int.from_bytes(file.read(2), byteorder='little', signed=True)
        if num_channels < 0:
            print("TBN file designates origin as NI for this data.")
            num_channels *= -1
        BNC_ratio = int.from_bytes(file.read(2), byteorder='little')
        num_fp_pts = BNC_ratio * num_pts
        print("Found", num_channels, "channels in BNC ratio:", BNC_ratio)

        self.fp_arr = np.transpose(np.fromfile(file, dtype=np.float64, count=num_channels * num_fp_pts).reshape(num_channels, num_fp_pts))
        
        file.close()
        
        
    def get_fp_arr(self):
        return self.fp_arr
        
    def get_dark_frame(self):
        return self.dark_frame
    
    def get_images(self):
        return self.images
