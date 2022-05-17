import os
import struct
import numpy as np

from pyPhoto21.database.file import File
from pyPhoto21.database.metadata import Metadata


# RedshirtImaging website says it works with ImageJ, which supports:
#   https://imagej.nih.gov/ij/docs/guide/146-7.html#sub:Native-Formats
class TSM_Reader(File):

    def __init__(self):
        super().__init__(Metadata())
        
        self.width = None
        self.height = None
        self.num_pts = None
        self.images = None
        self.fp_arr = None
        self.dark_frame = None
        
    def get_dim(self):
        return [self.num_pts, self.width, self.height]

    def load_tsm(self, filename):
        print(filename, "to be treated as TSM file to open")

        file = open(filename, 'rb')
        header = str(file.read(2880))

        # header parsing
        header = [x.strip() for x in header.split(" ") if x != "=" and len(x) > 0]
        for i in range(len(header)):
            if header[i] == "NAXIS1":
                width = int(header[i+1])
            if header[i] == "NAXIS2":
                height = int(header[i+1])
            if header[i] == "NAXIS3":
                num_pts = int(header[i+1])

        print("Reading file as", num_pts, "images of size", width, "x", height)

        self.images = np.fromfile(file, dtype=np.int16, count=num_pts * width * height).reshape(num_pts, height, width)
        self.dark_frame = np.fromfile(file, dtype=np.int16, count=width * height).reshape(height, width)
        file.close()

        tbn_filename = filename.split(".tsm")[0] + ".tbn"
        self.load_tbn(tbn_filename, db, num_pts)

    # read NI data from .tbn file
    def load_tbn(self, filename, db, num_pts, trial=0):

        if db.file_exists_in_own_path(filename):
            print("Found file to load FP data from:", filename)
        else:
            print("Could not find a matching .tbn file:", filename)
            return

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
