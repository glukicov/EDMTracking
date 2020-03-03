# Author: Gleb Lukicov (21 Feb 2020)
# A front-end module to run over fitWithBlinders_skim.py iteratively  

import sys, re, subprocess
import argparse
import pandas as pd
import numpy as np
np.set_printoptions(precision=3) # 3 sig.fig 
sys.path.append('../CommonUtils/') # https://github.com/glukicov/EDMTracking/blob/master/CommonUtils/CommonUtils.py
import CommonUtils as cu
from scipy import stats, optimize, fftpack
import matplotlib as mpl
mpl.use('Agg') # MPL in batch mode
font_size=15
import matplotlib.pyplot as plt

#Input fitting parameters 
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--all", action='store_true', default=False) # just run all 4 DS 
arg_parser.add_argument("--start", action='store_true', default=False) 
arg_parser.add_argument("--end", action='store_true', default=False) 
args=arg_parser.parse_args()

### Constants 
DS_path = ("../DATA/HDF/MMA/60h.h5", "../DATA/HDF/MMA/9D.h5", "../DATA/HDF/MMA/HK.h5", "../DATA/HDF/MMA/EG.h5")
start_times = np.linspace(30, 100, 70)

def main():
    '''
    As a quick solution use sub process 
    TODO: import and run module properly
    '''

    if(args.all): all(DS_path)

    if(args.start): time_scan(DS_path, start_times)


def all(DS_path):
    for path in DS_path:
        subprocess.call(["python3", "fitWithBlinders_skim.py", "--hdf", path])

def time_scan(DS_path, times):
    if (args.start==True): key = "--min"
    if (args.end==True): key = "--max"
    for path in DS_path:
        for start_time in times:
            subprocess.call(["python3", "fitWithBlinders_skim.py", "--hdf", path, "--cbo", "--scan", key, str(start_time)])

if __name__ == "__main__":
    main()