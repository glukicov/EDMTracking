# Author: Gleb Lukicov (21 Feb 2020)
# Perform a 5-parameter blinded fit 
# on skimmed data in HDF5 format (from skimTrees.py module)

#Blinding lib imported in CommonUtils:
import sys, re
sys.path.append('../CommonUtils/') # https://github.com/glukicov/EDMTracking/blob/master/CommonUtils/CommonUtils.py
import CommonUtils as cu
import argparse
from scipy import stats, optimize
import numpy as np
np.set_printoptions(precision=3) # 3 sig.fig 
import matplotlib as mpl
mpl.use('Agg') # MPL in batch mode
import matplotlib.pyplot as plt
import pandas as pd

stations=(12, 18)

#Input fitting parameters 
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--p0", nargs='+', type=float, default=[85263, 64.0, 0.1, 1.0, 2.0]) #fit parameters (initial guess)
arg_parser.add_argument("--max", type=float, default=400) #max fit time 
arg_parser.add_argument("--hdf", type=str, default="../DATA/HDF/MMA/60h.h5") #input data 
arg_parser.add_argument("--key", type=str, default="QualityTracks") # or QualityVertices
arg_parser.add_argument("--label", type=str, default="60h") # or QualityVertices
args=arg_parser.parse_args()
    
def main():

    #open the hdf file and fit! 
    fit()


def fit():
    print("Opening", args.key, "in", args.hdf, "...")
    data = pd.read_hdf(args.hdf, args.key)
    print("Found", data.shape[0], "entries")
    
    #define station cuts to loop over 
    s12_cut = (data['station'] == stations[0])
    s18_cut = (data['station'] == stations[1])
    station_cut = (s12_cut, s18_cut)

    for i_cut, station in enumerate(stations):
        data_station=data[station_cut[i_cut]]
        # resolve into t variable for ease for a station
        t = data_station['trackT0'] # already expected in us (or can just *1e-3 here if not)
        N=data_station.shape[0]
        print("Entries: ", N, " in", station)
        
        #define modulation and limits (more can be made as arguments) 
        bin_w = 150*1e-3 # 150 ns 
        min_x = 30 # us
        max_x = args.max # us 
        t_mod=100 # us; fold plot every N us 

        print("digitising data (binning)...")
        # just call x,y = frequencies, bin_centres for plotting and fitting 
        x, y = cu.get_freq_bin_c_from_data( t, bin_w, (min_x, max_x) )
        y_err = np.sqrt(y) # sigma =sqrt(N)
        print("digitised data done!")

        print("Fitting...")
        # Levenberg-Marquardt algorithm as implemented in MINPACK
        par, pcov = optimize.curve_fit(f=cu.blinded_wiggle_function, xdata=x, ydata=y, sigma=y_err, p0=args.p0, absolute_sigma=False, method='lm')
        par_e = np.sqrt(np.diag(pcov))
        print("Pars  :", np.array(par))
        print("Pars e:",np.array(par_e))
        chi2_ndf, chi2, ndf=cu.chi2_ndf(x, y, y_err, cu.blinded_wiggle_function, par)
        print("Fit 𝝌2/DoF="+str(round(chi2_ndf,2)) )

        print("Plotting fit and data!")
        #use pre-define module wiggle function from CU
        # plt.tight_layout()
        
        #make more automated things for "plot prettiness"
        data_type = re.findall('[A-Z][^A-Z]*', args.key) # should break into "Quality" and "Tracks"/"Vertices"

        fig,ax = cu.modulo_wiggle_5par_fit_plot(x, y, t_mod, max_x, min_x, N, par, par_e, chi2_ndf, bin_w,
                                                key=data_type[0]+" "+data_type[1], legend_data="Run-1: "+args.label+" dataset S"+str(station) )
        plt.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, 1.1) )
        plt.savefig("../fig/wiggle_blind_S"+str(station)+"_"+args.label+".png", dpi=300)
        print("Done for", station)

if __name__ == "__main__":
    main()