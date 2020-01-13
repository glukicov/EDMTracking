# Make a profile plot from ROOT histogram
# Gleb Lukicov (11 Jan 2020)  

import os, sys
sys.path.append('CommonUtils/')
import CommonUtils as cu
import RUtils as ru

import argparse 
import math
from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--file_path", type=str, default="DATA/VLEDM.root") 
arg_parser.add_argument("--hist_path", type=str, default="AllStationsNoTQ/VertexExt/t>0/0<p<3600/thetay_vs_time_modg2") 
# arg_parser.add_argument("--hist_path", type=str, default="AllStations/VertexExt/t>0/0<p<3600/vertexPosSpread") 
arg_parser.add_argument("--read", action='store_true', default=False) # read and write TH data into numpy file
arg_parser.add_argument("--beam", action='store_true', default=False)
arg_parser.add_argument("--hist", action='store_true', default=False) # Make a 2D plot 
arg_parser.add_argument("--profile", action='store_true', default=False)
arg_parser.add_argument("--fit", action='store_true', default=False)
args=arg_parser.parse_args()

# get data from 2D hist
if(args.read):
    dataXY, n_binsXY, dBinsXY = ru.hist2np(file_path=args.file_path, hist_path=args.hist_path)
    # store that data 
    np.save("dataXY.npy", dataXY)
    print("Data saved to dataXY.npy, re-run with --profile or --hist to make plots")
    sys.exit()

# load 
dataXY=np.load("dataXY.npy")
x=dataXY[0]
y=dataXY[1]

# Plot a 2D histogram
if (args.hist):
    print("Plotting a histogram...")
    if(args.beam):
        jg,cb,legendX,legendY =cu.plotHist2D(dataXY[0], dataXY[1], n_binsXY=(100,100))
        jg.ax_joint.set_ylabel("Vertical beam position [mm]", fontsize=18)
        jg.ax_joint.set_xlabel("Radial beam position [mm]", fontsize=18)
        cu.textL(jg.ax_joint, 1.32, 1.15, "Vertical [mm]:"+"\n"+str(legendX), font_size=15)
        cu.textL(jg.ax_joint, 1.32, 1.00, "Radial [mm]:"+"\n"+str(legendY), font_size=15)
        N=cu.sci_notation(len(dataXY[0])) # format as a 
        cu.textL(jg.ax_joint, 1.32, 0.00, "N: "+N, font_size=15)
        plt.savefig("beam.png", dpi=300)
    else:
        jg,cb,legendX,legendY =cu.plotHist2D(dataXY[0], dataXY[1], n_binsXY=(75,75))
        jg.ax_joint.set_ylabel(r"$\theta_y$ [rad]", fontsize=18)
        jg.ax_joint.set_xlabel(r"$t^{mod}_{g-2} \ \mathrm{[\mu}$s]", fontsize=18)
        cu.textL(jg.ax_joint, 1.32, 1.15, r"$t^{mod}_{g-2} \ \mathrm{[\mu}$s]:"+"\n"+str(legendX), font_size=15)
        cu.textL(jg.ax_joint, 1.32, 1.00, r"$\theta_y$ [rad]:"+"\n"+str(legendY), font_size=15)
        N=cu.sci_notation(len(dataXY[0])) # format as a 
        cu.textL(jg.ax_joint, 1.32, 0.00, "N: "+N, font_size=15)
        plt.savefig("thetavsT.png", dpi=300)


# Profile Plot 
if (args.profile):

    #convert 𝛉 into um
    y=y*1e3 # rad -> mrad 

    print("Plotting a profile...")
    fig,ax=plt.subplots()
    ax, df_binned, df_input =cu.Profile(x, y, ax, nbins=15, xmin=np.min(x),xmax=np.max(x), mean=True)
    ax.set_ylabel(r"$\theta_y$ [mrad]", fontsize=16)
    ax.set_xlabel(r"$t^{mod}_{g-2} \ \mathrm{[\mu}$s]", fontsize=16)
    plt.tight_layout() 
    plt.savefig("profile.png")

    df_binned.to_csv("df.csv")

if (args.fit):
    print("Fitting a profile...") 

    # normal fit to 16 points 

    # Include errors in the fit (investigate the formulae)

    # Gaussian fit to bins from data frame 


print("Done!")
