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
import pandas as pd
from pandas import Series, DataFrame

# MPL in batch mode
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

arg_parser = argparse.ArgumentParser()
# arg_parser.add_argument("--file_path", type=str, default="DATA/noEDM.root") 
# arg_parser.add_argument("--hist_path", type=str, default="AllStations/VertexExtap/t>0/0<p<3600/thetay_vs_time_modg2") 
# arg_parser.add_argument("--hist_path", type=str, default="AllStationsNoTQ/VertexExtap/t>0/0<p<3600/thetay_vs_time_modg2") 

arg_parser.add_argument("--file_path", type=str, default="DATA/VLEDM.root") 
# arg_parser.add_argument("--hist_path", type=str, default="AllStations/VertexExt/t>0/0<p<3600/thetay_vs_time_modg2") 
arg_parser.add_argument("--hist_path", type=str, default="AllStationsNoTQ/VertexExt/t>0/0<p<3600/thetay_vs_time_modg2") 

# arg_parser.add_argument("--hist_path", type=str, default="AllStations/VertexExt/t>0/0<p<3600/vertexPosSpread") 
arg_parser.add_argument("--read", action='store_true', default=False) # read and write TH data into numpy file
arg_parser.add_argument("--beam", action='store_true', default=False)
arg_parser.add_argument("--hist", action='store_true', default=False) # Make a 2D plot 
arg_parser.add_argument("--profile", action='store_true', default=False)
arg_parser.add_argument("--fit", action='store_true', default=False)
arg_parser.add_argument("--iter", action='store_true', default=False)
args=arg_parser.parse_args()

#TODO impliment check of at least 1 arg 
# args_v = vars(arg_parser.parse_args())
# if all(not args_v):
#     arg_parser.error('No arguments provided.')

# get data from 2D hist
if(args.read):
    dataXY, n_binsXY, dBinsXY = ru.hist2np(file_path=args.file_path, hist_path=args.hist_path)
    # store that data 
    np.save("dataXY.npy", dataXY)
    print("Data saved to dataXY.npy, re-run with --profile or --hist to make plots")
    sys.exit()

# load the data 
dataXY=np.load("dataXY.npy")
x=dataXY[0]
y=dataXY[1]

# Plot a 2D histogram
if (args.hist):
    print("Plotting a histogram...")
    if(args.beam):
        jg,cb,legendX,legendY =cu.plotHist2D(dataXY[0], dataXY[1], n_binsXY=(100,100), cmin=5, prec=2)
        jg.ax_joint.set_ylabel("Vertical beam position [mm]", fontsize=18)
        jg.ax_joint.set_xlabel("Radial beam position [mm]", fontsize=18)
        cu.textL(jg.ax_joint, 1.32, 1.15, "Vertical [mm]:"+"\n"+str(legendX), font_size=15)
        cu.textL(jg.ax_joint, 1.32, 1.00, "Radial [mm]:"+"\n"+str(legendY), font_size=15)
        N=cu.sci_notation(len(dataXY[0])) # format as a 
        cu.textL(jg.ax_joint, 1.32, 0.00, "N: "+N, font_size=15)
        plt.savefig("beam.png", dpi=300)
    else:
        jg,cb,legendX,legendY =cu.plotHist2D(dataXY[0], dataXY[1], n_binsXY=(75,75), cmin=5, prec=3)
        jg.ax_joint.set_ylabel(r"$\theta_y$ [rad]", fontsize=18)
        jg.ax_joint.set_xlabel(r"$t^{mod}_{g-2} \ \mathrm{[\mu}$s]", fontsize=18)
        cu.textL(jg.ax_joint, 1.32, 1.15, r"$t^{mod}_{g-2} \ \mathrm{[\mu}$s]:"+"\n"+str(legendX), font_size=15)
        cu.textL(jg.ax_joint, 1.32, 1.00, r"$\theta_y$ [rad]:"+"\n"+str(legendY), font_size=15)
        N=cu.sci_notation(len(dataXY[0])) # format as a 
        cu.textL(jg.ax_joint, 1.32, 0.00, "N: "+N, font_size=17)
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
    N=cu.sci_notation(len(x)) # format as a 
    cu.textL(ax, 0.88, 0.9, "N: "+N, font_size=14)
    plt.tight_layout() 
    plt.savefig("profile.png")

    # can save the profile points with errors to a file
    # df_binned.to_csv("df.csv")


if(args.iter):

# AllStationsNoTQ/VertexExtap/t>0/0<p<3600/thetay_vs_time_modg2

    #loop over common plots
    fileLabel=("LargeEDM", "noEDM")
    fileName=("DATA/VLEDM.root", "DATA/noEDM.root")
    
    plotLabel=("Tracks", "Vertices")
    plotName=["TrackFit", "VertexExt"]
    
    qLabel=("QT", "NQT")
    qName=("AllStations", "AllStationsNoTQ")
    
    cutLabel=("0_p_3600", "400_p_2700", "700_p_2400")
    cutName=("t>0/0<p<3600", "t>0/400<p<2700", "t>0/700<p<2400")
    
    for i_file, i_fileName in enumerate(fileName):
        
        # change of name on noEDM
        if (i_file == 1):
            plotName[1]="VertexExtap"

        for i_cut, i_cutName in enumerate(cutName):
            for i_plot, i_plotName in enumerate(plotName):
                for i_q, i_qName in enumerate(qName):   

                    fullPath=i_qName+"/"+i_plotName+"/"+i_cutName+"/thetay_vs_time_modg2"
                    fullLabel=fileLabel[i_file]+"_"+plotLabel[i_plot]+"_"+qLabel[i_q]+"_"+cutLabel[i_cut]

                    dataXY, n_binsXY, dBinsXY = ru.hist2np(file_path=i_fileName, hist_path=fullPath)

                    #get time (x), and 𝛉 (y) 
                    x=dataXY[0]
                    y=dataXY[1] 
                    #convert 𝛉 into um
                    y=np.array(y)*1e3 # rad -> mrad 

                    print("Plotting a profile for", fullPath)
                    fig,ax=plt.subplots()
                    ax, df_binned, df_input =cu.Profile(x, y, ax, nbins=15, xmin=np.min(x),xmax=np.max(x), mean=True)
                    ax.set_ylabel(r"$\langle\theta_y\rangle$ [mrad]", fontsize=16)
                    ax.set_xlabel(r"$t^{mod}_{g-2} \ \mathrm{[\mu}$s]", fontsize=16)
                    N=cu.sci_notation(len(x)) # format as a 
                    cu.textL(ax, 0.88, 0.9, "N: "+N, font_size=14)
                    cu.textL(ax, 0.3, 0.9, fullLabel, font_size=14)
                    plt.tight_layout() 
                    plt.savefig(cutLabel[i_cut]+"/prof_"+fullLabel+"_.png")


if (args.fit):
    print("Fitting a profile...") 

    # normal fit to 16 points 

    # Include errors in the fit (investigate the formulae)

    # Gaussian fit to bins from data frame 


print("Done!")
