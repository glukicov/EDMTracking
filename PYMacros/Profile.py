# Make profile plots from ROOT histogram
# Gleb Lukicov (11 Jan 2020)  
import os, sys
sys.path.append('../CommonUtils/')
import CommonUtils as cu
import RUtils as ru
import Functions as fu

import argparse 
import math
from scipy import stats, optimize
import numpy as np
import seaborn as sns
import pandas as pd
from pandas import Series, DataFrame


# MPL in batch mode
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

arg_parser = argparse.ArgumentParser()
# arg_parser.add_argument("--file_path", type=str, default="../ DATA/noEDM.root") 
# arg_parser.add_argument("--hist_path", type=str, default="AllStations/VertexExtap/t>0/0<p<3600/vertexPosSpread") 
# arg_parser.add_argument("--hist_path", type=str, default="AllStations/VertexExtap/t>0/0<p<3600/thetay_vs_time_modg2") 
# arg_parser.add_argument("--hist_path", type=str, default="AllStationsNoTQ/VertexExtap/t>0/0<p<3600/thetay_vs_time_modg2") 
arg_parser.add_argument("--file_path", type=str, default="../DATA/VLEDM.root") 
# arg_parser.add_argument("--hist_path", type=str, default="AllStations/VertexExt/t>0/0<p<3600/thetay_vs_time_modg2") 
# arg_parser.add_argument("--hist_path", type=str, default="AllStationsNoTQ/VertexExt/t>0/0<p<3600/thetay_vs_time_modg2") 
# arg_parser.add_argument("--hist_path", type=str, default="AllStationsNoTQ/VertexExt/t>0/0<p<3600/vertexPosSpread") 
arg_parser.add_argument("--hist_path", type=str, default="AllStationsNoTQ/VertexExt/t>0/1800<p<3600/thetay_vs_time_modg2") 
arg_parser.add_argument("--read", action='store_true', default=False) # read and write TH data into numpy file
arg_parser.add_argument("--beam", action='store_true', default=False)
arg_parser.add_argument("--hist", action='store_true', default=False) # Make a 2D plot 
arg_parser.add_argument("--profile", action='store_true', default=False)
arg_parser.add_argument("--iter", action='store_true', default=False)
arg_parser.add_argument("--gauss", action='store_true', default=False)
arg_parser.add_argument("--bins", action='store_true', default=False)
args=arg_parser.parse_args()

#TODO impliment check of at least 1 arg 
# args_v = vars(arg_parser.parse_args())
# if all(not args_v):
#     arg_parser.error('No arguments provided.')

#set some global variables  
info=[]
names=[]

# get data from 2D hist
if(args.read):
    dataXY, n_binsXY, dBinsXY = ru.hist2np(file_path=args.file_path, hist_path=args.hist_path)
    # store that data
    np.save("../DATA/misc/dataXY.npy", dataXY)

    # and info
    edm_setting=args.file_path.split("/")[1].split(".")[0]
    print(args.hist_path)
    Q_cut=args.hist_path.split("/")[0]
    data_type=args.hist_path.split("/")[1]
    time_cut=args.hist_path.split("/")[2]+r" $\mathrm{\mu}$s"
    p_cut=args.hist_path.split("/")[3]+" MeV"
    y_label=args.hist_path.split("/")[4].split("_")[0]
    x_label=args.hist_path.split("/")[4].split("_")[2:]
    print(x_label)
    N=len(dataXY[0])
    #some specific string transforms
    if (edm_setting == "VLEDM"):
        edm_setting=r"$d_{\mu} = 5.4\times10^{-18} \ e\cdot{\mathrm{cm}}$"
    if (edm_setting == "noEDM"):
        edm_setting=r"$d_{\mu} = 0 \ e\cdot{\mathrm{cm}}$"
    if(y_label=="thetay"):
        y_label=r"$\langle\theta_y\rangle$ [mrad]"
    if(x_label[0]=="time" and x_label[1]=="modg2"):
        x_label=r"$t^{mod}_{g-2} \ \mathrm{[\mu}$s]"
    # two lists into dict 
    info=[edm_setting, data_type, Q_cut, time_cut, p_cut, x_label, y_label, N]
    names=["edm_setting", "data_type", "Q_cut", "time_cut", "p_cut", "x_label", "y_label", "N"]
    info_dict = dict(zip(names, info))
    #now pass along the information to the fitter
    df_info = pd.DataFrame(info_dict, index=[0]) 
    df_info.to_csv("../DATA/misc/df_info.csv")
    
    print("Data saved to dataXY.npy, re-run with --profile or --hist to make plots")
    sys.exit()

# Plot a 2D histogram
if (args.hist):
    print("Plotting a histogram...")
        
    # load the data 
    dataXY=np.load("../DATA/misc/dataXY.npy")
    x=dataXY[0]
    y=dataXY[1]

    if(args.beam):
        jg,cb,legendX,legendY =cu.plotHist2D(x, y, n_binsXY=(100,100), cmin=5, prec=2)
        jg.ax_joint.set_ylabel("Vertical beam position [mm]", fontsize=18)
        jg.ax_joint.set_xlabel("Radial beam position [mm]", fontsize=18)
        cu.textL(jg.ax_joint, 1.32, 1.15, "Radial [mm]:"+"\n"+str(legendX), font_size=15)
        cu.textL(jg.ax_joint, 1.32, 1.00, "Vertical [mm]:"+"\n"+str(legendY), font_size=15)
        N=cu.sci_notation(len(x)) # format as a 
        cu.textL(jg.ax_joint, 1.32, 0.00, "N: "+N, font_size=15)
        plt.savefig("../fig/beam.png", dpi=300)
    else:
        jg,cb,legendX,legendY =cu.plotHist2D(x, y, n_binsXY=(75,75), cmin=5, prec=3)
        jg.ax_joint.set_ylabel(r"$\theta_y$ [rad]", fontsize=18)
        jg.ax_joint.set_xlabel(r"$t^{mod}_{g-2} \ \mathrm{[\mu}$s]", fontsize=18)
        cu.textL(jg.ax_joint, 1.32, 1.15, r"$t^{mod}_{g-2} \ \mathrm{[\mu}$s]:"+"\n"+str(legendX), font_size=15)
        cu.textL(jg.ax_joint, 1.32, 1.00, r"$\theta_y$ [rad]:"+"\n"+str(legendY), font_size=15)
        N=cu.sci_notation(len(x)) # format as a 
        cu.textL(jg.ax_joint, 1.32, 0.00, "N: "+N, font_size=17)
        plt.savefig("../fig/thetavsT.png", dpi=300)


# Profile Plot 
if (args.profile):

    # load the data 
    dataXY=np.load("../DATA/misc/dataXY.npy")
    x=dataXY[0]
    y=dataXY[1]

    #convert 𝛉 into um
    y=y*1e3 # rad -> mrad 

    print("Plotting a profile...")
    fig,ax=plt.subplots()
    ax, df_binned, df_input =cu.Profile(x, y, ax, nbins=15, xmin=np.min(x),xmax=np.max(x), mean=True)
    ax.set_ylabel(r"$\langle\theta_y\rangle$ [mrad]", fontsize=16)
    ax.set_xlabel(r"$t^{mod}_{g-2} \ \mathrm{[\mu}$s]", fontsize=16)
    N=cu.sci_notation(len(x)) # format as a 
    cu.textL(ax, 0.88, 0.9, "N: "+N, font_size=14)
    plt.tight_layout() 
    plt.savefig("../fig/profile.png")

    # can save the profile points with errors to a file 
    df_binned.to_csv("../DATA/misc/df_binned.csv")

# iterative fits over many profiles 
if(args.iter and not args.bins):
    print("Plotting iterative profiles...")
    # df = pd.DataFrame(columns=['cut', 'A_mu','A_edm','c','w','chi2'])
    fu.iter_plots(n_prof_bins=15, gauss=args.gauss, df=df)

    # print(df)
    # df.to_csv("../DATA/misc/df_fits.csv")

if(args.iter and args.bins):
    print("Plotting iterative profiles with bins...")
    bins=np.arange(5, 40, 1)
    A=[]
    A_e=[]
    Chi2=[]
    for i_bin in bins:
        print("Bin:", i_bin)
        a, a_e, chi2 = fu.iter_plots(n_prof_bins=i_bin,  extraLabel=str(i_bin), outdir="bins", best=True, vertex=True)
        A.append(a)
        A_e.append(a_e)
        Chi2.append(chi2)

    #dict -> DF -> csv for plotting 
    d = {'A':A,'A_e':A_e,"Chi2":Chi2}
    df = pd.DataFrame(d)
    df.to_csv("../DATA/misc/df_bins.csv")


print("Done!")