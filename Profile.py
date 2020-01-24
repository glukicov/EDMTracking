# Make profile plots from ROOT histogram
# Gleb Lukicov (11 Jan 2020)  
import os, sys
sys.path.append('CommonUtils/')
import CommonUtils as cu
import RUtils as ru

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
arg_parser.add_argument("--iter", action='store_true', default=False)
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
    np.save("data/dataXY.npy", dataXY)

    # and info
    edm_setting=args.file_path.split("/")[1].split(".")[0]
    Q_cut=args.hist_path.split("/")[0]
    data_type=args.hist_path.split("/")[1]
    time_cut=args.hist_path.split("/")[2]+r" $\mathrm{\mu}$s"
    p_cut=args.hist_path.split("/")[3]+" MeV"
    y_label=args.hist_path.split("/")[4].split("_")[0]
    x_label=args.hist_path.split("/")[4].split("_")[2:]
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
    df_info.to_csv("data/df_info.csv")
    
    print("Data saved to dataXY.npy, re-run with --profile or --hist to make plots")
    sys.exit()

# Plot a 2D histogram
if (args.hist):
    print("Plotting a histogram...")
        
    # load the data 
    dataXY=np.load("dataXY.npy")
    x=dataXY[0]
    y=dataXY[1]

    if(args.beam):
        jg,cb,legendX,legendY =cu.plotHist2D(x, y, n_binsXY=(100,100), cmin=5, prec=2)
        jg.ax_joint.set_ylabel("Vertical beam position [mm]", fontsize=18)
        jg.ax_joint.set_xlabel("Radial beam position [mm]", fontsize=18)
        cu.textL(jg.ax_joint, 1.32, 1.15, "Vertical [mm]:"+"\n"+str(legendX), font_size=15)
        cu.textL(jg.ax_joint, 1.32, 1.00, "Radial [mm]:"+"\n"+str(legendY), font_size=15)
        N=cu.sci_notation(len(x)) # format as a 
        cu.textL(jg.ax_joint, 1.32, 0.00, "N: "+N, font_size=15)
        plt.savefig("beam.png", dpi=300)
    else:
        jg,cb,legendX,legendY =cu.plotHist2D(x, y, n_binsXY=(75,75), cmin=5, prec=3)
        jg.ax_joint.set_ylabel(r"$\theta_y$ [rad]", fontsize=18)
        jg.ax_joint.set_xlabel(r"$t^{mod}_{g-2} \ \mathrm{[\mu}$s]", fontsize=18)
        cu.textL(jg.ax_joint, 1.32, 1.15, r"$t^{mod}_{g-2} \ \mathrm{[\mu}$s]:"+"\n"+str(legendX), font_size=15)
        cu.textL(jg.ax_joint, 1.32, 1.00, r"$\theta_y$ [rad]:"+"\n"+str(legendY), font_size=15)
        N=cu.sci_notation(len(x)) # format as a 
        cu.textL(jg.ax_joint, 1.32, 0.00, "N: "+N, font_size=17)
        plt.savefig("thetavsT.png", dpi=300)


# Profile Plot 
if (args.profile):

    # load the data 
    dataXY=np.load("dataXY.npy")
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
    plt.savefig("profile.png")

    # can save the profile points with errors to a file
    df_binned.to_csv("df_binned.csv")

# iterative fits over many profiles 
if(args.iter):

    #loop over common plots
    
    # fileLabel=("LargeEDM", "noEDM")
    # fileName=("DATA/VLEDM.root", "DATA/noEDM.root")
    fileLabel=[("LargeEDM")]
    fileName=[("DATA/VLEDM.root")]
    
    plotLabel=("Tracks", "Vertices")
    plotName=["TrackFit", "VertexExt"]
    # qLabel=("QT", "NQT")
    # qName=("AllStations", "AllStationsNoTQ")

    qLabel=[("NQT")]
    qName=[("AllStationsNoTQ")]
    
    cutLabel=("0_p_3600", "400_p_2700", "700_p_2400")
    cutName=("t>0/0<p<3600", "t>0/400<p<2700", "t>0/700<p<2400")
    
    for i_file, i_fileName in enumerate(fileName):
        
        # change of name on noEDM
        if (i_file == 1):
            plotName[1]="VertexExtap" #old art code had a typo... 

        for i_cut, i_cutName in enumerate(cutName):
            for i_plot, i_plotName in enumerate(plotName):
                for i_q, i_qName in enumerate(qName):   

                    # form full paths and labels  
                    fullPath=i_qName+"/"+i_plotName+"/"+i_cutName+"/thetay_vs_time_modg2"
                    fullLabel=fileLabel[i_file]+"_"+plotLabel[i_plot]+"_"+qLabel[i_q]+"_"+cutLabel[i_cut]

                    print("Plotting a profile for", fullPath)

                    #extract data from the histogram
                    dataXY, n_binsXY, dBinsXY = ru.hist2np(file_path=i_fileName, hist_path=fullPath)
                    
                    #bin data into a profile 
                    df_data=cu.Profile(dataXY[0], dataXY[1], False, nbins=15, xmin=np.min(dataXY[0]),xmax=np.max(dataXY[0]), mean=True, only_binned=True)
                    x=df_data['bincenters']
                    y=df_data['ymean']
                    x_err=df_data['xerr']
                    y_err=df_data['yerr']
                    y=y*1e3 # rad -> mrad 
                    y_err=y_err*1e3 # rad -> mrad 

                    # extract info
                    edm_setting=fullPath.split("/")[1].split(".")[0]
                    Q_cut=fullPath.split("/")[0]
                    data_type=fullPath.split("/")[1]
                    time_cut=fullPath.split("/")[2]+r" $\mathrm{\mu}$s"
                    p_cut=fullPath.split("/")[3]+" MeV"
                    y_label=fullPath.split("/")[4].split("_")[0]
                    x_label=fullPath.split("/")[4].split("_")[2:]
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

                    #fit a function and get pars 
                    par, pcov = optimize.curve_fit(
    cu.sin_unblinded, x, y, sigma=y_err, p0=[1.0, 1.0, 1.0], absolute_sigma=False, method='lm')
                    par_e = np.sqrt(np.diag(pcov))
                    chi2_n=cu.chi2_ndf(x, y, y_err, cu.sin_unblinded, par)

                    # plot the fit and data 
                    fig, ax = plt.subplots()
                    # data
                    ax.errorbar(x,y,xerr=x_err, yerr=y_err, linewidth=0, elinewidth=2, color="green", marker="o", label="Data")
                    # fit 
                    ax.plot(x, cu.sin_unblinded(x, par[0], par[1], par[2]), color="red", label='Fit')
                  
                    # deal with fitted parameters (to display nicely)
                    parNames=[" A", "b", "c"]
                    units=["mrad", "MHz", "mrad"]
                    prec=2 # set custom precision 
                    
                    #form complex legends 
                    legend1=r"$\frac{\chi^2}{DoF}$= "+"{0:.{prec}f}".format(chi2_n, prec=prec)+"\n"
                    print(legend1)
                    for i, i_name in enumerate(parNames):
                            value=i_name+"= {0:+.{prec}f}".format(par[i], prec=prec)+" \u00B1 {0:.{prec}f}".format( par_e[i], prec=prec)+" "+units[i]
                            print(value)
                            legend1+=value+"\n"

                    legend2=data_type+"\n"+p_cut+"\n N="+cu.sci_notation(N)

                    #decide on the position based on the plot type (TODO as dictionary?)
                    y1=0.15
                    y2=0.85
                    if (data_type=="VertexExt" or data_type=="VertexExtap"):
                        x1=0.20
                        x2=0.65
                    if (data_type=="TrackFit"):
                        x1=0.65
                        x2=0.20

                    #place on the plot and save 
                    cu.textL(ax, x1, y1, legend1, font_size=16, color="red")    
                    cu.textL(ax, x2, y2, legend2, font_size=16)
                    ax.legend(loc='center right')
                    ax.set_ylabel(y_label, fontsize=18)
                    ax.set_xlabel(x_label, fontsize=18)
                    plt.tight_layout() 
                    plt.savefig("fig/"+cutLabel[i_cut]+"/prof_"+fullLabel+"_.png")


print("Done!")
