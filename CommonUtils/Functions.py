# Put some large functions here 
# Gleb Lukicov (11 Jan 2020) 

import RUtils as ru
import CommonUtils as cu
import numpy as np
import pandas as pd
from scipy import stats, optimize
# MPL in batch mode
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def iter_plots(n_prof_bins=15, extraLabel="", outdir="profFits", gauss=False, best=False, vertex=False, df=None):
    #loop over common plots
    
    # fileLabel=("LargeEDM", "noEDM")
    # fileName=("DATA/VLEDM.root", "DATA/noEDM.root")
    fileLabel=[("LargeEDM")]
    fileName=[("../DATA/VLEDM.root")]
    
    plotLabel=("Tracks", "Vertices")
    plotName=["TrackFit", "VertexExt"]

    if (best):
        plotLabel=[("Vertices")]
        plotName=[("VertexExt")]

    # qLabel=("QT", "NQT")
    # qName=("AllStations", "AllStationsNoTQ")\

    qLabel=[("NQT")]
    qName=[("AllStationsNoTQ")]
    
    cutLabel=("0_p_3600", "400_p_2700", "700_p_2400", "1500_p_3600", "1600_p_3600", "1700_p_3600", "1800_p_3600")
    cutName=("t>0/0<p<3600", "t>0/400<p<2700", "t>0/700<p<2400", "t>0/1500<p<3600", "t>0/1600<p<3600", "t>0/1700<p<3600", "t>0/1800<p<3600")

    if (best):
        cutLabel=[("700_p_2400")]
        cutName=[("t>0/700<p<2400")]
    
    for i_file, i_fileName in enumerate(fileName):
        
        # change of name on noEDM
        if (i_file == 1):
            plotName[1]="VertexExtap" #old art code had a typo... 

        for i_cut, i_cutName in enumerate(cutName):
            for i_plot, i_plotName in enumerate(plotName):
                for i_q, i_qName in enumerate(qName):   

                    # form full paths and labels  
                    fullPath=i_qName+"/"+i_plotName+"/"+i_cutName+"/thetay_vs_time_modg2"
                    fullLabel=extraLabel+fileLabel[i_file]+"_"+plotLabel[i_plot]+"_"+qLabel[i_q]+"_"+cutLabel[i_cut]

                    print("Plotting a profile for", fullPath)

                    #extract data from the histogram
                    dataXY, n_binsXY, dBinsXY = ru.hist2np(file_path=i_fileName, hist_path=fullPath)
                    
                    #bin data into a profile 
                    if(gauss):
                        df_data=cu.Profile(dataXY[0], dataXY[1], False, nbins=n_prof_bins, xmin=np.min(dataXY[0]),xmax=np.max(dataXY[0]), full_y=True, only_binned=True)
                        y=df_data['y']
                    else:
                        df_data=cu.Profile(dataXY[0], dataXY[1], False, nbins=n_prof_bins, xmin=np.min(dataXY[0]),xmax=np.max(dataXY[0]), mean=True, only_binned=True)
                        y=df_data['ymean']
                        
                    x=df_data['bincenters']
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

                    # if extracting y and delta(y) from a Gaussian fit 
                    if(gauss):
                        means=[]
                        means_errors=[]
                        for i_point in range(0, len(df_data)):
                            
                            # fit for a range of data 
                            n_bins=25
                            y_min, y_max=-25, +25

                            # all data y_hist, range is y 
                            y_hist=y[i_point]
                            y_select  = y_hist[np.logical_and(y_hist >= y_min, y_hist <= y_max)]

                            #bin the data in range 
                            hist, bin_edges = np.histogram(y_select, bins=n_bins, density=False)
                            bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
                            bin_width=bin_edges[1]-bin_edges[0] 
                            #find the right number of bins for. all data
                            n_bins_hist=int( (np.max(y_hist)-np.min(y_hist))/bin_width ) 
                            y_err=np.sqrt(hist) # sqrt(N) per bin

                            # fit in range 
                            p0=[1, 1, 1]
                            par, pcov = optimize.curve_fit(cu.gauss, bin_centres, hist, p0=p0, sigma=y_err, absolute_sigma=False, method='trf')
                            par_e = np.sqrt(np.diag(pcov))
                            chi2ndf= cu.chi2_ndf(bin_centres, hist, y_err, cu.gauss, par)

                            #append the fit parameters
                            means.append(par[1])
                            means_errors.append(par_e[1])

                            #plot and stats + legend 
                            units="mrad"
                            legend_fit=cu.legend4_fit(chi2ndf[0], par[1], par_e[1], par[2], par_e[2], units, prec=2)
                            ax, legend = cu.plotHist(y_hist, n_bins=n_bins_hist, units=units, prec=2)
                            ax.plot(bin_centres, cu.gauss(bin_centres, *par), color="red", linewidth=2, label='Fit')
                            cu.textL(ax, 0.8, 0.78, r"$\theta_y$:"+"\n"+str(legend), font_size=15)
                            cu.textL(ax, 0.2, 0.78, "Fit:"+"\n"+str(legend_fit), font_size=15, color="red")
                            ax.set_xlabel(r"$\theta_y$ [mrad]", fontsize=18)
                            ax.set_ylabel(r"$\theta_y$ / "+str(round(bin_width))+" mrad", fontsize=18)
                            plt.tight_layout()
                            plt.savefig("../fig/Gauss/Gauss_"+fullLabel+"_"+str(i_point)+".png", dpi=300)
                            plt.clf()

                        #done looping over y bins
                        #reassign data "pointer names"
                        y=means
                        y_err=means_errors


                    #fit a function and get pars 
                    par, pcov = optimize.curve_fit(cu.thetaY_unblinded, x, y, sigma=y_err, p0=[0.0, 0.18, -0.06, 1.4], absolute_sigma=False, method='lm')
                    par_e = np.sqrt(np.diag(pcov))
                    chi2_n=cu.chi2_ndf(x, y, y_err, cu.thetaY_unblinded, par)

                    # plot the fit and data 
                    fig, ax = plt.subplots()
                    # data
                    ax.errorbar(x,y,xerr=x_err, yerr=y_err, linewidth=0, elinewidth=2, color="green", marker="o", label="Sim.")
                    # fit 
                    ax.plot(x, cu.thetaY_unblinded(x, par[0], par[1], par[2], par[3]), color="red", label='Fit')
                  
                    # deal with fitted parameters (to display nicely)
                    parNames=[r"$ A_{\mu}$", r"$ A_{\rm{EDM}}$", "c", r"$\omega$"]
                    units=["mrad", "mrad", "mrad", "MhZ"]
                    prec=2 # set custom precision 
                    
                    #form complex legends 
                    legend1_chi2=cu.legend1_fit(chi2_n[0])
                    legned1_par=""
                    legned1_par=cu.legend_par(legned1_par, parNames, par, par_e, units)
                    legend1=legend1_chi2+"\n"+legned1_par
                    print(legend1)
                    legend2=data_type+"\n"+p_cut+"\n N="+cu.sci_notation(N)

                    #place on the plot and save 
                    y1,y2,x1,x2=0.15,0.85,0.25,0.70
                    cu.textL(ax, x1, y1, legend1, font_size=16, color="red")    
                    cu.textL(ax, x2, y2, legend2, font_size=16)
                    ax.legend(loc='center right', fontsize=16)
                    ax.set_ylabel(y_label, fontsize=18)
                    ax.set_xlabel(x_label, fontsize=18)
                    plt.tight_layout() 
                    plt.savefig("../fig/"+outdir+"/"+fullLabel+".png")
                    plt.clf()

                    # if (df):
                    #     df.loc[-1] = [p_cut, par[0], par[1], par[2], par[3], chi2_n[0]]

    # return p_cut, par[0], par[1], par[2], par[3], chi2_n[0]
