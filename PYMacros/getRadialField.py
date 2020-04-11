######
# Gleb (20 Sep 2019)
# Plot QHV db data vs <Y> offline data to estimate B_r 
#######
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
from matplotlib.ticker import MaxNLocator
import argparse 
import collections, itertools
import sys, os
import subprocess
import datetime
import pickle
import scipy.interpolate
import scipy.ndimage
from scipy import stats
import itertools # smart lines in plotting 

#Pass some commands 
arg_parser = argparse.ArgumentParser(description='Input data files')
arg_parser.add_argument('--dbFile', type=str, required=True, dest='dbFile')  # DB data (time, qhv)
arg_parser.add_argument('--gpsFile', type=str, required=True, dest='gpsFile') # Offline data (time YS12, YS18)
#Individual plots if passed a bin number, otherwise loop over a define range 
arg_parser.add_argument('--bins', type=int, default=np.arange(5, 40, 1), dest='bins', nargs='+') 
arg_parser.add_argument('--mode', type=int, default=0, help='"0"= 1/qhv "1"=qhv')
args = arg_parser.parse_args()
dbFile = args.dbFile
gpsFile = args.gpsFile
bins = args.bins
mode = args.mode

### Constants 
db_type = [('times',np.float64),('qhv',np.float64)]
gps_type = [('times',np.float64),('s12',np.float64), ('s18',np.float64)]
stations=("S12", "S18")
# plot constants
font=14
plt.rc('xtick',labelsize=font)
plt.rc('ytick',labelsize=font)

#Physics constants
B_0 = 1.45 # T
R_0 = 7.11e3 # mm 
n = 0.005715 # TODO calculate n explicitly (TDR, how James did it) for a range of voltages in question 

# Data types 
db_collection_shape = ['time', 'qhv']
db_collection_reshaped_shape = ['time', 'qhv', 'eqhv']
gps_collection_shape= ['time', 's12', 's18']
gps_collection_reshaped_shape= ['time', 's12', 'es12', 's18', 'es18']
db_collection = collections.namedtuple('db_collection', db_collection_shape)
db_collection_clean = collections.namedtuple('db_collection_clean', db_collection_shape)
db_collection_reshaped = collections.namedtuple('db_collection_reshaped', db_collection_reshaped_shape)
gps_collection = collections.namedtuple('gps_collection', gps_collection_shape)
gps_collection_reshaped = collections.namedtuple('gps_collection', gps_collection_reshaped_shape)

# storage 
chi2_array = [] # cors_array[i_bin][i_station]
slope_array = [] 
slopeE_array = []

def main():

    print("Starting plots on:", datetime.datetime.now())

    #Get data 
    print("Getting data from files...")
    db_data = getDBData(dbFile)
    gps_data = getGPSData(gpsFile)

    #Clean DB data (same start/end points)
    print("Cleaning DB data...")
    db_data_clean = cleanDB(db_data, gps_data)

    for i_bin in bins:

        #Rebin data 
        print("Re-binning data into",i_bin,"bins with error calculation for GPS data")
        db_data_reshaped, rescale_db = dbReshape(db_data_clean, nbins=i_bin)
        gps_data_reshaped, rescale_gps = gpsReshape(gps_data, nbins=i_bin)

        #Plot data 
        print("Plotting data..")
        chi2, slope, slopeE = plotData(db_data_reshaped, gps_data_reshaped, rescale_db, rescale_gps, i_bin, mode)
        chi2_array.append(chi2)
        slope_array.append(slope)
        slopeE_array.append(slopeE)
    
    
    #make final FoM plots if looping over many number of bis 
    if (len(bins) > 1):
        plotFinal(bins, chi2_array, slopeE_array) # print cor vs bins,
        plotFinal(bins, slope_array, slopeE_array, slope=True) # print slope vs bins

    print("Finishing plots on:", datetime.datetime.now())

# TODO calculate n explicitly (TDR, how James did it) for a range of voltages in question 
def get_Br(n, R_0, slope, slopeE):
    B_r = (slope * n) / (R_0)
    B_r_error = (slopeE * n) / (R_0)
    return B_r, B_r_error

def chi2Calc(data=[], error=[], pred=[]):
    N=len(data)
    if (N != len(pred)):
        print("chi2Calc::Data/pred array mismatch!")
        sys.exit()
    chi2= ((np.array(data)-np.array(pred))**2) / ((np.array(error))**2)
    chi2ndf=float(np.sum(chi2))/float(N-2)
    return chi2ndf

def getDBData(file_name):
    db_data = np.genfromtxt(file_name, dtype=db_type, delimiter=',')
    times = db_data['times']
    qhv = db_data['qhv'] 
    data = db_collection(times, qhv)
    return data

def getGPSData(file_name):
    gps_data = np.genfromtxt(file_name, dtype=gps_type, delimiter=',')
    times = gps_data['times']
    s12 = gps_data['s12'] 
    s18 = gps_data['s18'] 
    data = gps_collection(times, s12, s18)
    return data

def cleanDB(db_collection, gps_collection):
    time_db = db_collection.time
    qhv = db_collection.qhv
    time_gps = gps_collection.time

    new_time_db = [] 
    new_qhv=[]

    # remove extreme db times 
    for i_entry in range(len(time_db)):
        if(time_db[i_entry] >= time_gps[0] and time_db[i_entry] <= time_gps[-1]):
            new_time_db.append(time_db[i_entry])
            new_qhv.append(qhv[i_entry])

     
    print("DB length", len(new_time_db))
    print("GPS length", len(time_gps))
    print( "DB min, max", np.min(mdate.num2date(new_time_db)),  np.max(mdate.num2date(new_time_db)) )
    print("GPS min, max", np.min(mdate.num2date(time_gps)), np.max(mdate.num2date(time_gps)) )
    clean_data = db_collection_clean(new_time_db, new_qhv)
    return clean_data


def dbReshape(db_collection_clean, nbins=20): 
    data_len = len(db_collection_clean[0])
    rescale = int(data_len/nbins) # elements per bin 
    total_kept = rescale * nbins # can fit in bins equally
    total_removed = data_len - total_kept # drop elements that can fit 
    rescaled_data_array =[] 
    for i_array, array in enumerate(db_collection_clean):  #lop over each array in the collection 
        array = (array[:-total_removed or None]) # remove elements 
        rescaled_array = [] 
        rescaled_array_range = [] 
        for i_bin in range(nbins):  #loop over bins, as a range of rescaled elements 
            lower = rescale*i_bin 
            upper = rescale*(i_bin+1)
            sliced_array = array[lower:upper]
            rescaled_array.append(np.mean(sliced_array)) # take the mean of the slice (bin)
            if (i_array == 1):
                range_min_max = np.max(sliced_array)-np.min(sliced_array)
                rescaled_array_range.append(range_min_max)
        rescaled_data_array.append(rescaled_array)
        if (i_array == 1):
            rescaled_data_array.append(rescaled_array_range)

    dataR = db_collection_reshaped(*rescaled_data_array)  # push arrays into the collection 
    return dataR, rescale

def gpsReshape(gps_collection, nbins=20):
    data_len = len(gps_collection[0])
    rescale = int(data_len/nbins) # elements per bin 
    total_kept = rescale * nbins # can fit in bins equally
    total_removed = data_len - total_kept # drop elements that can fit 
    rescaled_data_array =[] 
    for i_array, array in enumerate(gps_collection):  #lop over each array in the collection 
        array = (array[:-total_removed or None]) # remove elements 
        rescaled_array = [] 
        rescaled_array_sem = [] 
        for i_bin in range(nbins):  #loop over bins, as a range of rescaled elements 
            lower = rescale*i_bin 
            upper = rescale*(i_bin+1)
            sliced_array = array[lower:upper]
            rescaled_array.append(np.mean(sliced_array)) # take the mean of the slice (bin)
            # if data is the <Y> from the station 
            if (i_array==1 or i_array==2):
                sem = stats.sem(sliced_array)
                rescaled_array_sem.append(sem)
        rescaled_data_array.append(rescaled_array)
        if (i_array==1 or i_array==2):
            rescaled_data_array.append(rescaled_array_sem)

    dataR = gps_collection_reshaped(*rescaled_data_array)  # push arrays into the collection 
    return dataR, rescale
    

def plotData(db_collection, gps_collection, rescale_db, rescale_gps, i_bin, mode):
    qhv = db_collection.qhv
    eqhv = db_collection.eqhv # not currently using x-error
    s12 = gps_collection.s12
    es12 = gps_collection.es12
    s18 = gps_collection.s18
    es18 = gps_collection.es18
    ver = [s12, s18]
    error = [es12, es18]

    #return arrays 
    chi2_array = []
    slope_array = []
    slopeE_array = []

    # inverse if default mode is passed    
    if (mode == 0):
        qhv = np.array(1/np.array(qhv))
        x_label = r"1/QHV [kV$^{-1}$]"
    else:
        x_label = "QHV [kV]"

    fig, ax = plt.subplots(2, 1, figsize=(8,10))
    
    for i_station, station in enumerate(stations):
        
        ### Linear fit 
        #get correlation, fit a line
        corr_matrix = np.corrcoef(qhv, ver[i_station])
        x_gen = np.linspace(float(min(qhv)), float(max(qhv)), num=1000) # generate x-points for evaluation 
        coefs, cov_matrix = np.polyfit(qhv, ver[i_station], 1, cov=True, w=error[i_station]) # x1 line [0=slope, 1=inter]
        fit = np.polyval(coefs, x_gen) # fit over generated points
        
        # unpack parameters and plot the fitted function 
        slope = coefs[0]
        intercept= coefs[1]
        #corr = np.round(corr_matrix[0][1], 3) # get correlation as a diag. element 
        chi2 = chi2Calc(data=ver[i_station],  error=error[i_station],  pred=np.polyval(coefs, qhv))
        slopeE = np.sqrt(cov_matrix[0][0])
        slope_array.append(slope) # per station 
        chi2_array.append(chi2) # per station
        slopeE_array.append(slopeE) # per station 
        ax[i_station].plot(x_gen, fit, color="red", linestyle="-", linewidth=2, label=r"$\frac{\chi^2}{DoF}$="+str(round(chi2,2))+"\n"+r"$\Delta \langle y \rangle \Delta QHV$"+"="+str(round(slope,3))+r" $\pm$ "+str(round(slopeE,3))+r" mm$\cdot$kV")

        #plot data 
        ax[i_station].minorticks_on()    
        ax[i_station].tick_params(axis='x', which='both', bottom=True, top=False, direction='inout')
        ax[i_station].tick_params(axis='y', which='both', left=True, right=False, direction='inout')
        ax[i_station].errorbar(qhv, ver[i_station], yerr=error[i_station], elinewidth=1, linewidth=0, capsize=0, color='green', label=station, marker="o")  
        ax[i_station].set_ylabel(r"$\langle y \rangle$"+": "+station + " [mm]", fontsize=font+2)
        ax[i_station].set_xlabel(x_label, fontsize=font+2)
        ax[i_station].legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), prop={'size': 15}) # outside (R) of the plot 
        ax[i_station].set_ylim(-0.08, 0.12)
        
        print("station:", station)
        print("slope [mm]:", round(slope,3),"+-",round(slopeE,3))
        print("slope [um]:", int(round(slope*1e3)),"+-",int(round(slopeE*1e3))) # nearest um 
        
        B_r, B_r_error = get_Br(n, R_0, slope, slopeE)
        print("B_r [ppm]", round(B_r*1e6,3), "+-", round(B_r_error*1e6,3))
        print("\n")

    #write to disk if only passing a single bin number
    if (len(bins) == 1 ): 
        plt.tight_layout()
        plt.savefig("../fig/YvsQHV_"+str(i_bin)+".png", dpi=300)

    return chi2_array, slope_array, slopeE_array

def plotFinal(bins, data_array, slopeE_array, slope=False):
    stationM=[ [], [] ] # mean per station 
    word = ["chi2", "slope"] # corr=False, slope=True
    fig, ax = plt.subplots(2, 1, figsize=(8,10))
    for i_station, station in enumerate(stations):
        ax[i_station].minorticks_on()
        ax[i_station].grid()
        for i_bin, the_bin in enumerate(bins):
            if (slope==False):
                ax[i_station].scatter(the_bin, data_array[i_bin][i_station], color='purple', label=station if(i_bin==0) else "")  
            else:
                ax[i_station].scatter(the_bin, data_array[i_bin][i_station], color='orange')  
                ax[i_station].errorbar(the_bin, data_array[i_bin][i_station], yerr=slopeE_array[i_bin][i_station], color='orange', label=station if(i_bin==0) else "")  
                stationM[i_station].append(data_array[i_bin][i_station])  #accumulate statistics per station
        #end of loop over bins 

        if (slope==False):
            ax[i_station].set_ylabel(r"\Chi2/ndf", fontsize=font+2, fontweight='bold')
        else:
            mean=np.mean(stationM[i_station]) # for plotting only 
            line =  [bins[0], mean] ,  [bins[-1], mean]
            ax[i_station].set_ylabel(r"Slope [mm$\cdot$kV$^{-1}$]", fontsize=font+2, fontweight='bold')
            ax[i_station].plot(*zip(*itertools.chain.from_iterable(itertools.combinations(line, 2))), color = 'black', linestyle="--", label="<slope>="+str(round(mean,3)) )
        ax[i_station].set_xlabel("Number of bins", fontsize=font+2, fontweight='bold')
        ax[i_station].tick_params(axis='x', which='both', bottom=True, direction='inout')
        ax[i_station].tick_params(axis='y', which='both', left=True, direction='inout')
        ax[i_station].legend(loc='upper center', bbox_to_anchor=(0.9, 1.2), prop={'size': 14}) # outside (R) of the plot    
    #write to disk
    plt.tight_layout()
    plt.savefig("../fig/Final_"+str(word[slope])+".png", dpi=100)

if __name__ == "__main__":
    main()
