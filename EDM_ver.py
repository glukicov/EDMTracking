### Gleb 
# Looking into vertical beam from trackers at different quad settings 
from os import listdir
from os.path import isfile, join
import argparse, sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from ROOT import TFile

#Define constants
stations = ("12", "18")
colors = ("red", "blue")

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str) # offset scale (SD) [um]
args = parser.parse_args()
path=args.path

#get files 
files = [file for file in listdir(path) if isfile(join(path, file))]
files.sort()
print("Found", len(files),"files:", files)

# open and keep all in scope 
tfiles = []
for i_file in files:
    tfiles.append(TFile.Open(path+"/"+i_file))

histArray=[] #1D vertical beam 

# loop over files and stations
track_number = [[], []]
vertical = [ [], []]
vertical_error = [ [], []]
for i_file, file in enumerate(tfiles):
    for i_station, station in enumerate(stations):
        # histo_2D = file.Get("MomentumSlices/vertices/station"+str(station)+"/h_verticalPos_vs_time")
        histo_2D = file.Get("Extrapolation/vertices/station"+str(station)+"/h_verticalPos_vs_time")
        first_bin = histo_2D.GetXaxis().FindBin(30.0) # 30 us 
        tmpNameTH1 = "tmpNameTH1_"+str(i_file)+str(i_station) # assign a new "name pointer" to the TH1 object
        histo_1D = histo_2D.ProjectionY(tmpNameTH1, first_bin, -1) 
        histArray.append(histo_1D) # keep TH1D in scope if want them later 
        vertical[i_station].append(histo_1D.GetMean())
        vertical_error[i_station].append(histo_1D.GetMeanError())
        track_number[i_station].append(histo_1D.GetEntries())

for i_station, station in enumerate(stations):
    print("Tracks in S "+str(station)+" :", track_number[i_station])

### Plotting 
def makePlots(name, y_title, value, value_error, both=False):
    fig, ax = plt.subplots()
    x_ticks=np.char.replace(files, '.root', '')
    print("Making plot:", name)
    if (both==False):
        ax.scatter(x_ticks, value, label=label, color=color)
        plt.errorbar(x_ticks, value, yerr=value_error, color=color, markersize=12, elinewidth=1, capsize=2, linewidth=0)
    else:
        #then the passed value is 2D array 
        for i_station, station in enumerate(stations):
            ax.scatter(x_ticks, value[i_station], label="S"+station, color=colors[i_station])
            plt.errorbar(x_ticks, value[i_station], yerr=value_error[i_station],  color=colors[i_station], markersize=12, elinewidth=1, capsize=2, linewidth=0)

    #massaging plot 
    ax.set_ylabel(y_title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Setting/Run", fontsize=14, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 12}) # outside (R) of the plot 
    plt.tight_layout()
    plt.savefig("vertical_"+str(name)+".png", dpi=100)


#S12 and S18 
y_title="Vertical Beam position [mm]"
name="both"
makePlots(name, y_title, vertical, vertical_error, both=True)

# S18 - S12
name="diff"
y_title=r"$\Delta$ (S18-S12) Vertical [mm]"
label = r"$\Delta$ (S18-S12)"
color="green"
value = np.array(vertical[1])-np.array(vertical[0])
error = np.sqrt( np.array(vertical_error[1])**2 + np.array(vertical_error[0])**2 )
makePlots(name, y_title, value, error)