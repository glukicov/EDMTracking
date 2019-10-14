# /*
# *   Gleb Lukicov (g.lukicov@ucl.ac.uk)
# *   Created: 21 August 2019
# * /

import argparse # command line inputs sub
import re, sys, os # to get offsets from file
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
from matplotlib.ticker import MaxNLocator
from scipy import stats
from io import StringIO

## Constants 
BNL = 8.6348e9 # e+/e-
CUT_SUBRUN = 50
CUT_DELTA = 0.0 # min
CUT_DELTA_DAY = CUT_DELTA/float(60*24) # days (mpl date format)
CUT_CTAG = 50.0

arg_parser = argparse.ArgumentParser(description='Input data file')
arg_parser.add_argument('--dbFile', type=str, required=True, dest='dbFile',help='csv or hd5 DB file')
arg_parser.add_argument('--dbFileNew', type=str, required=False, default=None, help='csv or hd5 DB file')
args = arg_parser.parse_args()
dbFile = args.dbFile
dbFileNew = args.dbFileNew

#if input and outputs DB files are supplied: make the new one 
if (dbFileNew != None):
    
    print("Writing a new DB file with cuts...")

    types = [('run',np.int), ('run_type','|S32'), ('start','|S32'), ('stop','|S32'),\
     ('run_length',np.float32),  ('subruns',np.int), ('dqm_ctag_fill',np.float32), \
     ('nl_ctag_fill',np.float32), ('nl_frac',np.float32), ('nl_ctags',np.float32),]
    
    db_data = np.genfromtxt(dbFile, dtype=types, delimiter=',', skip_header=1)
    
    run = db_data['run']
    run_type = db_data['run_type']
    run_type_str=[]
    for i_str in range(len(run_type)):
        run_type_str.append(str(run_type[i_str].split()[0]).split("'")[1])
    run_length = db_data['run_length']/(24*60*60)
    ctag = db_data['dqm_ctag_fill']
    subruns = db_data['subruns']
    ctag_total = db_data['nl_ctags']

    f=open(dbFileNew, "w+")

    for i in range(len(run)):
        if (ctag[i] > CUT_CTAG and subruns[i] > CUT_SUBRUN):
                f.write(str(run[i])+ " "+str(run_length[i]) + " " + str(ctag[i]) + " "\
                 + str(subruns[i]) + " " + str(ctag_total[i])+ "\n")

# if only one DB file is supplied: make some plots
if (dbFileNew == None):
    
    print("Plotting over a DB file...")

    types = [('run',np.int), ('run_length',np.float32), ('ctag',np.float32), ('subruns',np.int), ('ctag_run',np.float32)]
    db_data = np.genfromtxt(dbFile, dtype=types, delimiter=' ')
    run = db_data['run']
    run_length = db_data['run_length']
    ctag = db_data['ctag']
    subruns = db_data['subruns']
    ctag_total = db_data['ctag_run']

    run_length=run_length*24 # convert to hours 

    print("Total number of runs:",len(run))

    data = [run, run_length, ctag, subruns]

    #start, stop, step
    cut_ctag = np.arange(0, 200, 1)
    cut_length =  np.arange(0, 1.0, 0.01)
    cut_subruns= np.arange(0, 400, 1)

    cuts = [cut_length, cut_ctag, cut_subruns]
    cuts_names = ["Run length cut [h]", "CTAG cut", "Subrun cut"]

    #loop over the cuts 
    for i in range(len(cuts)):
        cut = cuts[i]
        cut_name = cuts_names[i]
        first_word = cut_name.split(' ', 1)[0] # get plot name 

        #add runs that has a value above the threshold 
        selected_runs = []
        selected_ctag = []
        for cut_value in cut:
            data_column=data[i+1]
            total_runs = 0
            total_ctag = 0
            for i_run in range(len(data[0])):
                if ( data_column[i_run] > cut_value):
                    total_runs+=1
                    total_ctag+=ctag_total[i_run]
            selected_runs.append(total_runs)
            selected_ctag.append(total_ctag)

        fig, ax = plt.subplots()
        ax.scatter(cut, np.array(selected_ctag)/BNL, color="green", s=12.0)
        ax.set_ylabel("Fraction of BNL")
        ax.set_xlabel(cut_name)
        plt.savefig(first_word+".png", dpi=600)
        print("Total CTAG:", selected_ctag[0]/BNL)

    print("Min run length [h]:", min(run_length))
    print("Max run length [h]:", max(run_length))

    font = {'size'   : 14}
    plt.rc('font', **font)

    fig, ax = plt.subplots()
    bins = 100
    bin_min=0.0
    bin_max=3.0
    bin_width = (bin_max-bin_min)/bins
    plt.hist(run_length, bins=bins, range=[bin_min, bin_max],  color='green')  
    plt.xlabel("Run Length [h]")
    plt.ylabel("Frequency / "+str(bin_width) + " [h]")
    mean = np.mean(run_length)
    sd = np.std(run_length)
    plt.title("$\mu= $"+str(round(mean,2))+" $\sigma= $"+str(round(sd,2)))
    # plt.show()
    plt.savefig("time_runs.png", dpi=600)

    print("Min ctag:", min(ctag))
    print("Max ctag:", max(ctag))

    fig, ax = plt.subplots()
    ax.scatter(run, ctag, color="green", s=12.0)
    ax.set_ylabel("<DQM CTAG>")
    ax.set_xlabel("run #")
    mean = np.mean(ctag)
    sd = np.std(ctag)
    plt.title("$\mu= $"+str(round(mean,2))+" $\sigma= $"+str(round(sd,2)))
    plt.xticks(np.arange(min(run), max(run)+1, 1000))
    plt.savefig("ctag_runs.png", dpi=600)
    # plt.show()

    print("Min subruns:", min(subruns))
    print("Max subruns:", max(subruns))

    fig, ax = plt.subplots()
    bins = 100
    bin_min=0
    bin_max=500
    bin_width = (bin_max-bin_min)/bins
    plt.hist(subruns, bins=bins, range=[bin_min, bin_max],  color='green')  
    plt.xlabel("Subruns per run")
    plt.ylabel("Frequency / "+str(bin_width) )
    runN = len(subruns)
    subrunN = np.sum(subruns)
    plt.title("Runs: "+str(runN)+" subruns: "+str(subrunN))
    # plt.show()
    plt.savefig("subrun_runs.png", dpi=600)