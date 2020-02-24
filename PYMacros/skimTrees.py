# Author: Gleb Lukicov (21 Feb 2020)
# Open large ROOT Trees, apply cuts and save skimmed data as HDF5 

# Input: directory of ROOT trees
# Output: HDF5 skimmed files (both QualityTracks and QualityVertices as keys)

# Optional: combine HDF5 files into one 

import argparse
import re, glob, os, sys, re, subprocess
import pandas as pd
from root_pandas import read_root # see https://github.com/scikit-hep/root_pandas 
import h5py # https://github.com/h5py/h5py

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--trees", type=str, default="../DATA/Trees/60h_in1File") # dir with ROOT Trees 
arg_parser.add_argument("--df", type=str, default="../DATA/HDF/60h_skim") # path+fileLabel will be appended with filecount + ".hd5"
arg_parser.add_argument("--t_cut", type=int, default=30) # us 
arg_parser.add_argument("--p_cut", type=int, default=1800) # MeV
arg_parser.add_argument("--add", action='store_true', default=False)
arg_parser.add_argument("--add_label", type=str, default=None) # file label name
arg_parser.add_argument("--add_dir", type=str, default=None) # file label name
args=arg_parser.parse_args() 

#read both track and vertices
keys=('QualityTracks', 'QualityVertices')

#counter for all and skimmed tracks/vertices
total_tv, total_tv_skim = [0, 0], [0, 0]
    
def main():

    #default is to skim many Tree
    if(args.add==False):
        skim()

    #add HDF5s into one based on label (name)
    if(args.add==True):
        add()

 
def add():
    print("Adding HDF5 from", args.add_dir, "with label", args.add_label)
    selected = [] # selected files from the folder 

    #loop over all files and select based on label and ".h5" file format
    for i_file, file in enumerate(sorted(os.listdir(args.add_dir))):
        name=re.split(r'_|\.', file) # split by "_" and "."
        if(args.add_label in name and "h5" in name):
            selected.append(file)
    print("Found", len(selected), "file(s):", *selected, "adding...")
    
    #loop over the know keys and selected files 
    for key in keys:
        frames=[] # store all DF for that key 
        for file in selected:
            path=args.add_dir+"/"+file
            frames.append( pd.read_hdf(path, key) )
        result = pd.concat(frames) # add 
        result.to_hdf(args.add_dir+"/"+args.add_label+".h5", key=key, mode='a', complevel=9, complib="zlib", format="fixed")
    print("Added all the HDF files to", args.add_dir)

def skim():
    #open only the columns we need (for speed) over all the trees (files in folder)
    for i_file, file in enumerate(sorted(os.listdir(args.trees))):
        for i_key, key in enumerate(keys):
            print("Opening", key, "data in", args.trees+"/"+file)
            data_all = read_root(args.trees+"/"+file, key, columns=["station", "trackT0", "trackMomentum"])
            data_all['trackT0']=data_all['trackT0']*1e-3   # ns -> us
            total_tv[i_key]+=data_all.shape[0] # add to total from each file 
            print("Total of", data_all.shape[0], "entries")
           
            #define the time and energy cuts (otherwise the wiggle looks wobbly!)
            time_cut= (data_all['trackT0'] > args.t_cut)  # us, define a time_cut with time > 30 us 
            mom_cut = (data_all['trackMomentum'] > args.p_cut) # MeV 
            #Apply cuts in one go! 
            data = data_all[time_cut & mom_cut]
            total_tv_skim[i_key]+=data.shape[0] # add to total from each file 
            print("Total of", data.shape[0], "entries after energy and momentum cuts")

            #save the skimmed dataframe in a compressed HDF5 format
            # here we are appending to the file over tracks and then vertices
            print("Saving compressed data...")
            data=data.reset_index() # reset index from 0 
            cols_to_keep = ["station", "trackT0"] # only write for time and station 
            data[cols_to_keep].to_hdf(args.df+"_"+str(i_file)+".h5", key=key, mode='a', complevel=9, complib="zlib", format="fixed")
            print("Skimmed dataframe saved to disk", args.df, "\n")

    print("Grand total of (M)", total_tv[0]/1e6, "tracks,", total_tv[1]/1e6, "vertices")
    print("After the cuts (M)", total_tv_skim[0]/1e6, "tracks,", total_tv_skim[1]/1e6, "vertices")

if __name__ == "__main__":
    main()