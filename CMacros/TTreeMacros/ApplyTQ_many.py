'''
Python script to automate ApplyTQ.C over many runs
Functionality:
1) Form desired run range of files to apply TQ to
2) Run ApplyTQ.C over these files (one-by-one due to the virtual memory req.)
3) hadd the output them (again minding the virtual memory)
'''
import argparse 
import subprocess
import re, glob, os 
import numpy as np

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--start_run", type=int, default=15921) # inclusive 
arg_parser.add_argument("--end_run", type=int, default=15992)  # inclusive  
# arg_parser.add_argument("--input_dir", type=str, default="/gm2/data/users/glukicov/Run1_QualityTrees/")  
# arg_parser.add_argument("--out_dir", type=str, default="/gm2/data/users/glukicov/Run1_QualityTrees")
arg_parser.add_argument("--input_dir", type=str, default="/Users/gleb/software/EDMTracking/DATA/Trees/")  
arg_parser.add_argument("--out_dir", type=str, default="/Users/gleb/software/EDMTracking/TTreeMacros/")
arg_parser.add_argument("--applyTQ", action='store_true', default=False) 
arg_parser.add_argument("--hadd", action='store_true', default=False) 
args=arg_parser.parse_args()

if(args.applyTQ):
    start=args.start_run
    end=args.end_run
    inDir=args.input_dir+"/"
    outDir=args.out_dir+"/"
    print("Will apply TQ (copying) to Trees for runs:", start, "to", end)
    print("From", inDir)
    print("To", outDir)

    input("Press Enter to continue...")

    #now find all runs in range
    file_list = [] # full paths 
    runs =[] # some runs in range won't have a Tree (e.g. run is not a production run)
    all_runs = np.arange(start, end+1, 1) # inclusive of the last run, so+1 
    out_files =[]

    #look through directory 
    for run in all_runs:
        for name in glob.glob(inDir+"/trackRecoTrees_"+str(run)+".root"):
            print(name)
            file_list.append(name)
            runs.append(run)
            out_files.append(outDir+"qualityTrees_"+str(run)+".root")

    n_run=len(runs)
    print("Found", n_run, " files in range of runs")

    outfile1=open("input.txt", "w")
    outfile2=open("output.txt","w")

    outfile1.write("\n".join(file_list))
    outfile2.write("\n".join(out_files))


    print("Done! You can now run with '--hadd' on the quality tree")
        

else:
    raise Exception("Either '-hadd' or '--applyTQ' options must be given")