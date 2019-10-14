# /*
# *   Gleb Lukicov (g.lukicov@ucl.ac.uk)
# *   Created: 21 August 2019
# *   Estimating golden runs in gm2 
# * /
import psycopg2  #db query
import argparse # command line inputs sub
import numpy as np # fast arrays 
import matplotlib.dates as mdate # mpl date format 
import datetime
import os, sys 
from collections import defaultdict

#Define some constants 
START_TIME = datetime.datetime(2019,3,18,0,0,0) #Run-2 
END_TIME = datetime.datetime(2019,7,6,8,0,0) # Run-2
BNL = 8.6348e9 # e+/e-
CUT_SUBRUN = 10
CUT_DELTA = 0.0 # min
CUT_DELTA_DAY = CUT_DELTA/float(60*24) # days (mpl date format)
CUT_CTAG = 50.0 

UNPACK = 9 # MB 
TRACK = 170 # MB

#Storage devices 
runs_dump=[] # store all runs in the range (duplicates)
all_runs=[] # store all runs in the range (unique)

good_runs=[] # runs with more than CUT_SUBRUN subruns 
good_deltas = [ ] # stop-start times for good runs
good_times = [ [], [] ] # start and stop times for good runs
good_subruns = [] # subruns per good run 
good_ctags = [] # inst/N
#golden selection:
golden_runs = []
golden_deltas = []
golden_ctags = []
golden_subruns = []

###========================READ FROM FHICL AND WRITE ALIGNMENT CONSTANTS===============================##
def main():

    sys.stdout = Logger() ### Duplicate all cout into a log file

    print "Starting on:", datetime.datetime.now()

    print "Looking for subruns in range:",START_TIME,"to",END_TIME
   
    # ###OPEN CONNECTION (as a writer!)
    dsn  = "dbname=gm2_online_prod user=gm2_writer host=g2db-priv port=5433"
    cnx = psycopg2.connect(dsn)
    cur = cnx.cursor()

    #count total subruns in range 
    command = "select count(*) from gm2dq.subrun_time where start_time >= '" + str(START_TIME) + \
     "'::timestamp and end_time <= '" + str(END_TIME) + "'::timestamp ;"
    cur.execute(command)
    rows = cur.fetchall()
    total_subruns = 0
    for row in rows:
        total_subruns = int(row[0])

    # store all runs in range 
    command = "select run from gm2dq.subrun_time where start_time >= '" + str(START_TIME) + \
     "'::timestamp and end_time <= '" + str(END_TIME) + "'::timestamp ;"
    cur.execute(command)
    rows = cur.fetchall()
    for row in rows:
        runs_dump.append(int(row[0]))
        
    all_runs=set(runs_dump) # unique
    print "Total runs:",len(all_runs),"with",total_subruns,"subruns"
   
    #now loop over all runs and get the last and first subruns ordered by time to get the run duration (delta) 
    for i_run in all_runs:
        command = "select start_time, end_time from gm2dq.subrun_time where run = "+ str(i_run) +\
         " ORDER BY start_time ASC;"
        cur.execute(command)
        rows = cur.fetchall()
        # for runs with more than N subruns...
        if ( len(rows) >= CUT_SUBRUN ):
            first = rows[0][0] # 1st element of 1st return 
            last = rows[-1][1] # 2nd element of last return
            # if run crashed on the last subrun, get (n-1)th  
            if (last.year == 1969):
                last = rows[-2][1] 
            # same check for the first subrun get the 2nd element then 
            if (first.year == 1969):
                first = rows[1][0]
            else:
                delta = mdate.date2num(last)-mdate.date2num(first)
                if (delta >= CUT_DELTA_DAY):
                    good_runs.append(i_run)
                    good_deltas.append( delta )
                    good_times[0].append(first) 
                    good_times[1].append(last)    

    print 'Found',len(good_runs), "runs with more than",CUT_SUBRUN,"subruns and longer than", CUT_DELTA,"min"

    #count ctags for the good runs based on the start and stop time of a run
    ctag_sum=0.0 # instantaneous ctag 
    for i in range(len(good_runs)):
        start = good_times[0][i]
        stop = good_times[1][i]
        command = "select time, ctags from gm2ctag_dqm where time >= '"+ str(start)+ "'::timestamp and time <=  '"+  str(stop) +"'::timestamp ;"
        cur.execute(command)
        rows = cur.fetchall()
        if (len(rows) == 0):
            #remove a run if there is not ctag information  
            del good_runs[i]
            del good_times[0][i]
            del good_times[1][i]
            del good_deltas[i]
        else: 
            for row in rows:
                ctag_sum+=float(row[1])
            good_ctags.append(ctag_sum/float(len(rows))) # instantaneous ctag 
            ctag_sum=0.0

    # now that we know the good runs, count total good subruns 
    for i_run in good_runs:
        command = "select count(*) from gm2dq.subrun_time where run = "+ str(i_run) +" ;"
        cur.execute(command)
        rows = cur.fetchall()
        for row in rows:
            good_subruns.append(int(row[0])) # per good run 

    #get ctags for good runs 
    good_CTAGRun, good_ignoreRuns = getNearline(good_runs, cur, cnx)
    # remove runs with no nearline info 
    print "Removed",len(good_ignoreRuns),"runs with no nearline info"
    for i_run in good_ignoreRuns:
        element_id_array = [i for i,x in enumerate(good_runs) if x == i_run]
        element_id=element_id_array[0] # safe for list of unique runs 
        del good_runs[element_id]
        del good_ctags[element_id]
        del good_times[0][element_id]
        del good_times[1][element_id]
        del good_deltas[element_id]
        del good_subruns[element_id]

    print 'Found',len(good_runs), "runs,",np.sum(good_subruns),"subruns, with ctag information from gm2ctag_dqm"

    if (  (len(good_runs) != len(good_deltas)) or len(good_runs) != len(good_ctags) or  len(good_runs) != len(good_subruns) or len(good_runs) !=len(good_CTAGRun)):
        print "The good run info is not of the same size!"
        sys.exit()

    print "Total of",sum(good_subruns),"good subruns in",len(good_runs),"runs with more than",CUT_SUBRUN,"subruns and longer than", CUT_DELTA,"min"

    #write all the good runs, duration, ctag, and subruns per run 
    f=open("DBDump_all.csv", "w+")
    for i in range(len(good_runs)):
        f.write(str(good_runs[i])+ " "+str(good_deltas[i]) + " " + str(good_ctags[i]) + " "\
         + str(good_subruns[i]) + " " + str(good_CTAGRun[i])+ "\n")

    print 'Wrote info (DBDump_all.csv) for',len(good_runs),"good runs"
    
    ##
    ## Golden runs
    ## 
    #now form a list of golden runs based on ctag 
    for i in range(len(good_runs)):
        if (good_ctags[i] >= CUT_CTAG):
            golden_runs.append(good_runs[i])
            golden_deltas.append(good_deltas[i])
            golden_ctags.append(good_ctags[i])
            golden_subruns.append(good_subruns[i])

    #get ctags for golden runs 
    golden_CTAGRun, golden_ignoreRuns = getNearline(golden_runs, cur, cnx)
    # remove runs with no nearline info 
    print "Removed",len(golden_ignoreRuns),"runs with no nearline info"
    for i_run in golden_ignoreRuns:
        element_id_array = [i for i,x in enumerate(golden_runs) if x == i_run]
        element_id=element_id_array[0] # safe for list of unique runs
        del golden_runs[element_id]
        del golden_ctags[element_id]
        del golden_times[0][element_id]
        del golden_times[1][element_id]
        del golden_deltas[element_id]
        del golden_subruns[element_id]

    if (  (len(golden_runs) != len(golden_deltas)) or len(golden_runs) != len(golden_ctags) or  len(golden_runs) != len(golden_subruns) or len(golden_runs) !=len(golden_CTAGRun)):
        print "The gold run info is not of the same size!"
        sys.exit()

    print "Total of",sum(golden_subruns),"golden subruns in",len(golden_runs),"runs after CTAG cut of",CUT_CTAG

    f=open("DBDump_golden.csv", "w+")
    for i in range(len(golden_runs)):
            f.write(str(golden_runs[i])+ " "+str(golden_deltas[i]) + " " + str(golden_ctags[i])\
             + " " +  str(golden_subruns[i])  + " " + str(golden_CTAGRun[i])+ "\n")

    print 'Wrote info (DBDump_golden.csv) for',len(golden_runs),"golden runs"


    print "Estimation for UK tracking: Unpacked:",round(UNPACK*sum(golden_subruns)*1e-6,2),\
     "TB Tracks:",round(TRACK*sum(golden_subruns)*1e-6,2),"TB"


    # close communication with the databaes before committing 
    cur.close()
    cnx.commit()
    cnx.close()
    print "Finished on:", datetime.datetime.now()

### Helper functions 

# Mark's function 
def getNearline(all_runs, cur, cnx):
    CTAGRun = []
    ignoreRuns=[] # runs with no ctag 
    print "Getting nearline info (slow) for ctags..."
    # Get the nearline ctag for these runs
    for i_run in all_runs:
        #get the ctag from nearline table 
        sql = "select nearline_ctag from nearline_processing where run_number = "+str(i_run)+" ;"
        cur.execute(sql)
        rows = cur.fetchall()
        ctags = 0
        nsr = 0 # subruns 
        ncx = 0 # subruns with ctag 
        for row in rows:
            ct = int(row[0]) # per subrun 
            nsr = nsr + 1
            if (ct > 0):
                ncx   = ncx + 1 # count non-zero ctag subruns 
                ctags += ct # sum-up 
        frac = float(ncx)/float(nsr) # fraction with ctag
        if (frac == 0):
            ignoreRuns.append(i_run)
        if (frac != 0):
            corrFac = 1.0/frac # correct for non-ctag subruns s
            ctagsTotal = ctags*corrFac 
            if (i_run > 26088 and i_run < 26168): # HACK SINCE NEARLINE IS WRONG (multiply by 2.88: average of previous 50 runs to last 10)
                ctagsTotal = ctagsTotal*2.88
            if (i_run > 26476 and i_run < 26491): # RATIO OF DQM/NL SLOPES
                ctagsTotal = ctagsTotal*1.79
            CTAGRun.append(ctagsTotal)
    return CTAGRun, ignoreRuns

### Duplicate all cout into a log file
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("countSubRuns.log", "w+")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.terminal.flush()
        self.log.flush()

if __name__=="__main__":
    main()



# ##### Some psql shortcuts ####
# select * from gm2dq.subrun_time;
# psql -U gm2_writer -h localhost -p 5433 -d gm2_online_prod

# psql -U gm2_reader -h ifdbprod.fnal.gov -d gm2_online_prod -p 5452
# [connects to the Production DB, which is duplicated from the Online DB]

# \dt  - list all table
# \dn - list all schemas 
# \dt gm2tracker_sc.* - list all tables under the Tracker SC schema 
# set schema 'gm2tracker_sc';
# set schema 'gm2dq'; 

# select * from slow_control_items where name like '%HV%' limit 10;
# select * from slow_control_data where scid=946 limit 10; 

#cur.execute("select * from slow_control_data where scid=946 limit 10;")
# fetch all the rows
# rows = cur.fetchall()
# print rows


# CREATE TABLE gm2dq.tracker_hv (
#     id  SERIAL PRIMARY KEY,
#     station  smallint,
#     hv_status  BIT(64),
#     run integer,
#   subrun integer
# );

#####Correlating Run Sunbrun with timestamp ######
#You could access the content of DAQ ODB in table "gm2daq_odb", like the following
#select json_data->'Experiment'->'Security'->'RPC hosts'->'Allowed hosts' from gm2daq_odb ;
#select run_num, json_data->'Runinfo'->'Start time' from gm2daq_odb where run_num = 8000;
# select Subrun, json_data->'Logger'->'Channels'->'0'->'Settings' from gm2daq_odb

'''
 #print ("len(rows)", len(rows))
    for i_row in range(len(rows)):
        
        #print ("i_subrun", i_subrun, "i_row", i_row)
        i_subrun+=1
        subrun = int(rows[i_row][1])
        run = int(rows[i_row][0])
        

        # get start time of the first subrun 
        if (subrun == 0):
            start_time = rows[i_row][2]
            all_runs.append(run)
            #print "start_time", start_time

        # exit logic for the last element 
        if (i_row ==  len(rows)-1):
            #print "reached last row"
            subrunCount.append(i_subrun)
            end_time = rows[i_row][3]
            times[0].append(start_time)
            times[1].append(end_time)
            break 

        next_run = int(rows[i_row+1][0]) # (!) this is handled by interrupt exit log
        
        if ( run == next_run):
            pass # do nothing 
        else:
            subrunCount.append(i_subrun)
            end_time = rows[i_row][3]
            times[0].append(start_time)
            times[1].append(end_time)
            i_subrun=0
            print "start_time", start_time
            print "end_time", end_time
            print ("run", run, "subrun", subrun, "next_run", next_run)
'''