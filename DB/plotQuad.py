#########################################################################
# Plot Quad set points at various intervals  (quad_voltage_2)
# Based on Mark's DAQ plotter
# Modified by Gleb (14 Aug)  
#########################################################################
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

#Pass some commands (with defaults)
arg_parser = argparse.ArgumentParser(description='Input data file')
arg_parser.add_argument('--dbFile', type=str, required=True, dest='dbFile', default="data/Run2Data.csv", help='csv or hd5 DB file')
args = arg_parser.parse_args()
dbFile = args.dbFile

### Constants 
# data types from DB
types = [('times','|S32'),('pot',np.float32),('daq','|S32'),('quad_voltage_1',np.float32),\
('quad_voltage_2',np.float32),('quad_current',np.float32),('k1_voltage',np.float32),\
('k2_voltage',np.float32),('k3_voltage',np.float32),('inflector_current',np.float32),\
('magnet_current',np.float32),('dqm_ctag',np.float32),('dqm_t0',np.float32)]

days_range=-1 # number of days in range
GMT = 6.0/24.0 # dt conversion 

#These are detector threshold value for "Y" DQ
daq_DQ = ' DAQY' # on, (!) notice space 
quad_voltage_1_DQ = 10.0 # kV
quad_voltage_2_DQ = 14.5 # kV
quad_current_DQ = 10.0 # A 
kicker_voltage_DQ = 140.0  # kV k1+k2+k3
magnet_current_DQ = 5000  # A
inflector_current_DQ = 2700 # A
dqm_ctag_DQ = 0 # ctag 
dqm_t0_DQ = 0 # ctag 
pot_DQ = 1e14 # pot 

### Global containers
db_collection_shape = ['time', 'QV', 'pot', 'daq', 'quad_voltage_1',\
 'quad_current', 'k1_voltage', 'k2_voltage', 'k3_voltage', 'inflector_current',\
  'magnet_current', 'dqm_ctag', 'dqm_t0']
db_collection_shape_rescaled = ['time', 'QV', 'pot', 'quad_voltage_1',\
 'quad_current', 'k1_voltage', 'k2_voltage', 'k3_voltage', 'inflector_current',\
  'magnet_current', 'dqm_ctag', 'dqm_t0']
db_collection = collections.namedtuple('db_collection', db_collection_shape)
db_collection_rescaled = collections.namedtuple('db_collection_rescaled', db_collection_shape_rescaled)
db_collection_len = len(db_collection_shape)
status_array = [] 
### Main 
def main():
    sys.stdout = Logger() ### Duplicate all cout into a log file

    print "Starting plots on:", datetime.datetime.now()

    # matplotlib.use('Agg') # batch mode for plotting TODO

    print "Getting data from file..."
    data = getData(dbFile) # all db entries 

    #print "Reshaping data..."
    #dataR = dataReshape(data, nbins=100) # rescaled data by a factor of bins 

    print "Computing experiment status w.r.t Quads..."
    status_array = getStatus(data)

    plotData(times=data.time, values=data.QV, status=status_array, label="Quad Voltage [kV]", mode="days")

    #plotHist(values=data.QV, status=status_array, label="Quad Voltage [kV]", bins=17)

    # rescaled data in 5 min bins
    #dataR = dataReshape(data, nbins=13227) 
    #grad_data = getGradient(time=dataR.time, value=dataR.QV) # find dV/dt
    #Now plot quad values for each day  
    #plotData(times=dataR.time, values=np.gradient(dataR.QV), status=status_array, label="scaled dV/dt "+r"[kV min$^-1$]", mode="days")

    #rescaled data 
    #plotData(times=dataR.time, values=dataR.QV, label="Quad Voltage [kV]", mode="rescaled")
     #rescaled data 
    #plotData(times=dataR.time, values=dataR.magnet_current, label="Magnet Current [A]", mode="rescaled")
    # non-rescaled 
    # plotData(times=data.time, values=data.magnet_current, label="Magnet Current [A]", mode="rescaled")

    #print "Computing experiment status w.r.t Quads..."
    #status_array = getStatus(data)
    
    #Now plot quad values for each day  
    #plotData(times=data.time, values=data.QV, status=status_array, label="Quad Voltage [kV]", mode="days")

    print "Finishing plots on:", datetime.datetime.now()

### Functions 
# def getGradient(time=db_collection.time, value=db_collection.QV):

#     grad_data=[ [], [] ]

#     grad_data = [time, np.gradient(value)]
    
#     # print grad
#     # i=0
#     # while (i < len(time)-1):
#     #     print i
#     #     dVdt = (value[i+1] - value[i]) / (time[i+1]-time[i]) # gradient 
#     #     t = (time[i+1]+time[i]) / 2  # mean 
#     #     grad_data[0].append(t)
#     #     grad_data[1].append(dVdt)
#     #     i+=1

#     # print grad_data[1]
    
#     return grad_data

def getData(file_name):
    #Get data from csv 
    # TODO get based on namedtuple names/types in loop  
    db_data = np.genfromtxt(file_name, dtype=types, delimiter=',')
    dates = db_data['times']
    print"Dates [GMT] (first, last):",dates[0], dates[-1], "| len:",len(dates)
    # convert to matplotlib 0001 epoch [int are days since 0001]
    dates = np.array([mdate.datestr2num(item) for item in dates]) # apply time conversion GMT->GMT-6
    # dates = np.array([mdate.datestr2num(item)-GMT for item in dates]) # apply time conversion GMT->GMT-6
    days_range=int(dates[-1] - dates[0])
    print "Range of",days_range,"days of data"
    quad_voltage_2 = db_data['quad_voltage_2'] 
    print"Dates(first, last):",dates[0], dates[-1], "| len:",len(dates)
    print"QHV(first, last):",quad_voltage_2[0], quad_voltage_2[-1], "| len:",len(quad_voltage_2)
    pot =  db_data['pot']
    daq = db_data['daq']
    quad_voltage_1 = db_data['quad_voltage_1']
    quad_current = db_data['quad_current']
    k1_voltage = db_data['k1_voltage']
    k2_voltage = db_data['k2_voltage']
    k3_voltage = db_data['k3_voltage']
    inflector_current = db_data['inflector_current']
    magnet_current = db_data['magnet_current']
    dqm_ctag = db_data['dqm_ctag']
    dqm_t0 = db_data['dqm_t0']
    data = db_collection(dates, quad_voltage_2, pot, daq, quad_voltage_1, quad_current, k1_voltage,\
     k2_voltage, k3_voltage, inflector_current, magnet_current, dqm_ctag, dqm_t0)
    return data

def getStatus(db_data): 
    # loop over all entries to establish status
    for i in range(len(db_data.time)):
        kicker_voltage = db_data.k1_voltage[i] + db_data.k2_voltage[i] + db_data.k3_voltage[i]
        # if (db_data.daq[i] == ' DAQY' and db_data.quad_voltage_1[i] > quad_voltage_1_DQ and db_data.quad_current[i] > quad_current_DQ \
        #  and kicker_voltage > kicker_voltage_DQ and db_data.magnet_current[i] > magnet_current_DQ \
        #   and db_data.inflector_current[i] > inflector_current_DQ and db_data.pot[i] > pot_DQ and \
        #    db_data.dqm_t0[i] > dqm_t0_DQ and db_data.dqm_ctag[i] > dqm_ctag_DQ):
        if (db_data.daq[i] == daq_DQ and kicker_voltage > kicker_voltage_DQ  and db_data.magnet_current[i] > magnet_current_DQ \
            and db_data.inflector_current[i] > inflector_current_DQ and db_data.pot[i] > pot_DQ and db_data.dqm_t0[i] > dqm_t0_DQ \
             and db_data.dqm_ctag[i] > dqm_ctag_DQ):
            status_array.append(1.0)
        else:
            status_array.append(0.0)
    #print "status_array", status_array
    return status_array

def dataReshape(db_data, nbins=100):
    ## Data reshaping 
    # Average this every Na points so that we have nbins
    nbins = float(nbins)
    print "Using",nbins,"bins"
    total_data_len = len(db_data[0])
    print "Total of",total_data_len,"data points"
    rescale = int(total_data_len/nbins)
    print "rescale:",rescale
    #Rescale data as the mean in the bin
    rescaled_data_array =[]
    for i, i_tuple in enumerate(db_data):
        print"i_tuple before: ", i_tuple[0], i_tuple[-1],"| len:", len(i_tuple) # TODO get name
        if (i_tuple[-1] == " DAQY" or i_tuple[-1]== " DAQN"):
            print "Skipping DAQ data from rescale..."
            continue
        i_tuple = np.mean(i_tuple[:(len(i_tuple)/rescale)*rescale].reshape(-1,rescale), axis=1)
        print"i_tuple after: ", i_tuple[0], i_tuple[-1],"| len:", len(i_tuple) # TODO get name
        rescaled_data_array.append(i_tuple)
    dataR = db_collection_rescaled(*rescaled_data_array)
    return dataR

def plotData(times=db_collection.time, values=db_collection.QV, status=status_array, label="Quad Voltage [kV]", mode="rescaled"):    
    
    first_word = label.split(' ', 1)[0] # get plot name 

    if (mode=="rescaled"):
        # create directory for plots
        if (not os.path.isdir("rescaled")):
            subprocess.call(["mkdir", "rescaled"])

        fig, ax = plt.subplots()
        ax.set_ylabel(str(label), fontsize=14)
        plt.grid()
        ax.xaxis.set_major_formatter(mdate.DateFormatter('%d-%b'))
        fig.autofmt_xdate()
        ax.scatter(times, values, color='green', s=12.0)
        #Stats 
        mean = np.mean(values)
        meanVal = "Mean"+str(label)+" = "+ str(round(mean,3))
        #plt.text(times[2],mean/2,meanVal,fontsize=18)
        plt.savefig(mode+"/"+first_word+"_"+str(mode)+".png")
   
    if (mode == "days"):

        # create directory for plots
        if (not os.path.isdir(mode+first_word)):
            subprocess.call(["mkdir", mode+first_word])
        
        # temp containers [day][element]
        this_day_times=[[]]
        this_day_values=[[]] 
        this_day_status=[[]] 
        total_size=0 # check we got all data 
        day=0 # array dimension
        
        #loop over all times and check the day 
        for i_entry in range(len(times)):
            # exit logic for the last element 
            if (i_entry ==  len(times)-1):
                this_day_times[day].append(times[i_entry])
                this_day_values[day].append(values[i_entry])
                this_day_status[day].append(status[i_entry])
                break
            this_day = int(times[i_entry])
            next_day = int(times[i_entry+1]) # (!) this is handled by interrupt exit logic  
            if (this_day == next_day):
                this_day_times[day].append(times[i_entry])
                this_day_values[day].append(values[i_entry])
                this_day_status[day].append(status[i_entry])
            else:
                #append the last element for this day 
                this_day_times[day].append(times[i_entry])
                this_day_values[day].append(values[i_entry])
                this_day_status[day].append(status[i_entry])
                day+=1   # new dimension for the next day s
                this_day_times.append([])
                this_day_values.append([])
                this_day_status.append([])
        
        days_with_data = len(this_day_times)
        # check we got all elements correctly 
        if (len(times) != sum(len(x) for x in this_day_times)):
            print "your logic is wrong!"
            sys.exit()

        #now do some plotting 
        for i_day in range(days_with_data):
            # make new plot and format 
            fig, ax = plt.subplots()
            converted = mdate.num2date(this_day_times[i_day][0])
            this_day_date= str(converted.year)+"_"+str(converted.month)+"_"+str(converted.day)
            plt.title(this_day_date.replace("_","/"),fontsize=18)
            ax.set_ylabel(str(label), fontsize=14, color="green", fontweight='bold')
            ax.set_xlabel("Time [CDT]", fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.minorticks_on()
            ax.tick_params(axis='x', which='both', bottom=True, direction='inout')
            ax.tick_params(axis='y', which='both', left=True, direction='inout')
            plt.grid()
            ax.xaxis.set_major_formatter(mdate.DateFormatter('%H:%M'))
            fig.autofmt_xdate()
            
            # now, the second axes that shares the x-axis with the axs
            ax2 = ax.twinx()
            ax2.yaxis.tick_right()
            ax2.yaxis.set_label_position("right")
            ax2.set_ylabel("g-2 status", fontsize=14, color='purple', fontweight='bold')
            ax2.set_ylim(-0.2, 1.2)
            ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
            
            #plot data 
            ax.scatter(this_day_times[i_day], this_day_values[i_day], color='green', s=5.0, label=label)                          
            ax2.scatter(this_day_times[i_day], this_day_status[i_day], color='purple', s=5.0, label="g-2 status")

            #legend at the end 
            # box = ax.get_position()
            # ax.set_position([box.x0, box.y0, box.width * 0.65, box.height])
            # ax2.set_position([box.x0, box.y0, box.width * 0.65, box.height])
            # Shrink current axis by 30% to fit legend
            # ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.8), prop={'size': 8}) # outside (R) of the plot 
            # ax2.legend(loc='center left', bbox_to_anchor=(1.1, 0.7), prop={'size': 8}) # outside (R) of the plot 
            
            #write to disk
            plt.savefig(mode+first_word+"/"+first_word+"_"+str(this_day_date)+".png", dpi=300)
            pickle.dump(fig, open(mode+first_word+"/"+first_word+"_"+str(this_day_date)+".pickle", 'wb')) # This is for Python 3 - py2 may need `file` instead of `open`
            
            #write values to file
            f=open(mode+first_word+"/"+first_word+"_"+str(this_day_date)+".txt", "w+")
            for i_entry in range(len(this_day_times[i_day])):
                f.write(str(this_day_times[i_day][i_entry])+", "+str(this_day_values[i_day][i_entry])+"\n")

            plt.show()
            plt.cla()
            plt.clf()


def plotHist(values=db_collection.QV, status=status_array, label="Quad Voltage [kV]", bins=100):

    bin_min=12.5
    bin_max=21.0
    bin_width = (bin_max-bin_min)/bins
    print "bin_width: [kV]", bin_width

    font = {'size'   : 11}
    plt.rc('font', **font)

    int_status = np.array(status_array).astype(bool) # create a boolean mask 
    passed_values = values[int_status] # select on g-2 status = ok 
    passed_values = passed_values[passed_values>0.0]
    # passed_values_1=passed_values[passed_values<18.25]
    # passed_values_2=passed_values[passed_values>18.35]
    #passed_values=np.concatenate((passed_values_1, passed_values_2), axis=None)
    print "Total entries (status=ok, non-zero):", len(passed_values)
    plt.hist(passed_values, bins=bins, range=[bin_min, bin_max], color='green')  # arguments are passed to np.histogram
    plt.xlabel(label)
    plt.ylabel("Frequency [min] / "+str(bin_width)+ "kV")
    plt.savefig("hist_Quad.png", dpi=600)

    #ax.tick_params(axis='x', which='both', bottom=True, direction='inout')
    #plt.show()
        
### Helper functions

### Duplicate all cout into a log file
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("Plotter.log", "w+")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.terminal.flush()
        self.log.flush()

if __name__ == "__main__":
    main()


# go from string -> through float date  
# def tConvert(dString):
#     GMT = 6.0/24.0
#     dt = parser.parse(dString)
#     dti = int(time.mktime(dt.timetuple()))
#     dateX = mdate.epoch2num(dti)-GMT
#     return dateX


 # XXX For later
    #quad_voltage_1 = db_data['quad_voltage_2']
    #quad_current = db_data['quad_current']
    # pot =  db_data['pot']
    # daq = db_data['daq']
    # k1_voltage = db_data['k1_voltage']
    # k2_voltage = db_data['k2_voltage']
    # k3_voltage = db_data['k3_voltage']
    # inflector_current = db_data['inflector_current']
    # magnet_current = db_data['magnet_current']
    # dqm_ctag = db_data['dqm_ctag']
    # dqm_t0 = db_data['dqm_t0']
