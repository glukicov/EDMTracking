import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
from dateutil import parser

def tConvert(dString):
	GMT = 6.0/24.0
	dt = parser.parse(dString)
	dti = int(time.mktime(dt.timetuple()))

	dateX = mdate.epoch2num(dti)-GMT

	return dateX;

types = [('times','|S32'),('pot',np.float32),('daq','|S32'),('quad_voltage_1',np.float32),\
('quad_voltage_2',np.float32),('quad_current',np.float32),('k1_voltage',np.float32),\
('k2_voltage',np.float32),('k3_voltage',np.float32),('inflector_current',np.float32),\
('magnet_current',np.float32),('dqm_ctag',np.float32),('dqm_t0',np.float32)]

data = np.genfromtxt('Run2Data.csv', dtype=types, delimiter=',')

dates = data['times']
pot =  data['pot']
daq = data['daq']
quad_voltage_1 = data['quad_voltage_1']
quad_voltage_2 = data['quad_voltage_2']
quad_current = data['quad_current']
k1_voltage = data['k1_voltage']
k2_voltage = data['k2_voltage']
k3_voltage = data['k3_voltage']
inflector_current = data['inflector_current']
magnet_current = data['magnet_current']
dqm_ctag = data['dqm_ctag']
dqm_t0 = data['dqm_t0']

times = np.array([])
values = np.array([])
for i in range(len(dates)):
	date = dates[i]
	kicker_voltage = k1_voltage[i] + k2_voltage[i] + k3_voltage[i]
	if (i < 5):
		print data[i]
	if (quad_voltage_1[i] > 10.0 and quad_voltage_2[i] > 14.5 and quad_current[i] > 10.0 and kicker_voltage > 140.0\
	    and magnet_current[i] > 5000 and inflector_current[i] > 2700 and pot[i] > 1e14):
		times = np.append(times,tConvert(date))
		if (daq[i] == ' DAQY'):
			values = np.append(values,1.0)
		elif (daq[i] == ' DAQN'):
			values = np.append(values,0.0)

plt.ion()
fig, ax = plt.subplots()

# Now average this every Na points so that we have nbins
nbins = float(100.0)
npX = len(values)
Na = int(npX/nbins)

Va = np.mean(values[:(len(values)/Na)*Na].reshape(-1,Na), axis=1)
Ta = np.mean(times[:(len(times)/Na)*Na].reshape(-1,Na), axis=1)

reorder = sorted(range(len(Ta)), key = lambda ii: Ta[ii])
Ta = [Ta[ii] for ii in reorder]
Va = [Va[ii] for ii in reorder]

mean = np.mean(values)
print Ta
print Va
ax.scatter(Ta,Va,color='red',s=12.0)
pstr = "Uptime = %.1f %%" % (100.0*mean)
print pstr
plt.text(Ta[2],1.1,pstr,fontsize=18)

ax.set_ylim([0.4,1.2])
date_fmt = '%d-%b'
date_formatter = mdate.DateFormatter(date_fmt)
ax.xaxis.set_major_formatter(date_formatter)
fig.autofmt_xdate()
ax.set_ylabel('DAQ Up Time')
plt.grid()
plt.show()
plt.savefig('plot_DAQUptime.png')
raw_input('Any key to continue')
plt.close()

