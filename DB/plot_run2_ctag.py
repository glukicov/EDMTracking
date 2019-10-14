import argparse,datetime,time,calendar
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import psycopg2
from collections import defaultdict
import sys

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def getLastRun(runMin,mins):
	# Get the last run within "mins" of now
	tEnd  = datetime.datetime.now() - datetime.timedelta(minutes=mins)
	sql = "select run,end_time,nevents from gm2dq.subrun_time \
	where run >= %d and end_time <='%s' and nevents > 1 order by end_time DESC limit 1" % (runMin,tEnd) 
	curr.execute(sql)
	conn.commit()
	rows = curr.fetchall()
	if (len(rows) != 1):
		print "Error: last run not found"
		sys.exit(-1)
	else:
		row = rows[0]
		run = int(row[0])
		return run

def getRunTimes(runMin,runMax,systematicRuns,ignoreRuns):
	# Get the number of fills and start time and end time for each subrun
	sql = "select run,subrun,start_time,end_time,nevents from gm2dq.subrun_time \
	where run >= %d and run <= %d and nevents > 1 and start_time > '1970-01-01 00:00:00' and end_time > '1970-01-01 00:00:00' order by run,subrun" % (runMin,runMax) 
	curr.execute(sql)
	conn.commit()
	rows = curr.fetchall()

	runTimes = defaultdict(int)
	for row in rows:

		run = int(row[0])
		subrun = int(row[1])
		start_time = int(time.mktime(row[2].timetuple()))
		end_time = int(time.mktime(row[3].timetuple()))
		nevents = int(row[4])

		if (run not in runTimes):
			d = {'Type' : 'P', 'NumSubRuns' : 0, 'StartTimes' : defaultdict(int), 'EndTimes' : defaultdict(int), 'Nevents' : 0}
			runTimes[run] = d

		d = runTimes[run]
		d['NumSubRuns'] = d['NumSubRuns'] + 1
		d['Nevents'] = d['Nevents'] + nevents

		for i in range(len(systematicRuns)/2):
			rMin = systematicRuns[2*i]
			rMax = systematicRuns[2*i+1]
			if (run >= rMin and run <= rMax):
				d['Type'] = 'S'  # systematic run
		for i in range(len(ignoreRuns)/2):
			rMin = ignoreRuns[2*i]
			rMax = ignoreRuns[2*i+1]
			if (run >= rMin and run <= rMax):
				d['Type'] = 'I'  # run to ignore


		st = d['StartTimes']
		et = d['EndTimes']
		st[subrun] = start_time
		et[subrun] = end_time 

	return runTimes

def StartEndTimes(runTimes,runMin,runMax):
	duration = 0
	startTime,x,y = StartEndTime(runTimes,runMin)
	x,endTime,y = StartEndTime(runTimes,runMax)

	for i in range(runMax-runMin+1):
		run = runMin + i 
		if (run in runTimes):
			d = runTimes[run]
			nsr = d['NumSubRuns']
			st = d['StartTimes'][0]
			et = d['EndTimes'][nsr-1]
			duration = duration + (et-st)
	return startTime,endTime,duration

def StartEndTime(runTimes,run):
	if (run in runTimes):
		d = runTimes[run]
		nsr = d['NumSubRuns']
		st = d['StartTimes'][0]
		et = d['EndTimes'][nsr-1]
		return datetime.datetime.fromtimestamp(st),datetime.datetime.fromtimestamp(et),et-st

	else:
		return None, None, None

def getDQMCTAG(runTimes):
	CTAG_DQMRun = defaultdict(int)
	for r in sorted(runTimes):
		d = runTimes[r]
		nsr = d['NumSubRuns']
		st = d['StartTimes'][0]
		et = d['EndTimes'][nsr-1]
		std = datetime.datetime.fromtimestamp(st)
		etd = datetime.datetime.fromtimestamp(et)
		sql = "select ctags from gm2ctag_dqm where time >= '%s' and time <= '%s'" % (std,etd)
		curr.execute(sql)
		conn.commit()
		rows = curr.fetchall()
		n = 0
		ct = 0.0
		for row in rows:
			n = n + 1
			ct = ct + float(row[0])
		if (n > 0):
			CTAG_DQMRun[r] = ct/float(n)
		else:
			CTAG_DQMRun[r] = -1.0
	return 	CTAG_DQMRun


def getNearline(runTimes):

	ctagsNL = np.array([])
	datesNL = np.array([])
	CTAGRun = defaultdict(int)
	CTAGRunTotal = defaultdict(int)
	CTAGNLFraction =  defaultdict(int)

	# Get the nearline ctag for these runs
	for r in sorted(runTimes):
		d = runTimes[r]
		Nevents = d['Nevents']
		sql = 'select nearline_ctag,run_number,subrun_number from nearline_processing where run_number = %d' % (r)
		curr.execute(sql)
		conn.commit()
		rows = curr.fetchall()
		ncx = 0
		nsr = 0
		stime = 0
		ntime = 0
		ctags = 0
		for row in rows:
			ct = int(row[0])
			srn = int(row[2])
			nsr = nsr + 1
			if (ct > 0):
				ctags = ctags + ct
				stime = stime + d['StartTimes'][srn]
				ntime = ntime + 1
				ncx   = ncx + 1

		nfills = Nevents*float(16.0)/float(34.0) # account for 16 laser trigers and 2 async
		if (nsr > 0 and ncx > 0):
			frac = float(ncx)/float(nsr)
			corrFac = 1.0/frac

			if (r > 26088 and r < 26168): # HACK SINCE NEARLINE IS WRONG (multiply by 2.88: average of previous 50 runs to last 10)
				corrFac = corrFac*2.88
			if (r > 26476 and r < 26491): # RATIO OF DQM/NL SLOPES
				corrFac = corrFac*1.79
			ctagsTotal = ctags*corrFac
			ctagsPerFill = ctagsTotal/float(nfills)
		else:
			frac = -1
			corrFac = -1
			ctagsTotal = 0
			ctagsPerFill = 0

		if (ntime > 0):
			avTime = int (float(stime)/float(ntime))
		else:
			avTime = -1
	

		CTAGNLFraction[r] = frac
		CTAGRun[r] = ctagsPerFill
		CTAGRunTotal[r] = ctagsTotal
		if (avTime > 0 and ctagsPerFill > 0):
			ctagsNL = np.append(ctagsNL,ctagsPerFill)
			datesNL = np.append(datesNL,avTime)

	return ctagsNL, datesNL,CTAGRun, CTAGRunTotal, CTAGNLFraction

def debug(runTimes,ctagsDQM,ctagNLRun,ctagNLRunTotal,ctagNLFraction,fname):

	f = open(fname,"w")
	f.write('Run, Run-Type,  Start,         End, Duration(s), # Subruns, DQM CTAG/fill, NL CTAG/fill, NL Frac, NL CTAGS\n')	

	startTime = 1000000000000;
	ctagTotal = 0
	for r in sorted(runTimes):
		if (r in ctagNLRunTotal):
	 		d = runTimes[r]
			nsr = d['NumSubRuns']
			rtype = d['Type']
			st = d['StartTimes'][0]
			if (st < startTime):
				startTime = st;
			et = d['EndTimes'][nsr-1]
			ct = ctagNLRunTotal[r]
			elapsed = (et - st)
			ctagTotal = ctagTotal + ct

			str = "%d, %s, %s, %s, %.1f, %d, %.1f, %.1f, %.3f, %.1f" % \
			(r,rtype,datetime.datetime.fromtimestamp(st),\
				datetime.datetime.fromtimestamp(et),\
				elapsed,nsr, ctagsDQM[r], ctagNLRun[r], ctagNLFraction[r], ctagNLRunTotal[r])
			f.write(str+"\n")

	f.close()

def doRun1(endTime):

	file = open('ctags_8hr_run1.txt','r')
	lines = file.readlines()
	t = datetime.datetime(2018, 1, 29, 0, 00, 00)

	times = []
	ctags = []

	for line in lines:
		ctag = float(line)
		tVal = int(time.mktime( t.timetuple() ))
		if (t > datetime.datetime(2018,3,18,0,00,00) and t < endTime):
			times.append(tVal)
			ctags.append(ctag)

		t = t + datetime.timedelta(hours=+8)

	d = np.array(mdate.epoch2num(times)) + 365.0
	ct = np.array(ctags)
	ctx = np.cumsum(ct)/BNL

	return d,ctx

def baseDesign(endTime):
	durationFinal = int(time.mktime(END_TIME.timetuple())) - int(time.mktime(START_TIME.timetuple()))
	duration = int(time.mktime(endTime.timetuple())) - int(time.mktime(START_TIME.timetuple()))
	fraction = float(duration)/float(durationFinal)

	ctagTotalBase = 1.65*fraction
	ctagTotalDesign = ctagTotalBase*2
	st = int(time.mktime(good_ctags.timetuple()))
	et = int(time.mktime(endTime.timetuple()))

	N = 200
	times = np.linspace(st, et, N, endpoint=True) - GMT  # so that it starts at midnight Chicago
	ctagsBase = np.linspace(0, ctagTotalBase, N, endpoint=True)
	ctagsDesign = np.linspace(0, ctagTotalDesign, N, endpoint=True)

	d = mdate.epoch2num(list(times))
	dates = np.array(d)

	return dates,ctagsBase,ctagsDesign

def doCTAG(endTime,runTimes,ctagNLRunTotal):

	ENDTIME = int(time.mktime( endTime.timetuple() ))
	ctags = np.array([])
	rtypes = np.array([])
	tV = []
	ctagTotal = 0.0
	for r in sorted(runTimes):
		if (r in ctagNLRunTotal):
	 		d = runTimes[r]
			nsr = d['NumSubRuns']
			et = d['EndTimes'][nsr-1]
			ct = ctagNLRunTotal[r]

			if (d['Type'] != 'I' and (et-ENDTIME) < 500):
				tV.append(et)
				ctagTotal = ctagTotal + ct/BNL
				ctags = np.append(ctags,ctagTotal)
				rtypes = np.append(rtypes,d['Type'])

	GAP = 6*3600 
	GAP =  100*24*3600
	cTotal = ctags[-1]
	rLast  = rtypes[-1]

	# Add extra point if nothing from nearline for GAP hours since if taking data should be there
	# Make this 6 hrs when taking data - else 100 days when not.
	if (NOW - tV[-1] > GAP):
		print "Now = %d, Last ctag NL = %d, Gap = %.1f hours" % (NOW,tV[-1],(NOW-tV[-1])/3600.0)
	# Add extra point at latest time with last value
		ctags = np.append(ctags,cTotal)
		rtypes = np.append(rtypes,rLast)
		tV.append(NOW)

	dates = np.array(mdate.epoch2num(tV))

# Sort by date time and apply this order to the ctags array 
	p = dates.argsort()
	xd = dates[p]
	yd = ctags[p]
	rd = rtypes[p]
	xd = xd - 6.0/24.0 # in local time

	return xd,yd,rd

def extactLastWeek(xd,yd,rd):
	endT = xd[-1]
	startT = endT - 7.0

	xdw = np.array([])
	ydw = np.array([])
	rdw = np.array([])

	for i in range(len(xd)):
		if (xd[i] >= startT and xd[i] <= endT):
			xdw = np.append(xdw,xd[i])
			ydw = np.append(ydw,yd[i])
			rdw = np.append(rdw,rd[i])

	ydw = ydw - ydw[0]

	return xdw, ydw,rdw


def doit(fname,dates,ctagsBase,ctagsDesign,xd,yd,rd,lines=None,run1x=None,run1y=None):
	date_fmt = '%m-%d %H:%M'
	fig, ax = plt.subplots()
	if (lines is not None):
		ax.plot(dates,ctagsDesign,label='Muon g-2 e+ (Run-2 = 2xRun-1)',color='red',linewidth=2)
		ax.plot(dates,ctagsBase,label='Muon g-2 e+ (Run-2 = 1xRun-1)',color='blue', linewidth=2)

# Plot systematic data in light green
	if (xd[0] < 737142):
		ax.plot(xd,yd,label='Muon g-2 Run-2 e+ (prodn.)',color='green',marker='o',\
			markersize=2.0,linewidth=2,markeredgewidth=0.0)
		# get the ID range when rd is 'S'
		ss = np.where(rd == 'S')[0]
		xx = consecutive(ss)
		for i in range(len(xx)):
			smin = xx[i][0]
			smax = xx[i][-1]
			if (i == 0):
				label = 'Muon g-2 Run-2 e+ (syst.)'
			else:
				label = None
			ax.plot(xd[smin:smax],yd[smin:smax],label=label,color='lightgreen',marker='o',\
				markersize=2.0,linewidth=2,markeredgewidth=0.0)
	else:
		date_fmt = '%m-%d %H:%M'
		ax.plot(xd,yd,label='Muon g-2 Run-2 e+ (last 7 days)',color='orange',marker='o',\
			markersize=2.0,linewidth=2,markeredgewidth=0.0)

	if (run1x is not None):
		ax.plot(run1x,run1y,label='Muon g-2 Run-1 e+',color='orange',marker='o',\
			markersize=2.0,linewidth=2,markeredgewidth=0.0,linestyle=':')

	#date_formatter = mdate.DateFormatter(date_fmt)
	#hours = mdate.HourLocator(interval = 96)
	#ax.xaxis.set_major_locator(hours)

	date_formatter = mdate.DateFormatter(date_fmt)
	ax.xaxis.set_major_formatter(date_formatter)
	fig.autofmt_xdate()
	ax.set_ylabel('Fraction of BNL')
	plt.legend(loc='upper left')
	plt.grid()
	plt.show()
#	plt.savefig(fname,dpi=1000)
	plt.savefig(fname)
	plt.close()


parser = argparse.ArgumentParser(description='Parameters for nearline CTAGs')
parser.add_argument('--db', type=str, required=True, dest='db', choices=['localhost','g2db-priv'], help='database connection')
args = parser.parse_args()
db = args.db

if (db ==  'localhost'):
    dsn  = "dbname=gm2_online_prod user=gm2_writer host=localhost port=5434"
elif (db == 'g2db-priv'):    
    dsn  = "dbname=gm2_online_prod user=gm2_writer host=g2db-priv port=5433"
else:
    print "None supported DB specified - run with --db=localhost or --db=g2db-priv"
    sys.exit(-1)


# GLOBALS....
NOW = int(time.mktime(datetime.datetime.now().timetuple()))
GMT = 5*3600
START_TIME = datetime.datetime(2019,3,18,0,0,0)
END_TIME = datetime.datetime(2019,7,6,8,0,0)
BNL = 8.6348e9 # e+/e-
conn = psycopg2.connect(dsn)
curr = conn.cursor()

# pairs start, end of systematic run periods.
systematicRuns = [24309, 24374, 25655, 25659, 25890, 25893, 25974, 25978, 26160, 
26167, 26181, 26196, 26342, 26357, 26342, 26357, 26580, 26584, 26596, 25607, 26598, 26601,
26654, 26659, 26672, 26674, 26681, 26682, 26690, 26690, 26775, 26782, 27088, 27125, 27173, 27187,27416,27435,27441,27543]

# fix 27480 --- presently at 27474

ignoreRuns = [26473, 26476]
runMin = 24309  # start of correct NL calibration: 24309; start of 1st production running: 24376; 24575 2nd prodn running 
runMax = getLastRun(runMin,120) # 2 hours prior to now
runTimes = getRunTimes(runMin,runMax,systematicRuns,ignoreRuns)

utc = int(calendar.timegm(time.gmtime()))
fname = 'Flask/static/%s_%d.csv' % ('run2',utc)
fname = 'Flask/static/%s_%d.csv' % ('XXX_ML',utc)
ctagsNL, datesNL, ctagNLRun, ctagNLRunTotal, ctagNLFraction = getNearline(runTimes)
ctagsDQM = getDQMCTAG(runTimes)

debug(runTimes,ctagsDQM,ctagNLRun,ctagNLRunTotal,ctagNLFraction,fname)

# Plots through to July without run-1
dates,ctagsBase,ctagsDesign = baseDesign(END_TIME)
fname = 'Flask/static/%s_%d.png' % ('ctagrun2end',utc)
fname = 'Flask/static/%s_%d.png' % ('XXX_ML',utc)
xd,yd,rt = doCTAG(END_TIME,runTimes,ctagNLRunTotal)
doit(fname,dates,ctagsBase,ctagsDesign,xd,yd,rt,1)

# Plots until now (change RunMin and START_TIME to get last week)
# endTime = datetime.datetime.now() + datetime.timedelta(hours=24)
# dates,ctagsBase,ctagsDesign = baseDesign(endTime)
# fname = 'Flask/static/%s_%d.png' % ('ctagrun2now',utc)
# xd,yd,rt = doCTAG(endTime,runTimes,ctagNLRunTotal)
# endRun1 = datetime.datetime.now() - datetime.timedelta(hours=365*24)
# run1x,run1y = doRun1(endRun1)
# doit(fname,dates,ctagsBase,ctagsDesign,xd,yd,rt,1,run1x,run1y)

# Plot of last week without run-1 and without projections
# fname = 'Flask/static/%s_%d.png' % ('ctagrun2week',utc)
# xdw, ydw, rdw = extactLastWeek(xd,yd,rt)
# print "Fraction BNL in last week = %.3f" % (ydw[-1])
# doit(fname,dates,ctagsBase,ctagsDesign,xdw,ydw,rdw)


