import argparse,datetime,sys
import numpy as np
import psycopg2

def getThree(sql):
	curr.execute(sql)
	conn.commit()
	rows = curr.fetchall()
	x1 = np.array([])
	x2 = np.array([])
	x3 = np.array([])
	for row in rows:
		x1 = np.append(x1,float(row[0]))
		x2 = np.append(x2,float(row[1]))
		x3 = np.append(x3,float(row[2]))
	if (len(rows) > 0):
		return np.mean(x1),np.mean(x2),np.mean(x3)
	else:
		return float(-1.0),float(-1.0),float(-1.0)

def getOne(sql):
	curr.execute(sql)
	conn.commit()
	rows = curr.fetchall()
	x1 = np.array([])
	for row in rows:
		x1 = np.append(x1,float(row[0]))
	if (len(rows) > 0):
		return np.mean(x1)
	else:
		return float(-1.0)

def getOneArray(sql):
	curr.execute(sql)
	conn.commit()
	rows = curr.fetchall()
	x1 = np.array([])
	for row in rows:
		v = row[0]
		valid = v[3]
		if (valid > 0):
			x1 = np.append(x1,float(v[0]))
	if (len(rows) > 0):
		return np.mean(x1)
	else:
		return float(-1.0)


def getThreeTriplet(sql):
	curr.execute(sql)
	conn.commit()
	rows = curr.fetchall()
	x1 = np.array([])
	x2 = np.array([])
	x3 = np.array([])

	for row in rows:
		v = row[0]
		x1 = np.append(x1,float(v[0]))
		x2 = np.append(x2,float(v[1]))
		x3 = np.append(x3,float(v[2]))
	if (len(rows) > 0):
		return np.mean(x1),np.mean(x2),np.mean(x3)
	else:
		return float(-1.0),float(-1.0),float(-1.0)


parser = argparse.ArgumentParser(description='Parameters for nearline CTAGs')
parser.add_argument('--db', type=str, required=True, dest='db', choices=['localhost','g2db-priv'], help='database connection')
parser.add_argument('--run', type=str, required=True, dest='run', choices=['run1','run2'], help='dataset')
args = parser.parse_args()
db = args.db
run = args.run

if (db ==  'localhost'):
    dsn  = "dbname=gm2_online_prod user=gm2_writer host=localhost port=5434"
elif (db == 'g2db-priv'):    
    dsn  = "dbname=gm2_online_prod user=gm2_writer host=g2db-priv port=5433"
else:
    print "None supported DB specified - run with --db=localhost or --db=g2db-priv"
    sys.exit(-1)

conn = psycopg2.connect(dsn)
curr = conn.cursor()

# Set start and end times 
if (run == 'run2'):
	start = datetime.datetime(2019,03,23,00,00,00)
	end = datetime.datetime(2019,07,06,8,00,00)
	potCut = 1.0e14  # this is integrated
	fname = 'run2-Data.csv' 
	getMagnetData = 1
elif (run == 'run1'):
	start = datetime.datetime(2018,03,26,00,00,00)
	end = datetime.datetime(2018,07,9,23,59,00)
	potCut = 1.0e9 # this is instantaneous
	fname = 'run1-Data.csv' 
	getMagnetData = 0

timeX = start

f = open(fname,'w')
while (timeX < end):
	ts = timeX
	te = ts + datetime.timedelta(days=1)
	timeX = te
	sql = "select time,pot from gm2pot where time >= '%s' and time < '%s' " % (ts,te)
	curr.execute(sql)
	conn.commit()
	rows = curr.fetchall()
	total = 0.0
	daq = 0.0

	i = 0
	for row in rows:
		tV = row[0]
		pot = row[1]
		if (pot > potCut):
			total = total + 1.0
			tsd = tV - datetime.timedelta(seconds=30)
			ted = tV + datetime.timedelta(seconds=30)
			sql = "select start_time,end_time,nevents from gm2dq.subrun_time \
			where start_time >= '%s' and end_time < '%s'" % (tsd,ted)

			curr.execute(sql)
			conn.commit()
			rows = curr.fetchall()
			if (len(rows) > 0):
				daq = 'DAQY'
			else:
				daq = 'DAQN'

			sql = "select step1_voltage,step2_voltage,total_current from gm2quad_status where time >= '%s' and time < '%s'" % (tsd,ted)
			step1_voltage,step2_voltage,total_current = getThree(sql)

# Kicker
			if (run == "run1"):
			    sql = "select value from g2sc_values where channel='mscb282_DAC_P6' and\
			     time >= '%s' and time <= '%s' order by time ASC" % (tsd,ted)
			    k1_voltage,k2_voltage,k3_voltage = getThreeTriplet(sql)
			elif (run == "run2"):     
				sql= "select hv from gm2kicker_hv where hv_type='KVLJ' and kicker=1 and time >= '%s' and time < '%s'" % (tsd,ted)
				k1_voltage = getOne(sql)
				sql= "select hv from gm2kicker_hv where hv_type='KVLJ' and kicker=2 and time >= '%s' and time < '%s'" % (tsd,ted)
				k2_voltage = getOne(sql)
				sql= "select hv from gm2kicker_hv where hv_type='KVLJ' and kicker=3 and time >= '%s' and time < '%s'" % (tsd,ted)
				k3_voltage = getOne(sql)


			inflector_current = -99.0
			magnet_current = -99.0

			if (run == "run1" and getMagnetData == 1):
				sql = "select value from g2sc_values where channel='IFIX' and time >= '%s' and time <= '%s' order by time ASC" % (tsd,ted)
				inflector_current = getOneArray(sql)
			elif (run == "run2" and getMagnetData == 1):
				sql = "select value from gm2seeq_data where sqid=1 and time >= '%s' and time < '%s'" % (tsd,ted)
				inflector_current = getOne(sql)
				sql = "select value from gm2seeq_data where sqid=2 and time >= '%s' and time < '%s'" % (tsd,ted)
				magnet_current = getOne(sql)

			if (daq == 'DAQY'):
				sql = "select ctags from gm2ctag_dqm where time >= '%s' and time < '%s'" % (tsd,ted)
				ctags_dqm = getOne(sql)

				sql = "select value from gm2t0b_dqm where time >= '%s' and time < '%s'" % (tsd,ted)
				t0_dqm = getOne(sql)
			else:
				ctags_dqm = -1.0
				t0_dqm = -1.0

			str = '%s, %.2e, %s, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f' % \
			(tV,pot,daq,step1_voltage,step2_voltage,total_current,k1_voltage,\
				k2_voltage,k3_voltage,inflector_current,magnet_current,ctags_dqm,t0_dqm)
			f.write('%s\n' % (str))
			print str
			if (i%60 == 0):
				print datetime.datetime.now(),str
			i = i + 1			


f.close()
