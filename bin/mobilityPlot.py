# standard
import os
import argparse
import json
import glob
import sys
from collections import OrderedDict

# third party
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas
import numpy

if __name__ == '__main__':
    here = sys.path[0]
    sys.path.insert(0, os.path.join(here, '..'))

#simulator
from SimEngine import SimLog
from SimEngine import SimEngine
import SimEngine.Mote.MoteDefines as d


# =========================== defines =========================================

DAGROOT_ID = 0  # we assume first mote is DAGRoot
DAGROOT_IP = 'fd00::1:0'

def load_data(inputfile):

    allstats = {} # indexed by run_id, mote_id

    file_settings = json.loads(inputfile.readline())  # first line contains settings

    # === gather raw stats

    for line in inputfile:
        logline = json.loads(line)

        # shorthands
        run_id = logline['_run_id']

        # populate
        if run_id not in allstats:
            allstats[run_id] = {}

        

        if   logline['_type'] == SimLog.LOG_RPL_CHURN['type']:

            # shorthands
            mote_id    = logline['_mote_id']
            asn        = logline['_asn']
            rank 	   = logline['rank']
            parent     = logline['preferredParent']

            # only log non-dagRoot sync times
            if mote_id == DAGROOT_ID:
                continue

            # populate
            if mote_id not in allstats[run_id]:
                allstats[run_id][mote_id] = {}
                        # test for existence of mote
           	#test for existence of churn
            if 'churn' not in allstats[run_id][mote_id]:
            	allstats[run_id][mote_id]['churn'] = [] 

            allstats[run_id][mote_id]['churn'].append({'asn' : asn,
            	'time_s' : asn*file_settings['tsch_slotDuration'], 
            	'parent' : parent, 
            	'rank' : rank}) 

        elif   logline['_type'] == SimLog.LOG_RPL_DIO_TX ['type']:

            # shorthands
            mote_id    = logline['_mote_id']
            asn        = logline['_asn']

            # only log non-dagRoot sync times
            if mote_id == DAGROOT_ID:
                continue

            # populate
            if mote_id not in allstats[run_id]:
                allstats[run_id][mote_id] = {}
                        # test for existence of mote
           	#test for existence of churn
            if 'dios' not in allstats[run_id][mote_id]:
            	allstats[run_id][mote_id]['dios'] = [] 

            allstats[run_id][mote_id]['dios'].append({'asn' : asn,
            	'time_s' : asn*file_settings['tsch_slotDuration']}) 

        elif   logline['_type'] == SimLog.LOG_LOCATION_UPDATE ['type']:

            # shorthands
            mote_id    = logline['_mote_id']
            asn        = logline['_asn']
            x		   = logline['x']
            y	       = logline['y']

            # populate
            if mote_id not in allstats[run_id]:
                allstats[run_id][mote_id] = {}
                        # test for existence of mote
           	#test for existence of x and y
            if 'location' not in allstats[run_id][mote_id]:
            	allstats[run_id][mote_id]['location'] = [] 

            allstats[run_id][mote_id]['location'].append({'asn' : asn,
            	'time_s' : asn*file_settings['tsch_slotDuration'],
            	'x' : x,
            	'y' : y}
            	) 

        elif   logline['_type'] == SimLog.LOG_LOCATION_CONNECTIVITY ['type']:

            # shorthands
          
            asn        = logline['_asn']
            mote_id		   = logline['src_mote']
            dst_mote	= logline['dst_mote']
            rssi		= logline['rssi']
            pdr 		= logline['pdr']
            #print connectivity

        

            # populate
            if mote_id not in allstats[run_id]:
                allstats[run_id][mote_id] = {}
                        # test for existence of mote
           	#test for existence of x and y
            if 'connectivity' not in allstats[run_id][mote_id]:
            	allstats[run_id][mote_id]['connectivity'] = [] 

            allstats[run_id][mote_id]['connectivity'].append({
            	'asn' : asn,
            	'rssi' : rssi,
            	'pdr' : pdr,
            	'time_s' : asn*file_settings['tsch_slotDuration'],
            	'dst_mote' : dst_mote
            	}
            ) 

        elif logline['_type'] == SimLog.LOG_PACKET_DROPPED['type']:
			# packet dropped

			# shorthands
			mote_id    = logline['_mote_id']
			reason     = logline['reason']
			asn 	   = logline['_asn']
			# populate
			if mote_id not in allstats[run_id]:
				allstats[run_id][mote_id] = {}

			if 'packet_drops' not in allstats[run_id][mote_id]:
				allstats[run_id][mote_id]['packet_drops'] = []

			allstats[run_id][mote_id]['packet_drops'].append({'asn' : asn, 'time_s': asn*file_settings['tsch_slotDuration']}) 

        elif logline['_type'] == SimLog.LOG_MOBILITY_ETX['type']:
            # packet dropped

            # shorthands
            mote_id    = logline['_mote_id']
            neighbor     = logline['neighbor']['mac_addr']
            numTx      =logline['neighbor']['numTx']
            numTxAck    =logline['neighbor']['numTxAck']
            asn        = logline['_asn']
            etx        = logline['etx']
            # populate
            if mote_id not in allstats[run_id]:
                allstats[run_id][mote_id] = {}

            if 'etx' not in allstats[run_id][mote_id]:
                allstats[run_id][mote_id]['etx'] = []

            allstats[run_id][mote_id]['etx'].append({'asn' : asn, 'time_s': asn*file_settings['tsch_slotDuration'],'neighbor': neighbor,'numTx':numTx,'numTxAck':numTxAck,'etx':etx}) 

        elif logline['_type'] == SimLog.LOG_MOBILITY_NEIGHBORS['type']:
            # packet dropped

            # shorthands
            mote_id    = logline['_mote_id']
            neighbor_table     = logline['neighbor_table']
            asn        = logline['_asn']
            # populate
            if mote_id not in allstats[run_id]:
                allstats[run_id][mote_id] = {}

            if 'neighbor_table' not in allstats[run_id][mote_id]:
                allstats[run_id][mote_id]['neighbor_table'] = []

            allstats[run_id][mote_id]['neighbor_table'].append({'asn' : asn, 'time_s': asn*file_settings['tsch_slotDuration'],'neighbor_table': neighbor_table}) 

    run=0
    mote_num=1
    #print allstats
    if 'dios' in allstats[run][mote_num]:
    	dios = pandas.DataFrame.from_dict(allstats[run][mote_num]['dios'])
    	plt.stem(dios['time_s'],numpy.ones(len(dios['time_s'])))

    if 'churn' in allstats[run][mote_num]:
        churn = pandas.DataFrame.from_dict(allstats[run][mote_num]['churn'])
        print churn
        plt.figure()
        plt.step(churn['time_s'],churn['parent'])
    
    if 'connectivity' in allstats[run][mote_num]:
		conn = pandas.DataFrame.from_dict(allstats[run][mote_num]['connectivity'])
		print conn
		plt.figure()
		plt.plot(conn['time_s'],conn['rssi'])

    if 'packet_drops' in allstats[run][mote_num]:
		packet_drops = pandas.DataFrame.from_dict(allstats[run][mote_num]['packet_drops'])
		plt.figure()
		plt.stem(packet_drops['time_s'],numpy.ones(len(packet_drops['time_s'])))
 	        plt.title('packet drops')

    if 'location' in allstats[run][mote_num]:
        locations = pandas.DataFrame.from_dict(allstats[run][mote_num]['location'])
        plt.figure()
        for mote in allstats[0]:
            mote_traj = pandas.DataFrame.from_dict(allstats[run][mote]['location'])
            plt.plot(mote_traj['x'],mote_traj['y'])
            plt.scatter(mote_traj['x'].iloc[-1],mote_traj['y'].iloc[-1])
            plt.annotate(str(mote),(mote_traj['x'].iloc[-1],mote_traj['y'].iloc[-1]))

        for mote in range(0,len(allstats[run])):
            if 'churn' in allstats[run][mote] and 'location' in allstats[run][mote]:
                churn = pandas.DataFrame.from_dict(allstats[run][mote]['churn'])
                print str(churn['parent'].iloc[-1][-2:])
                parent_num = int(str(churn['parent'].iloc[-1][-2:]),16)
                child_location = pandas.DataFrame.from_dict(allstats[run][mote]['location'])
                parent_location = pandas.DataFrame.from_dict(allstats[run][parent_num]['location'])
                plt.plot((child_location['x'].iloc[-1],parent_location['x'].iloc[-1]),(child_location['y'].iloc[-1],parent_location['y'].iloc[-1]))

        plt.figure()
        plt.subplot(311)
        plt.title('Mote 1 Locations')
        plt.plot(locations['time_s'],locations['x'])
        plt.xlabel('Time (s)')

        plt.subplot(312)
        plt.plot(locations['time_s'],locations['y'])
        plt.xlabel('Time (s)')

        plt.subplot(313)
        plt.plot(locations['time_s'],numpy.sqrt(numpy.square(locations['x']) + numpy.square(locations['y'])))


    for mote_num in range(1,2):
        print "mote number: ",mote_num
        if 'etx' in allstats[run][mote_num]:
            etxs = pandas.DataFrame.from_dict(allstats[run][mote_num]['etx'])
            print etxs
            plt.figure()
            plt.plot(etxs['time_s'],etxs['etx'])

        if 'neighbor_table' in allstats[run][mote_num]:
            neighbor_table = pandas.DataFrame.from_dict(allstats[run][mote_num]['neighbor_table'])
            print neighbor_table



    plt.show()
    

data = OrderedDict()

# chose lastest results
'''
subfolders = list(
    map(lambda x: os.path.join(options.inputfolder, x),
        os.listdir(options.inputfolder)
    )
)
'''
subfolders = list(
    map(lambda x: os.path.join("SimData", x),
        os.listdir("SimData")
    )
)
subfolders = glob.glob(os.path.join('SimData', '*'))
subfolder = max(subfolders, key=os.path.getmtime)




files = glob.glob(os.path.join(subfolder,'*.dat'))
print file
#


for file in files:
	with open(file, "r") as read_file:
		load_data(read_file)

