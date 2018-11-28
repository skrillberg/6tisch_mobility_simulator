# standard
import os
import argparse
import json
import glob
import sys
from collections import OrderedDict
import imageio

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

override = False
dirname = "SimData/20181016-120404-311"
no_figures = False
plot_all = False
plot_mote_num = 1
animation = False
matplotlib.use('TkAgg')
# =========================== defines =========================================
imgs =[]
DAGROOT_ID = 0  # we assume first mote is DAGRoot
DAGROOT_IP = 'fd00::1:0'
save_data_ebs = [] #list of dicts
def load_data(inputfile):
    global save_data_ebs 
    allstats = {} # indexed by run_id, mote_id

    file_settings = json.loads(inputfile.readline())  # first line contains settings
    print "Slotframe Length: ",file_settings["tsch_slotframeLength"]
    print "num motes: ",file_settings["exec_numMotes"]
    print "EB Probability" , file_settings["tsch_probBcast_ebProb"]
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
            rank       = logline['rank']
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
            x          = logline['x']
            y          = logline['y']
            reward      = logline['reward']
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
                'y' : y,
                'reward' : reward}
                ) 

        elif   logline['_type'] == SimLog.LOG_LOCATION_CONNECTIVITY ['type']:

            # shorthands
          
            asn        = logline['_asn']
            mote_id        = logline['src_mote']
            dst_mote    = logline['dst_mote']
            rssi        = logline['rssi']
            pdr         = logline['pdr']
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
            asn        = logline['_asn']
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
        
        elif logline['_type'] == SimLog.LOG_TSCH_EB_TX['type']: 
            mote_id    = logline['_mote_id']
            asn        = logline['_asn']
            packet     = logline['packet']
            # populate
            if mote_id not in allstats[run_id]:
                allstats[run_id][mote_id] = {}

            if 'eb_tx' not in allstats[run_id][mote_id]:
                allstats[run_id][mote_id]['eb_tx'] = []

            allstats[run_id][mote_id]['eb_tx'].append({'asn' : asn, 'time_s': asn*file_settings['tsch_slotDuration']}) 

        elif logline['_type'] == SimLog.LOG_TSCH_EB_RX['type']: 
            mote_id    = logline['_mote_id']
            asn        = logline['_asn']
            packet     = logline['packet']
            # populate
            if mote_id not in allstats[run_id]:
                allstats[run_id][mote_id] = {}

            if 'eb_rx' not in allstats[run_id][mote_id]:
                allstats[run_id][mote_id]['eb_rx'] = []

            allstats[run_id][mote_id]['eb_rx'].append({'asn' : asn, 'time_s': asn*file_settings['tsch_slotDuration']}) 

        elif logline['_type'] == SimLog.LOG_PROP_INTERFERENCE['type']: 
            mote_id    = logline['_mote_id']
            asn        = logline['_asn']
            channel     = logline['channel']
            # populate
            if mote_id not in allstats[run_id]:
                allstats[run_id][mote_id] = {}

            if 'collision' not in allstats[run_id][mote_id]:
                allstats[run_id][mote_id]['collision'] = []

            allstats[run_id][mote_id]['collision'].append({'asn' : asn, 'time_s': asn*file_settings['tsch_slotDuration']}) 

    run=0
    mote_num=1
    plot_div = 5
    count = 0
    for run in allstats:
        for mote_num in allstats[run]:
            if mote_num == plot_mote_num or plot_all:
                #print allstats
                print mote_num
                dataline =   {"slotframe_length" : file_settings["tsch_slotframeLength"],
                                "num_motes" : file_settings["exec_numMotes"],
                                "square_size" : file_settings["conn_random_square_side"],
                                "eb_rx_rate" : None,
                                "eb_prob" : file_settings["tsch_probBcast_ebProb"]


                                }
                if 'dios' in allstats[run][mote_num]:
                    plt.figure()
                    dios = pandas.DataFrame.from_dict(allstats[run][mote_num]['dios'])
                    plt.stem(dios['time_s'],numpy.ones(len(dios['time_s'])))
                    if no_figures:
                        plt.clf()

                if 'eb_tx' in allstats[run][mote_num]:
                    plt.figure()
                    eb_txs = pandas.DataFrame.from_dict(allstats[run][mote_num]['eb_tx'])
                    plt.stem(eb_txs['time_s'],numpy.ones(len(eb_txs['time_s'])))
                    plt.title('EB Transmissions: ' + str(len(eb_txs)/
                                                    (eb_txs['time_s'].iloc[-1] - eb_txs['time_s'].iloc[0])+0.000001) +
                                                    " Hz"

                    )
                    print("Average EB Tx Rate: ", len(eb_txs)/
                                                    (eb_txs['time_s'].iloc[-1] - eb_txs['time_s'].iloc[0]+0.000001)
                    )
                    dataline['eb_txs']=len(eb_txs)
                    if no_figures:
                        plt.clf()                

                if 'eb_rx' in allstats[run][mote_num]:
                    plt.figure()
                    eb_rxs = pandas.DataFrame.from_dict(allstats[run][mote_num]['eb_rx'])
                    plt.stem(eb_rxs['time_s'],numpy.ones(len(eb_rxs['time_s'])))
                    plt.title('EB Receptions: ' + str(len(eb_rxs)/
                                                    (eb_rxs['time_s'].iloc[-1] - eb_rxs['time_s'].iloc[0]+0.000001)) +
                                                    " Hz"
                    )
                    print("Average EB Rx Rate: ", len(eb_rxs)/
                                                    (eb_rxs['time_s'].iloc[-1] - eb_rxs['time_s'].iloc[0]+0.000001)
                    )
                    dataline["eb_rx_rate"] = len(eb_rxs)/ (eb_rxs['time_s'].iloc[-1] - eb_rxs['time_s'].iloc[0]+0.000001)
                    if no_figures:
                        plt.clf()

                if 'collision' in allstats[run][mote_num]:
                    plt.figure()
                    collisions = pandas.DataFrame.from_dict(allstats[run][mote_num]['collision'])
                    plt.stem(collisions['time_s'],numpy.ones(len(collisions['time_s'])))
                    plt.title('packet Collisions')

                    dataline["collisions"] = len(collisions)
                    if no_figures:
                        plt.clf()

                if 'churn' in allstats[run][mote_num]:
                    churn = pandas.DataFrame.from_dict(allstats[run][mote_num]['churn'])
                    #print churn
                    plt.figure()
                    plt.tight_layout()
                    plt.step(churn['time_s'],churn['parent'])
                    plt.xlabel('Time (s)')
                    plt.ylabel('Network Parent')
                    if no_figures:
                        plt.clf()

                if 'connectivity' in allstats[run][mote_num]:
                    conn = pandas.DataFrame.from_dict(allstats[run][mote_num]['connectivity'])
                    #print conn
                    plt.figure()
                    plt.plot(conn['time_s'],conn['rssi'])
                    if no_figures:
                        plt.clf()

                if 'packet_drops' in allstats[run][mote_num]:
                    packet_drops = pandas.DataFrame.from_dict(allstats[run][mote_num]['packet_drops'])
                    plt.figure()
                    plt.stem(packet_drops['time_s'],numpy.ones(len(packet_drops['time_s'])))
                    plt.title('packet drops')
                    if no_figures:
                        plt.clf()

                if 'location' in allstats[run][mote_num]:
                    
                    print "Plotting Locations and Topology"
                    locations = pandas.DataFrame.from_dict(allstats[run][mote_num]['location'])
                    fig = plt.figure()
                    for mote in allstats[run].keys():
                        mote_traj = pandas.DataFrame.from_dict(allstats[run][mote]['location'])
                        
                        plt.plot(mote_traj['x'],mote_traj['y'])
                        plt.xlabel("X Coordinate (km)")
                        plt.ylabel("Y Coordinate (km)")

                        plt.scatter(mote_traj['x'].iloc[-1],mote_traj['y'].iloc[-1])
                        plt.annotate(str(mote),(mote_traj['x'].iloc[-1],mote_traj['y'].iloc[-1]))
                    #if no_figures:
                     #   plt.clf()
                    for mote in range(0,len(allstats[run])):
                        if 'churn' in allstats[run][mote] and 'location' in allstats[run][mote]:
                            churn = pandas.DataFrame.from_dict(allstats[run][mote]['churn'])
                            #print str(churn['parent'].iloc[-1][-2:])
                            #print mote
                            #print churn
                            parent_num = int(str(churn['parent'].iloc[-1][-2:]),16)
                            child_location = pandas.DataFrame.from_dict(allstats[run][mote]['location'])
                            parent_location = pandas.DataFrame.from_dict(allstats[run][parent_num]['location'])
                            plt.plot((child_location['x'].iloc[-1],parent_location['x'].iloc[-1]),(child_location['y'].iloc[-1],parent_location['y'].iloc[-1]), '--')

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
                    plt.xlabel('Time (s)')
                    plt.figure()
                    #print locations
                    plt.plot(locations['time_s'],locations['reward'])
                    plt.title("rewards")
                    plt.ylabel("Reward")
                    plt.xlabel("Time (s)")
                    #if no_figures:
                        #plt.clf()
                #print "dataline ", dataline
                #print save_data_ebs
                
                if 'location' in allstats[run][mote_num] and animation:
                    
                    print "Plotting Animation Frames"
                    global imgs
   
                    
                    #build animation dataframe
                    anidata = {}
                    for mote in allstats[run].keys():
                        anidata[mote] = allstats[run][mote]['location']
                    anidata = pandas.DataFrame.from_dict(anidata)

                    print anidata.shape
                    print range(0,anidata.shape[0],1000)
                    i, files, removed = 0, [], []
                    # kargs = { 'macro_block_size' : None } # include **kargs in writer if running in linux
                    writer = imageio.get_writer(os.path.join(subfolder, "animation" + str(run)+'.mp4'), fps=20)

                    for timestep in range(0,anidata.shape[0],1000):
                        fig = plt.figure()

                        time_s = anidata.iloc[timestep][mote]["time_s"]
                        print timestep, time_s
                        for mote in anidata.keys():
                            #print mote
                            #print anidata.iloc[timestep]
                            mote_loc_x = anidata.iloc[timestep][mote]["x"]
                            mote_loc_y = anidata.iloc[timestep][mote]["y"]
                            #plt.plot(mote_traj['x'],mote_traj['y'])
                            plt.scatter(mote_loc_x,mote_loc_y)
                            plt.annotate(str(mote),(mote_loc_x,mote_loc_y))
                        #if no_figures:
                         #   plt.clf()
                        
                            if 'churn' in allstats[run][mote] and 'location' in allstats[run][mote]:
                                churn = pandas.DataFrame.from_dict(allstats[run][mote]['churn'])
                                #print str(churn['parent'].iloc[-1][-2:])
                                #print "printing churn"
                                #print churn

                                #find time step
                                index = None
                                for idx in range(0,churn.shape[0]-1):
                                    if time_s >= churn.iloc[idx]["time_s"] and time_s < churn.iloc[idx+1]["time_s"]:
                                        index = idx 
                                        break
                                    elif time_s >= churn.iloc[churn.shape[0]-1]["time_s"]:
                                        index =  churn.shape[0]-1
                                        break
                                if churn.shape[0] == 1:
                                    index = 0
                                if index is not None:
                                    parent_num = int(str(churn['parent'].iloc[index][-2:]),16)
                                    child_location = pandas.DataFrame.from_dict(allstats[run][mote]['location'])
                                    parent_location = pandas.DataFrame.from_dict(allstats[run][parent_num]['location'])
                                    plt.plot((child_location['x'].iloc[timestep],parent_location['x'].iloc[timestep]),(child_location['y'].iloc[timestep],parent_location['y'].iloc[timestep]), '--')
                        fname = os.path.join(subfolder,'_frame' + str(i) + '.png')
                        files.append(fname) # TODO: tweak so fps converts to correct timestamp in seconds
                        plt.xlim([0, 1])
                        plt.ylim([-0.1, 0.2])
                        plt.xlabel("X Location (km)")
                        plt.ylabel("Y Location (km)")
                        plt.savefig(files[i])
                        plt.close(fig)
                        writer.append_data(imageio.imread(files[i])[:, :, :])
                        os.remove(files[i])
                        removed.append(fname)
                        
                        #print fname
                        i += 1

                for mote_num in range(1,2):
                    #print "mote number: ",mote_num
                    if 'etx' in allstats[run][mote_num]:
                        etxs = pandas.DataFrame.from_dict(allstats[run][mote_num]['etx'])
                        #print etxs
                        plt.figure()
                        plt.plot(etxs['time_s'],etxs['etx'])
                        plt.title("ETX")
                        if no_figures:
                            plt.clf()
                    if 'neighbor_table' in allstats[run][mote_num]:
                        neighbor_table = pandas.DataFrame.from_dict(allstats[run][mote_num]['neighbor_table'])
                        dataline["num_neighbors"] = len(neighbor_table['neighbor_table'])
                        #print neighbor_table

                save_data_ebs.append(dataline)


            #plt.show()

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

print(subfolder)

if override:
    subfolder = dirname

files = glob.glob(os.path.join(subfolder,'*.dat'))
print files

if len(files) ==0:
    files = []
    subsubfolders = glob.glob(os.path.join(subfolder, '*')) 
    for paramfolder in subsubfolders:
        files += (glob.glob(os.path.join(paramfolder, '*.dat')))

#print files
for file in files:
    with open(file, "r") as read_file:
        load_data(read_file)
       
'''
df = pandas.DataFrame(save_data_ebs)
df.to_csv(os.path.join(subfolder,"eb_data"))
plt.close('all')
plt.figure()
print df

mote_x = df.slotframe_length.unique()
mote_num_x = []
rate_std = []
rate_avg = []
for index in mote_x:
    mote_num_x.append(index)
    sub_df = df[df["slotframe_length"] == index]
    rate_avg.append(numpy.mean(sub_df["eb_rx_rate"]/sub_df["num_neighbors"]))
    rate_std.append(numpy.std(sub_df["eb_rx_rate"]/sub_df["num_neighbors"]))

plt.errorbar(mote_num_x,rate_avg,rate_std,None)

plt.scatter(df["slotframe_length"],df["eb_rx_rate"])
plt.title("EB Reception Rate")
plt.ylabel("EB Reception Rate")
plt.xlabel("Slotframe Length")

plt.figure()
mote_x = df.num_motes.unique()
mote_num_x = []
rate_std = []
rate_avg = []
for index in mote_x:
    mote_num_x.append(index)
    sub_df = df[df["num_motes"] == index]
    rate_avg.append(numpy.mean(sub_df["eb_rx_rate"]/sub_df["num_neighbors"]))
    rate_std.append(numpy.std(sub_df["eb_rx_rate"]/sub_df["num_neighbors"]))

plt.errorbar(mote_num_x,rate_avg,rate_std,None)

plt.title("Effect of Number of Motes on EB Reception Rate per Neighbor")
plt.ylabel("EB Reception Rate per Neighbor (Hz/neighbor)")
plt.xlabel("Number of Motes in Network")

plt.figure()
plt.scatter(df["num_motes"],df["num_neighbors"])
plt.title("Number of Neighbors")
plt.ylabel("Number of Neighbors")
plt.xlabel("Number of Motes in Network")

plt.figure()
plt.scatter(df["eb_prob"],df["eb_rx_rate"]/df["num_neighbors"])
#plt.scatter(df["eb_prob"],df["collision_rate"])
plt.title("EB rx rate vs. EB prob")
plt.ylabel("RX rate per Neighbor")
plt.xlabel("EB prob")
'''

plt.show()