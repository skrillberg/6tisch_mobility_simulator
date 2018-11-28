"""
\brief Discrete-event simulation engine.
"""

# ========================== imports =========================================

import hashlib
import platform
import random
import sys
import threading
import time
import traceback
import math

import Mote
import SimSettings
import SimLog
import Connectivity
import SimConfig
import numpy
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import sys
import pandas
import netaddr
import tensorflow as tf
import tensorflow.contrib.layers as layers



from SimpleXMLRPCServer import SimpleXMLRPCServer
from SimpleXMLRPCServer import SimpleXMLRPCRequestHandler
import xmlrpclib
import zerorpc


# =========================== defines =========================================

# =========================== body ============================================

class DiscreteEventEngine(threading.Thread):
    
    #===== start singleton
    _instance      = None
    _init          = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DiscreteEventEngine,cls).__new__(cls, *args, **kwargs)
        return cls._instance
    #===== end singleton

    def __init__(self, cpuID=None, run_id=None, verbose=False,session = None, alg = None):

        #===== singleton
        cls = type(self)
        if cls._init:
            return
        cls._init = True
        #===== singleton
        
        try:
            # store params
            self.cpuID                          = cpuID
            self.run_id                         = run_id
            self.verbose                        = verbose

            # local variables
            self.dataLock                       = threading.RLock()
            self.pauseSem                       = threading.Semaphore(0)
            self.simPaused                      = False
            self.goOn                           = True
            self.asn                            = 0
            self.exc                            = None
            self.events                         = []
            self.random_seed                    = None
            self._init_additional_local_variables()
            #self.session                        = session
            #self.alg                            = alg
            # initialize parent class
            threading.Thread.__init__(self)
            self.name                           = 'DiscreteEventEngine'
            self.socket = xmlrpclib.ServerProxy('http://127.0.0.1:8001')
            self.firststep = True



        except:
            # an exception happened when initializing the instance
            
            # destroy the singleton
            cls._instance         = None
            cls._init             = False
            raise

    def destroy(self):
        if self._Thread__initialized:
            # initialization finished without exception
            
            if self.is_alive():
                # thread is start'ed
                self.play()           # cause one more loop in thread
                self._actionEndSim()  # causes self.gOn to be set to False
                self.join()           # wait until thread is dead
                del self.socket
            else:
                # thread NOT start'ed yet, or crashed
                
                # destroy the singleton
                cls = type(self)
                cls._instance         = None
                cls._init             = False
        else:
            # initialization failed
            pass # do nothing, singleton already destroyed

    #======================== thread ==========================================

    def run(self):
        """ loop through events """
        try:
            # additional routine
            self._routine_thread_started()

            # consume events until self.goOn is False
            while self.goOn:

                with self.dataLock:
                    
                    # abort simulation when no more events
                    if not self.events:
                        break
                    
                    # make sure we are in the future
                    (a, b, cb, c) = self.events[0]
                    if c[1] != '_actionPauseSim':
                        assert self.events[0][0] >= self.asn
                    
                    # update the current ASN
                    self.asn = self.events[0][0]
                    
                    # find callbacks for this ASN
                    cbs = []
                    while True:
                        if (not self.events) or (self.events[0][0] != self.asn):
                            break
                        (_, _, cb, _) = self.events.pop(0)
                        cbs += [cb]
                        
                # call the callbacks (outside the dataLock)
                
                for cb in cbs:
                    cb()

        except Exception as e:
            # thread crashed
            
            # record the exception
            self.exc = e
            
            # additional routine
            self._routine_thread_crashed()
            
            # print
            output  = []
            output += ['']
            output += ['==============================']
            output += ['']
            output += ['CRASH in {0}!'.format(self.name)]
            output += ['']
            output += [traceback.format_exc()]
            output += ['==============================']
            output += ['']
            output += ['The following settings are used:']
            output += ['']
            for k, v in SimSettings.SimSettings().__dict__.iteritems():
                if (
                        (k == 'exec_randomSeed')
                        and
                        (v in ['random', 'context'])
                    ):
                    # put the random seed value in output
                    # exec_randomSeed: random
                    v = '{0} ({1})'.format(v, self.random_seed)
                output += ['{0}: {1}'.format(str(k), str(v))]
            output += ['']
            output += ['==============================']
            output += ['']
            output  = '\n'.join(output)
            sys.stderr.write(output)

            # flush all the buffered log data
            SimLog.SimLog().flush()

        else:
            # thread ended (gracefully)
            
            # no exception
            self.exc = None
            
            # additional routine
            self._routine_thread_ended()
            
        finally:
            
            # destroy this singleton
            cls = type(self)
            cls._instance                      = None
            cls._init                          = False

    def join(self):
        super(DiscreteEventEngine, self).join()
        if self.exc:
            raise self.exc

    #======================== public ==========================================
    
    # === getters/setters

    def getAsn(self):
        return self.asn

    def get_mote_by_mac_addr(self, mac_addr):
        for mote in self.motes:
            if mote.is_my_mac_addr(mac_addr):
                return mote
        return None
    
    def get_mote_by_mote_id(self, mote_id):
        for mote in self.motes:
            if mote.id == mote_id:
                return mote
        return None

    #=== scheduling
    
    def scheduleAtAsn(self, asn, cb, uniqueTag, intraSlotOrder):
        """
        Schedule an event at a particular ASN in the future.
        Also removed all future events with the same uniqueTag.
        """

        # make sure we are scheduling in the future
        assert asn > self.asn

        # remove all events with same uniqueTag (the event will be rescheduled)
        self.removeFutureEvent(uniqueTag)

        with self.dataLock:

            # find correct index in schedule
            i = 0
            while i<len(self.events) and (self.events[i][0] < asn or (self.events[i][0] == asn and self.events[i][1] <= intraSlotOrder)):
                i +=1

            # add to schedule
            self.events.insert(i, (asn, intraSlotOrder, cb, uniqueTag))
    
    def scheduleIn(self, delay, cb, uniqueTag, intraSlotOrder):
        """
        Schedule an event 'delay' seconds into the future.
        Also removed all future events with the same uniqueTag.
        """

        with self.dataLock:
            asn = int(self.asn + (float(delay) / float(self.settings.tsch_slotDuration)))

            self.scheduleAtAsn(asn, cb, uniqueTag, intraSlotOrder)

    # === play/pause

    def play(self):
        self._actionResumeSim()

    def pauseAtAsn(self,asn):
        self.scheduleAtAsn(
            asn              = asn,
            cb               = self._actionPauseSim,
            uniqueTag        = ('DiscreteEventEngine', '_actionPauseSim'),
            intraSlotOrder   = Mote.MoteDefines.INTRASLOTORDER_ADMINTASKS,
        )

    # === misc

    def removeFutureEvent(self, uniqueTag):
        with self.dataLock:
            i = 0
            while i<len(self.events):
                if (self.events[i][3]==uniqueTag) and (self.events[i][0]!=self.asn):
                    self.events.pop(i)
                else:
                    i += 1

    def terminateSimulation(self,delay):
        with self.dataLock:
            self.asnEndExperiment = self.asn+delay
            self.scheduleAtAsn(
                    asn                = self.asn+delay,
                    cb                 = self._actionEndSim,
                    uniqueTag          = ('DiscreteEventEngine', '_actionEndSim'),
                    intraSlotOrder     = Mote.MoteDefines.INTRASLOTORDER_ADMINTASKS,
            )


    # === location update
    def logInitialLocation(self):
        #self.alg.last_obs = numpy.array(self.connectivity.coordinates.values())
        #self.socket.save_last_obs(self.connectivity.coordinates.values())
        print "logged"
        for motename in self.connectivity.coordinates:

       
            current_coords = self.connectivity.coordinates[motename]

            self.log(
                SimLog.LOG_LOCATION_UPDATE,
                {
                    '_mote_id' : motename,
                    'x': current_coords[0],
                    'y': current_coords[1],
                    'z':      0,
                }
            )


    def updateLocation(self):

        square_side        = self.settings.conn_random_square_side

        #print "location updated placeholder", self.asn
        #print self.connectivity.coordinates
        rewards = self.nextModelStep()

        
        for motename in self.connectivity.coordinates:
            #print motename
                    # determine coordinates of the motes
            '''
            current_coords = self.connectivity.coordinates[motename]
            # select a tentative coordinate
            
            if motename == 1 and self.asn>self.settings.drift_delay:
                self.connectivity.coordinates[motename] = (
                    current_coords[0] + self.settings.location_drift,
                    current_coords[1] 
                    #current_coords[0] + (square_side * random.random()-square_side/2)*self.settings.location_drift,
                    #current_coords[1] + (square_side * random.random()-square_side/2)*self.settings.location_drift
                )
            else:
                self.connectivity.coordinates[motename] = (
                    current_coords[0] + self.settings.location_drift*0,
                    current_coords[1] 
                    #current_coords[0] + (square_side * random.random()-square_side/2)*self.settings.location_drift,
                    #current_coords[1] + (square_side * random.random()-square_side/2)*self.settings.location_drift
                        )
            '''
            self.log(
                SimLog.LOG_LOCATION_UPDATE,
                {
                    '_mote_id' : motename,
                    'x': self.connectivity.coordinates[motename][0],
                    'y': self.connectivity.coordinates[motename][1],
                    'z':      0,
                    'reward' : rewards[motename]
                }
            )

        self.connectivity.update_connectivity_matrix()

        
        self.scheduleAtAsn(
            asn = self.asn + self.settings.location_update_period,
            cb = self.updateLocation,
            uniqueTag = ('LocationManager','InitialUpdate'),
            intraSlotOrder     = Mote.MoteDefines.INTRASLOTORDER_ADMINTASKS,

        )

    def nextModelStep(self):
        nextState = []
        num_drones = len(self.motes)
        rewards = {}

                #print self.c.hello("RPC")
        for mote in self.motes:

            current_coords = self.connectivity.coordinates[mote.id]
            self.drone_pos[mote.id][0] = current_coords[0]*1000
            self.drone_pos[mote.id][1] = current_coords[1]*1000

        for i in range(num_drones):
            rewards[self.get_mote_by_mote_id(i).id] = self.calcRewards(self.get_mote_by_mote_id(i),self.drone_pos[i]) #calc current rewards from last action

        if self.started:
            #store effect and update model from previous actions
            for i in range (1,num_drones):
                #print self.last_observations_native
                #print rewards
                self.socket.store_effect(self.last_observations_native[i],float(rewards[i]),self.curr_actions[i],float(self.done),i)
                self.socket.update_model(i)

        self.last_actions = self.curr_actions
        self.last_observations = numpy.reshape(numpy.array(self.connectivity.coordinates.values()),(len(self.motes)*2,))
        self.last_observations_native = self.connectivity.coordinates.values() #these become current observations now




        for i in range(num_drones):



            
            #print rewards
            if (i != 0):

                

                if(self.settings.control_mode == "all_motes"):
                    x_i = numpy.ones((num_drones - 1)) * self.drone_pos[i][0]
                    x_j = numpy.array([self.drone_pos[j][0] for j in range(num_drones) if j != i])
                    y_i = numpy.ones((num_drones - 1)) * self.drone_pos[i][1]
                    y_j = numpy.array([self.drone_pos[j][1] for j in range(num_drones) if j != i])
                    z_i = numpy.ones((num_drones - 1, 2)) * self.drone_pos[i]
                    z_j = numpy.array([self.drone_pos[j] for j in range(num_drones) if j != i])
                    norm = numpy.linalg.norm(z_i - z_j, axis=1)**2

                    R = self.settings.repulsion_constant # repulsion coefficient, EXPERIMENT
                    rep_x = - R*numpy.sum((x_i - x_j)/(norm**2))
                    rep_y = - R*numpy.sum((y_i - y_j)/(norm**2))

                    A = 0.001 # attraction coefficient, EXPERIMENT
                    atr_x = 4*A*A*numpy.sum(norm*(x_i - x_j)*numpy.exp(A*(norm)))
                    atr_y = 4*A*A*numpy.sum(norm*(y_i - y_j)*numpy.exp(A*(norm)))

                    # clip velocities before upating are somewhat arbitrary
                    self.drone_vels[i][0] = - numpy.clip(rep_x + atr_x, -20, 20)
                    self.drone_vels[i][1] = - numpy.clip(rep_y + atr_y, -20, 20)

                elif(self.settings.control_mode == "parent_drag"):
                    parent = self.get_mote_by_mote_id(i).rpl.of.get_preferred_parent()
                    if parent != None:
                        #print parent
                        topology = self.get_mote_by_mote_id(0).rpl.parentChildfromDAOs
                        #print topology
                        find_children = lambda d , p : [k for k in d.keys() if d.get(k) == p]

                        my_addr = netaddr.EUI(self.get_mote_by_mote_id(i).get_mac_addr())
                        prefix = netaddr.IPAddress('fd00::')
                        #print my_addr
                        #print str(my_addr.ipv6(prefix))

                        children = find_children(topology,str(my_addr.ipv6(prefix)))
                        #print children
                        parent_id=self.get_mote_by_mac_addr(parent).id
                        #print parent_id
                        x_i = numpy.ones((num_drones - 1)) * self.drone_pos[i][0]
                        x_j = numpy.array([self.drone_pos[j][0] for j in range(num_drones) if j != i])
                        y_i = numpy.ones((num_drones - 1)) * self.drone_pos[i][1]
                        y_j = numpy.array([self.drone_pos[j][1] for j in range(num_drones) if j != i])
                        z_i = numpy.ones((num_drones - 1, 2)) * self.drone_pos[i]
                        z_j = numpy.array([self.drone_pos[j] for j in range(num_drones) if j != i])
                        norm = numpy.linalg.norm(z_i - z_j, axis=1)**2

                        R = self.settings.repulsion_constant # repulsion coefficient, EXPERIMENT
                        rep_x = - R*numpy.sum((x_i - x_j)/(norm**2))
                        rep_y = - R*numpy.sum((y_i - y_j)/(norm**2))

                        parent_coords = self.drone_pos[parent_id]
                        parent_norm_2 = numpy.linalg.norm(numpy.array(parent_coords)-numpy.array(self.drone_pos[i]))**2



                        if(len(children)>0):


                            x_i = numpy.ones(len(children)) * self.drone_pos[i][0]

                            x_j = numpy.array([self.drone_pos[int(child.split(":")[-1],16)][0] for child in children])        

                            y_i = numpy.ones(len(children)) * self.drone_pos[i][1]

                            y_j = numpy.array([self.drone_pos[int(child.split(":")[-1],16)][1] for child in children])  

 


                            stacked = numpy.vstack([x_i,y_i])

                            stacked_j = numpy.vstack([x_j,y_j])
                            #print stacked
                            #print stacked_j
                            norm_child = numpy.linalg.norm(stacked-stacked_j, axis=0)**2
                            #print norm_child

                            A = .0001 # attraction coefficient, EXPERIMENT
                            atr_x_child = 4*A*A*numpy.sum(norm_child*(x_i - x_j)*numpy.exp(A*(norm_child)))/len(children)
                            atr_y_child = 4*A*A*numpy.sum(norm_child*(y_i - y_j)*numpy.exp(A*(norm_child)))/len(children)

                        else:
                            atr_x_child = 0
                            atr_y_child = 0 
                        #y_i = numpy.ones(len(children)) * self.drone_pos[i][1]

                        #norm = numpy.linalg.norm(z_i - z_j, axis=1)**2


                        A = .0001 # attraction coefficient, EXPERIMENT
                        atr_x = 4*A*A*parent_norm_2*(numpy.array(self.drone_pos[i][0])-numpy.array(parent_coords[0]))*numpy.exp(A*(parent_norm_2)) + atr_x_child
                        atr_y = 4*A*A*parent_norm_2*(numpy.array(self.drone_pos[i][1])-numpy.array(parent_coords[1]))*numpy.exp(A*(parent_norm_2)) + atr_y_child

                        #print atr_x - atr_x_child, atr_x
                        # clip velocities before upating are somewhat arbitrary
                        self.drone_vels[i][0] = - numpy.clip(rep_x + atr_x, -20, 20)
                        self.drone_vels[i][1] = - numpy.clip(rep_y + atr_y, -20, 20) 
                        if(i ==1 ):
                            self.drone_vels[i][0] += self.settings.constant_vel

                elif(self.settings.control_mode == "smart_choice"):
                    parent = self.get_mote_by_mote_id(i).rpl.of.get_preferred_parent()
                    neighbor_dict = self.get_mote_by_mote_id(i).rpl.of.neighbors
                    neighbors = []
                    for neighbor_entry in neighbor_dict:
                        #print neighbor_entry
                        mote_id = self.get_mote_by_mac_addr(neighbor_entry["mac_addr"]).id
                        #print mote_id
                        #print self.drone_pos[mote_id] 
                        neighbors.append([mote_id, self.drone_pos[mote_id][0], self.drone_pos[mote_id][1]])
                    

                    target_location = numpy.tile(self.settings.goal_loc,(len(neighbors),1))
            
                    #print numpy.array(neighbors)
                    if len(neighbors)>1:
                        #print target_location, "goal location"
                        distance = numpy.linalg.norm(numpy.array(neighbors)[:,1:] - target_location, axis=1)
                        closest_neighbor = neighbors[numpy.argmax(distance)][0]
                    elif len(neighbors)==1:
                        closest_neighbor = neighbors[0][0]
                    else:
                        parent = None

                    if parent != None:
                        #print parent
                        topology = self.get_mote_by_mote_id(0).rpl.parentChildfromDAOs
                        #print topology
                        find_children = lambda d , p : [k for k in d.keys() if d.get(k) == p]

                        my_addr = netaddr.EUI(self.get_mote_by_mote_id(i).get_mac_addr())
                        prefix = netaddr.IPAddress('fd00::')
                        #print my_addr
                        #print str(my_addr.ipv6(prefix))

                        children = find_children(topology,str(my_addr.ipv6(prefix)))
                        #print children
                        
                        #parent_id=self.get_mote_by_mac_addr(parent).id
                        parent_id = closest_neighbor
                        #print parent_id
                        x_i = numpy.ones((num_drones - 1)) * self.drone_pos[i][0]
                        x_j = numpy.array([self.drone_pos[j][0] for j in range(num_drones) if j != i])
                        y_i = numpy.ones((num_drones - 1)) * self.drone_pos[i][1]
                        y_j = numpy.array([self.drone_pos[j][1] for j in range(num_drones) if j != i])
                        z_i = numpy.ones((num_drones - 1, 2)) * self.drone_pos[i]
                        z_j = numpy.array([self.drone_pos[j] for j in range(num_drones) if j != i])
                        norm = numpy.linalg.norm(z_i - z_j, axis=1)**2

                        R = self.settings.repulsion_constant # repulsion coefficient, EXPERIMENT
                        rep_x = - R*numpy.sum((x_i - x_j)/(norm**2))
                        rep_y = - R*numpy.sum((y_i - y_j)/(norm**2))

                        parent_coords = self.drone_pos[parent_id]
                        parent_norm_2 = numpy.linalg.norm(numpy.array(parent_coords)-numpy.array(self.drone_pos[i]))**2



                        if(len(children)>0):


                            x_i = numpy.ones(len(children)) * self.drone_pos[i][0]

                            x_j = numpy.array([self.drone_pos[int(child.split(":")[-1],16)][0] for child in children])        

                            y_i = numpy.ones(len(children)) * self.drone_pos[i][1]

                            y_j = numpy.array([self.drone_pos[int(child.split(":")[-1],16)][1] for child in children])  

 


                            stacked = numpy.vstack([x_i,y_i])

                            stacked_j = numpy.vstack([x_j,y_j])
                            #print stacked
                            #print stacked_j
                            norm_child = numpy.linalg.norm(stacked-stacked_j, axis=0)**2
                            #print norm_child

                            A = .0001 # attraction coefficient, EXPERIMENT
                            atr_x_child = 4*A*A*numpy.sum(norm_child*(x_i - x_j)*numpy.exp(A*(norm_child)))/len(children)
                            atr_y_child = 4*A*A*numpy.sum(norm_child*(y_i - y_j)*numpy.exp(A*(norm_child)))/len(children)

                        else:
                            atr_x_child = 0
                            atr_y_child = 0 
                        #y_i = numpy.ones(len(children)) * self.drone_pos[i][1]

                        #norm = numpy.linalg.norm(z_i - z_j, axis=1)**2


                        A = .0001 # attraction coefficient, EXPERIMENT
                        atr_x = 4*A*A*parent_norm_2*(numpy.array(self.drone_pos[i][0])-numpy.array(parent_coords[0]))*numpy.exp(A*(parent_norm_2)) + atr_x_child
                        atr_y = 4*A*A*parent_norm_2*(numpy.array(self.drone_pos[i][1])-numpy.array(parent_coords[1]))*numpy.exp(A*(parent_norm_2)) + atr_y_child

                        #print atr_x - atr_x_child, atr_x
                        # clip velocities before upating are somewhat arbitrary
                        self.drone_vels[i][0] = - numpy.clip(rep_x + atr_x, -20, 20)
                        self.drone_vels[i][1] = - numpy.clip(rep_y + atr_y, -20, 20) 
                        if(i ==1 ):
                            self.drone_vels[i][0] += self.settings.constant_vel

                elif(self.settings.control_mode == "deep_rl"):
                    #feed observations into nn policy 
                    #print "hiiiii"
                    #store effects from last steps actions
                                #deepRL observations
                    if self.firststep and i!=0:
                        #self.alg.last_obs = numpy.reshape(numpy.array(self.connectivity.coordinates.values()),(len(self.motes)*2,))
                        #self.alg.last_obs = numpy.reshape(numpy.array(self.connectivity.coordinates.values()),(len(self.motes)*2,))[0:2]
                        self.socket.save_last_obs(self.connectivity.coordinates.values()[i],i)
                        

                    final_asn = self.settings.exec_numSlotframesPerRun * self.settings.tsch_slotframeLength
                    if(self.asn + self.settings.location_update_period > final_asn):
                        self.done = 1
                        print final_asn
                        print self.asn
                        print "done reached"
                        
                    else:
                        self.done = 0
                    #if self.started:
                        #print self.last_actions
                        #print "reward: ", rewards[i]

                        #self.alg.store_effect(last_observations[i*2:i*2+2],rewards[i],self.last_actions[i],done)

                        #self.alg.update_model()
                        

                    #print last_observations[i*2:i*2+2].tostring()
                    #actions = self.alg.step_env(last_observations[i*2:i*2+2])
                    #print last_observations_native

                    actions = self.socket.step_env(self.last_observations_native[i],i)
                    if self.done ==1:
                        self.socket.store_effect(self.last_observations_native[i],float(rewards[i]),self.curr_actions[i],float(self.done),i)
                    #print i,last_observations_native[i], actions 
                    #print last_observations[i*2:i*2+2], actions
                    self.curr_actions[self.get_mote_by_mote_id(i).id] = actions
                    #print actions 
                    action_list = [(0,0),
                                    (0,4),
                                    (4,0),
                                    (2,2),
                                    (0,-4),
                                    (-4,0),
                                    (-2,-2),
                                    (-2,2),
                                    (2,-2)
                                ]

                    self.drone_vels[i][0] = action_list[actions][0]
                    self.drone_vels[i][1] = action_list[actions][1]

            #dagroot is stationary
            elif (i == 0):
                self.drone_vels[i][0] = 0
                self.drone_vels[i][1] = 0


        self.firststep = False
        for mote in self.motes:
            current_coords = self.connectivity.coordinates[mote.id]
            # update position, divide by 1000 to convert to km
            self.connectivity.coordinates[mote.id] = (current_coords[0] + self.drone_vels[mote.id][0]*self.settings.location_update_period*self.settings.tsch_slotDuration/1000,
                                                        current_coords[1] + self.drone_vels[mote.id][1]*self.settings.location_update_period*self.settings.tsch_slotDuration/1000
                                                        )
            #self.drone_pos[i][0] += self.drone_vels[i][0]*self.settings.location_update_period*self.settings.tsch_slotDuration
            #self.drone_pos[i][1] += self.drone_vels[i][1]*self.settings.location_update_period*self.settings.tsch_slotDuration
        self.started = True
        return rewards
        #ani = animation.FuncAnimation(self.fig, animate, interval=1)

        #plt.plot()
        #plt.show()
        #plt.close()
        #sys.exit()

        #return self.drone_pos
 
    def calcRewards(self,mote,position):

        stats = mote.getRewards()
        mote.resetRewards()

        #print stats
        etx_array = numpy.array(list(stats['etxs'].values()))

        etx_avg = numpy.nanmean(etx_array)

        #print numpy.array(position)
        #print numpy.array(self.settings.goal_loc)
        d_goal = numpy.linalg.norm(numpy.array(position) - numpy.array(self.settings.goal_loc))**2 #goal in meters
        #print d_goal
        #print stats["packets_lost"]
        if math.isnan(etx_avg):
            etx_avg = 1
        #print etx_avg
        #print stats["packets_lost"]*10
        #print stats["rpl_churn"]*10

        return -(etx_avg-1)*0 -stats["packets_lost"]*0 - stats["rpl_churn"]*0 - d_goal/1000000



    # ======================== private ========================================

    def _actionPauseSim(self):
        assert self.simPaused==False
        self.simPaused = True
        self.pauseSem.acquire()

    def _actionResumeSim(self):
        if self.simPaused:
            self.simPaused = False
            self.pauseSem.release()

    def _actionEndSim(self):
        with self.dataLock:
            self.goOn = False

    def _actionEndSlotframe(self):
        """Called at each end of slotframe_iteration."""

        slotframe_iteration = int(self.asn / self.settings.tsch_slotframeLength)

        # print
        if self.verbose:
            print('   slotframe_iteration: {0}/{1}'.format(slotframe_iteration, self.settings.exec_numSlotframesPerRun-1))

        # schedule next statistics collection
        self.scheduleAtAsn(
            asn              = self.asn + self.settings.tsch_slotframeLength,
            cb               = self._actionEndSlotframe,
            uniqueTag        = ('DiscreteEventEngine', '_actionEndSlotframe'),
            intraSlotOrder   = Mote.MoteDefines.INTRASLOTORDER_ADMINTASKS,
        )
    
    # ======================== abstract =======================================
    
    def _init_additional_local_variables(self):
        pass
    
    def _routine_thread_started(self):
        pass
    
    def _routine_thread_crashed(self):
        pass
    
    def _routine_thread_ended(self):
        pass


class SimEngine(DiscreteEventEngine):
    
    DAGROOT_ID = 0
    
    def _init_additional_local_variables(self):
        self.settings                   = SimSettings.SimSettings()

        # set random seed
        if   self.settings.exec_randomSeed == 'random':
            self.random_seed = random.randint(0, sys.maxint)
        elif self.settings.exec_randomSeed == 'context':
            # with context for exec_randomSeed, an MD5 value of
            # 'startTime-hostname-run_id' is used for a random seed
            startTime = SimConfig.SimConfig.get_startTime()
            if startTime is None:
                startTime = time.time()
            context = (platform.uname()[1], str(startTime), str(self.run_id))
            md5 = hashlib.md5()
            md5.update('-'.join(context))
            self.random_seed = int(md5.hexdigest(), 16) % sys.maxint
        else:
            assert isinstance(self.settings.exec_randomSeed, int)
            self.random_seed = self.settings.exec_randomSeed
        # apply the random seed; log the seed after self.log is initialized
        random.seed(a=self.random_seed)

        self.motes                      = [Mote.Mote.Mote(m) for m in range(self.settings.exec_numMotes)]
        self.connectivity               = Connectivity.Connectivity()
        self.log                        = SimLog.SimLog().log
        SimLog.SimLog().set_simengine(self)

        # log the random seed
        self.log(
            SimLog.LOG_SIMULATOR_RANDOM_SEED,
            {
                'value': self.random_seed
            }
        )
        
        # select dagRoot
        self.motes[self.DAGROOT_ID].setDagRoot()

        # boot all motes
        for i in range(len(self.motes)):
            self.motes[i].boot()
    
        #initialize animation
        self.size =  (1000, 300)
        #style.use('fivethirtyeight')

        self.fig, (ax1, ax2, ax3) = plt.subplots(3, 1, gridspec_kw = {'height_ratios':[2, 1, 1]})
        self.fig.tight_layout()

        self.drones = []
        self.drone_rects = []
        self.drone_pos = {}
        self.drone_vels = []
        start_sep = 5
        (w, h) = (self.size[0]/2 , self.size[1]/2)
        for mote in self.motes:

            self.drone_pos[mote.id] = [w, h + start_sep*numpy.floor((mote.id+1)/2)*(-1)**(i+1)]
            if (mote.id == 1):
                self.drone_vels.append([0.0, 0.0])
            else:
                self.drone_vels.append([0.0, 0.0])

        
        self.goal_loc = self.settings.goal_loc
        self.last_actions ={}
        self.curr_actions={}
        self.started = False
        print "engine restarted"







    def _routine_thread_started(self):
        # log
        self.log(
            SimLog.LOG_SIMULATOR_STATE,
            {
                "name":   self.name,
                "state":  "started"
            }
        )

        ########start tensorflow#############
      
        ########tf code done##########################

        #schedule location update
        if self.settings.location_update:
            self.scheduleAtAsn(
                asn = self.settings.location_update_period,
                cb = self.updateLocation,
                uniqueTag = ('LocationManager','InitialUpdate'),
                intraSlotOrder     = Mote.MoteDefines.INTRASLOTORDER_ADMINTASKS,

            )
        #otherwise just log initial locations
        else:
            self.scheduleAtAsn(
                asn = 1,
                cb = self.logInitialLocation,
                uniqueTag = ('LocationManager','InitialLocation'),
                intraSlotOrder     = Mote.MoteDefines.INTRASLOTORDER_ADMINTASKS,

            )
        # schedule end of simulation
        self.scheduleAtAsn(
            asn              = self.settings.tsch_slotframeLength*self.settings.exec_numSlotframesPerRun,
            cb               = self._actionEndSim,
            uniqueTag        = ('SimEngine','_actionEndSim'),
            intraSlotOrder   = Mote.MoteDefines.INTRASLOTORDER_ADMINTASKS,
        )

        # schedule action at every end of slotframe_iteration
        self.scheduleAtAsn(
            asn              = self.asn + self.settings.tsch_slotframeLength - 1,
            cb               = self._actionEndSlotframe,
            uniqueTag        = ('SimEngine', '_actionEndSlotframe'),
            intraSlotOrder   = Mote.MoteDefines.INTRASLOTORDER_ADMINTASKS,
        )

    def _routine_thread_crashed(self):
        # log
        self.log(
            SimLog.LOG_SIMULATOR_STATE,
            {
                "name": self.name,
                "state": "crash"
            }
        )

    def _routine_thread_ended(self):
        # log
        self.log(
            SimLog.LOG_SIMULATOR_STATE,
            {
                "name": self.name,
                "state": "stopped"
            }
        )
