"""
"""

# =========================== imports =========================================

import random
import math

# Mote sub-modules

# Simulator-wide modules
import SimEngine
import MoteDefines as d

# =========================== defines =========================================

# =========================== helpers =========================================

# =========================== body ============================================

class Rpl(object):

    def __init__(self, mote):

        # store params
        self.mote                      = mote

        # singletons (quicker access, instead of recreating every time)
        self.engine                    = SimEngine.SimEngine.SimEngine()
        self.settings                  = SimEngine.SimSettings.SimSettings()
        self.log                       = SimEngine.SimLog.SimLog().log

        # local variables
        self.rank                      = None
        self.preferredParent           = None
        self.parentChildfromDAOs       = {}      # dictionary containing parents of each node

    #======================== public ==========================================

    # getters/setters

    def setRank(self, newVal):
        self.rank = newVal
    def getRank(self):
        return self.rank
    def getDagRank(self):
        return int(self.rank/d.RPL_MINHOPRANKINCREASE)
    
    def addParentChildfromDAOs(self, parent_id, child_id):
        assert type(parent_id)==int
        assert type(child_id) ==int
        self.parentChildfromDAOs[child_id] = parent_id

    def getPreferredParent(self):
        return self.preferredParent
    def setPreferredParent(self, newVal):
        assert type(newVal)==int
        self.preferredParent = newVal

    # admin

    def activate(self):
        """
        Initialize the RPL layer
        """

        # start sending DAOs
        self._schedule_sendDAO(firstDAO=True)
    
    # === DIO
    
    def _create_DIO(self):
        
        # create
        newDIO = {
            'type':          d.PKT_TYPE_DIO,
            'app': {
                'rank':      self.rank,
            },
            'net': {
                'srcIp':     self.mote.id,            # from mote
                'dstIp':     d.BROADCAST_ADDRESS,     # broadcast (in reality "all RPL routers")
            },
            'mac': {
                'srcMac':    self.mote.id,            # from mote
                'dstMac':    d.BROADCAST_ADDRESS,     # broadcast
            }
        }
        
        # log
        self.log(
            SimEngine.SimLog.LOG_RPL_DIO_TX,
            {
                "_mote_id":  self.mote.id,
                "packet":    newDIO,
            }
        )
        
        return newDIO
    
    def action_receiveDIO(self, packet):
        
        assert packet['type'] == d.PKT_TYPE_DIO
        
        # abort if I'm the DAGroot
        if self.mote.dagRoot:
            return

        # abort if I'm not sync'ed
        if not self.mote.tsch.getIsSync():
            return

        # log
        self.log(
            SimEngine.SimLog.LOG_RPL_DIO_RX,
            {
                "_mote_id":  self.mote.id,
                "packet":    packet,
            }
        )
        
        # update rank with sender's information
        self.mote.neighbors[packet['mac']['srcMac']]['rank']  = packet['app']['rank']

        # trigger RPL housekeeping
        self._updateMyRankAndPreferredParent()

    # === DAO
    
    def _schedule_sendDAO(self, firstDAO=False):
        """
        Schedule to send a DAO sometimes in the future.
        """
        
        # abort it I'm the root
        if self.mote.dagRoot:
            return

        # abort if DAO disabled
        if self.settings.rpl_daoPeriod == 0:
            return
        
        asnNow = self.engine.getAsn()

        if firstDAO:
            asnDiff = 1
        else:
            asnDiff = int(math.ceil(
                random.uniform(
                    0.8 * self.settings.rpl_daoPeriod,
                    1.2 * self.settings.rpl_daoPeriod
                ) / self.settings.tsch_slotDuration)
            )

        # schedule sending a DAO
        self.engine.scheduleAtAsn(
            asn              = asnNow + asnDiff,
            cb               = self._action_sendDAO,
            uniqueTag        = (self.mote.id, '_action_sendDAO'),
            intraSlotOrder   = d.INTRASLOTORDER_STACKTASKS,
        )

    def _action_sendDAO(self):
        """
        Enqueue a DAO and schedule next one.
        """
        
        # enqueue
        self._action_enqueueDAO()

        # schedule next DAO
        self._schedule_sendDAO()

    def _action_enqueueDAO(self):
        """
        enqueue a DAO into TSCH queue
        """

        assert not self.mote.dagRoot
        
        # abort if not ready yet
        if self.mote.clear_to_send_EBs_DIOs_DATA()==False:
            return
        
        # create
        newDAO = {
            'type':                d.PKT_TYPE_DAO,
            'app': {
                'child_id':        self.mote.id,
                'parent_id':       self.preferredParent,
            },
            'net': {
                'srcIp':           self.mote.id,            # from mote
                'dstIp':           self.mote.dagRootId,     # to DAGroot
                'packet_length':   d.PKT_LEN_DAO,
            },
        }
        
        # log
        self.log(
            SimEngine.SimLog.LOG_RPL_DAO_TX,
            {
                "_mote_id": self.mote.id,
                "packet":   newDAO,
            }
        )
        
        # remove other possible DAOs from the queue
        self.mote.tsch.removeTypeFromQueue(d.PKT_TYPE_DAO)
        
        # send
        self.mote.sixlowpan.sendPacket(newDAO)
    
    def action_receiveDAO(self, packet):
        """
        DAGroot receives DAO, store parent/child relationship for source route calculation.
        """

        assert self.mote.dagRoot
        
        # log
        self.log(
            SimEngine.SimLog.LOG_RPL_DAO_RX,
            {
                "_mote_id": self.mote.id,
                "packet":   packet,
            }
        )

        # store parent/child relationship for source route calculation
        self.addParentChildfromDAOs(
            parent_id   = packet['app']['parent_id'],
            child_id    = packet['app']['child_id'],
        )

    # source route

    def computeSourceRoute(self, dest_id):
        """
        Compute the source route to a given mote.

        :param destAddr: [in] The EUI64 address of the final destination.

        :returns: The source route, a list of EUI64 address, ordered from
            destination to source, or None
        """
        assert type(dest_id)==int
        
        try:
            sourceRoute = []
            cur_id = dest_id
            while cur_id!=0:
                sourceRoute += [cur_id]
                cur_id       = self.parentChildfromDAOs[cur_id]
        except KeyError:
            returnVal = None
        else:
            # reverse (so goes from source to destination)
            sourceRoute.reverse()
            
            returnVal = sourceRoute
            
        return returnVal

    # forwarding

    def findNextHopId(self, packet):
        assert packet['net']['dstIp'] != self.mote.id
        
        if    packet['net']['dstIp'] == d.BROADCAST_ADDRESS:
            # broadcast packet
            
            # next hop is broadcast address
            nextHopId = d.BROADCAST_ADDRESS
        
        elif 'sourceRoute' in packet['net']:
            # unicast source routed downstream packet
            
            # next hop is the first item in the source route 
            nextHopId = self.engine.motes[packet['net']['sourceRoute'].pop(0)].id
            
        else:
            # unicast upstream packet
            
            if   self.mote.isNeighbor(packet['net']['dstIp']):
                # packet to a neighbor
                
                # next hop is that neighbor
                nextHopId = packet['net']['dstIp']
            elif packet['net']['dstIp'] == self.mote.dagRootId:
                # common upstream packet
                
                # next hop is preferred parent (returns None if no preferred parent)
                nextHopId = self.preferredParent
            else:
                raise SystemError()
        
        return nextHopId

    #======================== private ==========================================
    
    # misc

    def _updateMyRankAndPreferredParent(self):
        """
        RPL housekeeping tasks.

        This routine refreshes
        - self.rank
        - self.preferredParent
        """

        # calculate the rank I would have if choosing each of my neighbor as my preferred parent
        allPotentialRanks = {}
        for (nid,n) in self.mote.neighbors.items():
            if n['rank']==None:
                # I haven't received a DIO from that neighbor yet, so I don't konw its rank (normal)
                continue
            etx                        = self._estimateETX(nid)
            rank_increment             = (1*((3*etx)-2) + 0) * d.RPL_MINHOPRANKINCREASE # https://tools.ietf.org/html/rfc8180#section-5.1.1
            allPotentialRanks[nid]     = n['rank']+rank_increment
        
        # pick lowest potential rank
        (myPotentialParent,myPotentialRank) = sorted(allPotentialRanks.iteritems(), key=lambda x: x[1])[0]
        
        # switch parents
        if self.rank!=myPotentialRank or self.preferredParent!=myPotentialParent:
            
            # update
            self.rank            = myPotentialRank
            self.preferredParent = myPotentialParent
            
            # log
            self.log(
                SimEngine.SimLog.LOG_RPL_CHURN,
                {
                    "_mote_id":        self.mote.id,
                    "rank":            self.rank,
                    "preferredParent": self.preferredParent,
                }
            )
    
    def _estimateETX(self, neighbor_id):
        
        assert type(neighbor_id)==int
        
        # set initial values for numTx and numTxAck assuming PDR is exactly estimated
        # FIXME
        pdr                   = self.mote.getPDR(neighbor_id)
        numTx                 = d.NUM_SUFFICIENT_TX
        numTxAck              = math.floor(pdr*numTx)
        
        for (_, cell) in self.mote.tsch.getSchedule().items():
            if  (
                    (cell['neighbor'] == neighbor_id)
                    and
                    (d.CELLOPTION_TX in cell['cellOptions'])
                ):
                numTx        += cell['numTx']
                numTxAck     += cell['numTxAck']
        
        # abort if about to divide by 0
        if not numTxAck:
            return

        # calculate ETX
        etx = float(numTx)/float(numTxAck)

        return etx
