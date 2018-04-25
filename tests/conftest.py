import os
import pytest
import time

from SimEngine import SimConfig,   \
                      SimSettings, \
                      SimLog,      \
                      SimEngine
import SimEngine.Mote.MoteDefines as d

ROOT_DIR         = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
CONFIG_FILE_PATH = os.path.join(ROOT_DIR, 'bin/config.json')

def pdr_not_null(c,p,engine):
    returnVal = False
    for channel in range(engine.settings.phy_numChans):
        if engine.connectivity.get_pdr(c,p,channel)>0:
            returnVal = True
    return returnVal

@pytest.fixture(scope="function")
def sim_engine(request):

    def create_sim_engine(diff_config={}, force_initial_routing_and_scheduling_state=None):

        # get default configuration
        sim_config = SimConfig.SimConfig(CONFIG_FILE_PATH)
        config = sim_config.settings['regular']
        assert 'exec_numMotes' not in config
        config['exec_numMotes'] = sim_config.settings['combination']['exec_numMotes'][0]

        # update default configuration with parameters
        config.update(**diff_config)

        # create sim settings
        sim_settings = SimSettings.SimSettings(**config)
        sim_settings.setStartTime(time.strftime('%Y%m%d-%H%M%S'))
        sim_settings.setCombinationKeys([])

        # create sim log
        sim_log = SimEngine.SimLog.SimLog()
        sim_log.set_log_filters('all') # do not log

        # create sim engine
        engine = SimEngine.SimEngine()

        # force initial routing and schedule, if appropriate
        if force_initial_routing_and_scheduling_state:
            hoge(engine)

        # add a finalizer
        def fin():
            try:
                need_terminate_sim_engine_thread = engine.is_alive()
            except AssertionError:
                # engine thread is not initialized for some reason
                need_terminate_sim_engine_thread = False

            if need_terminate_sim_engine_thread:
                if engine.simPaused:
                    # if the thread is paused, resume it so that an event for
                    # termination is scheduled; otherwise deadlock happens
                    engine.play()
                engine.terminateSimulation(1)
                engine.join()

            engine.destroy()
            sim_settings.destroy()
            sim_log.destroy()

        request.addfinalizer(fin)

        return engine

    return create_sim_engine


def hoge(engine):

    # root is mote
    root = engine.motes[0]

    # start scheduling from slot offset 1 upwards
    cur_slot = 1

    # list all motes, indicate state as 'unseen' for all
    state = dict(zip(engine.motes, ['unseen']*len(engine.motes)))

    # start by having the root as 'active' mote
    state[root] = 'active'

    # loop over the motes, until all are 'seen'
    while state.values().count('seen')<len(state):

        # find an active mote, this is the 'parent' in this iteration
        parent = None
        for (k,v) in state.items():
            if v == 'active':
                parent = k
                break
        assert parent

        # for each of its children, set initial routing state and schedule
        for child in state.keys():
            if child == parent:
                continue
            if state[child] != 'unseen':
                continue
            if pdr_not_null(child,parent,engine):
                # there is a non-zero PDR on the child->parent link

                # set child's preferredparent to parent
                child.rpl.setPreferredParent(parent)
                # record the child->parent relationship at the root (for source routing)
                root.rpl.updateDaoParents({child:parent})
                # add a cell from child to parent
                child.tsch.addCells(parent,[(cur_slot,0,d.DIR_TX)])
                child.numCellsToNeighbors[parent] = 1
                parent.tsch.addCells(child,[(cur_slot,0,d.DIR_RX)])
                parent.numCellsFromNeighbors[child] = 1
                cur_slot += 1
                # mark child as active
                state[child]  = 'active'

        # mark parent as seen
        state[parent] = 'seen'
