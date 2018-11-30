#this program starts an rpc server 
from SimpleXMLRPCServer import SimpleXMLRPCServer
from SimpleXMLRPCServer import SimpleXMLRPCRequestHandler

import os
import platform
import sys
import argparse
import numpy
import zerorpc


if __name__ == '__main__':
	here = sys.path[0]
	sys.path.insert(0, os.path.join(here, '..'))

from SimEngine import SimConfig,   \
					  SimEngine,   \
					  SimLog, \
					  SimSettings, \
					  Connectivity, \
					  dqn, \
					  dqn_utils

import tensorflow as tf
import tensorflow.contrib.layers as layers



# Restrict to a particular path.
class RequestHandler(SimpleXMLRPCRequestHandler):
	rpc_paths = ('/RPC2',)


# Create server
server = SimpleXMLRPCServer(("127.0.0.1", 8001),
							requestHandler=RequestHandler,
							logRequests=False,
							allow_none = True)
server.register_introspection_functions()

####get config################
def parseCliParams():

	parser = argparse.ArgumentParser()

	parser.add_argument(
		'--config',
		dest       = 'config',
		action     = 'store',
		default    = 'config.json',
		help       = 'Location of the configuration file.',
	)
	cliparams      = parser.parse_args()
	return cliparams.__dict__

cliparams = parseCliParams()
simconfig = SimConfig.SimConfig(cliparams['config'])




########start tensorflow#############
tf.reset_default_graph()
tf_config = tf.ConfigProto(
inter_op_parallelism_threads=8,
intra_op_parallelism_threads=8)
session = tf.Session(config=tf_config)
#print("AVAILABLE GPUS: ", get_available_gpus())



num_timesteps = simconfig.settings.regular.exec_numSlotframesPerRun * simconfig.settings.combination.tsch_slotframeLength[0] *simconfig.execution.numRuns  #rough, todo make real slotframes
print num_timesteps
num_iterations = float(num_timesteps) /simconfig.settings.regular.location_update_period

#lr_multiplier = 10.0
lr_multiplier = simconfig.settings.regular.lr_multiplier
lr_schedule = dqn_utils.PiecewiseSchedule([
									 (0,                   1e-4 * lr_multiplier),
									 (num_iterations / 4, 1e-4 * lr_multiplier),
									 (num_iterations ,  1e-5 * lr_multiplier),
								],
								outside_value=1e-5 * lr_multiplier)

optimizer = dqn.OptimizerSpec(
	constructor=tf.train.AdamOptimizer,
	kwargs=dict(epsilon=1e-4),
	lr_schedule=lr_schedule
)
def stopping_criterion(env, t):
	# notice that here t is the number of steps of the wrapped env,
	# which is different from the number of steps in the underlying env
	return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

def lander_model(obs, num_actions, scope, reuse=False):
	with tf.variable_scope(scope, reuse=reuse):
		out = obs
		with tf.variable_scope("action_value"):
			out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
			out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
			out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

		return out

exploration_schedule = dqn_utils.PiecewiseSchedule(
	[
		(0, 1.0),
		(num_iterations/5, 0.2),
		(num_iterations/2 , 0.05),
	], outside_value=0.01
)
algs = {}
#spawn multiple q functions
for i in range(1,simconfig.settings.combination.exec_numMotes[0]):
	algs[i] = dqn.QLearner(env=None,
				q_func=lander_model,
				optimizer_spec=optimizer,
				session=session,
				exploration=exploration_schedule,
				stopping_criterion=stopping_criterion,
				replay_buffer_size=100000,
				batch_size=32,
				gamma=0.99,
				learning_starts=simconfig.settings.regular.steps_to_train,
				learning_freq=4,
				frame_history_len=4,
				target_update_freq=10000,
				grad_norm_clipping=simconfig.settings.regular.grad_clipping,
				double_q=True,
				num_motes = simconfig.settings.combination.exec_numMotes[0]-1,
				initial_state = {},
				agent = i,
				observation_dim = 2+ simconfig.settings.combination.exec_numMotes[0]-1)

# Register an instance; all the methods of the instance are
# published as XML-RPC methods (in this case, just 'div').
class MyFuncs:
	def save_last_obs(self,last_obs,agent):
		#print('last obs from rpc',last_obs)
		last_obs = numpy.array(last_obs)
		#print "agent:" ,agent
		algs[agent].last_obs[agent] = last_obs
		#print algs[agent].last_obs[agent] 
		#print('saving last obs',last_obs)
	def indicate_current_steps(self):
		pass
	def store_effect(self,last_observations,rewards,last_actions,done,agent):
		#print "store effect"
		#print "rewards, actions ", rewards, last_actions,
		algs[agent].store_effect(last_observations,rewards,numpy.array(last_actions),done,agent)
	def update_model(self,agent):
		algs[agent].update_model(agent)
	def step_env(self,last_observations,agent):
		#print('rpc',last_observations)
		#print "step_env"
		#print algs
		#print "stored last obs",numpy.array(last_observations)
		actions = algs[agent].step_env(numpy.array(last_observations),agent)
		return int(actions)
	def exit(self):
		pass

server.register_instance(MyFuncs())





# Run the server's main loop
server.serve_forever()
