#this program starts an rpc server 
from SimpleXMLRPCServer import SimpleXMLRPCServer
from SimpleXMLRPCServer import SimpleXMLRPCRequestHandler

import os
import platform
import sys

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

import dqn
import dqn_utils
from dqn_utils import *

# Restrict to a particular path.
class RequestHandler(SimpleXMLRPCRequestHandler):
	rpc_paths = ('/RPC2',)

# Create server
server = SimpleXMLRPCServer(("localhost", 8001),
							requestHandler=RequestHandler)
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
simconfig = SimConfig.SimConfig(configdata=config_data)




########start tensorflow#############
tf.reset_default_graph()
tf_config = tf.ConfigProto(
inter_op_parallelism_threads=8,
intra_op_parallelism_threads=8)
session = tf.Session(config=tf_config)
#print("AVAILABLE GPUS: ", get_available_gpus())



num_timesteps = 1000* 13 *simconfig.execution.numRuns  #rough, todo make real slotframes

num_iterations = float(num_timesteps) / 10

lr_multiplier = 1
lr_schedule = dqn_utils.PiecewiseSchedule([
									 (0,                   1e-4 * lr_multiplier),
									 (num_iterations / 10, 1e-4 * lr_multiplier),
									 (num_iterations / 2,  5e-5 * lr_multiplier),
								],
								outside_value=5e-5 * lr_multiplier)

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
		(num_iterations/5, 0.1),
		(num_iterations / 2, 0.01),
	], outside_value=0.01
)

alg = dqn.QLearner(env=None,
				q_func=lander_model,
				optimizer_spec=optimizer,
				session=session,
				exploration=exploration_schedule,
				stopping_criterion=stopping_criterion,
				replay_buffer_size=1000000,
				batch_size=32,
				gamma=0.99,
				learning_starts=simconfig.settings.regular.steps_to_train,
				learning_freq=4,
				frame_history_len=4,
				target_update_freq=10000,
				grad_norm_clipping=10,
				double_q=True)

# Register an instance; all the methods of the instance are
# published as XML-RPC methods (in this case, just 'div').

server.register_instance(MyFuncs())

# Run the server's main loop
server.serve_forever()