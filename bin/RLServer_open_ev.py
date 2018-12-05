#this program starts an rpc server 


import os
import platform
import sys
import argparse
import numpy
import datetime
import uuid

here = sys.path[0]
sys.path.insert(0, os.path.join(here, '..'))

from SimEngine import dqn, \
					  dqn_utils
					  

import tensorflow as tf
import tensorflow.contrib.layers as layers
import make_env
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd

####get config################
def parseCliParams():

	parser = argparse.ArgumentParser()

	parser.add_argument('--batch_size',type = int,default = 32)
	parser.add_argument('--lr_mult',type = float,default = 1.0)
	parser.add_argument('--clip',type = int, default = 10)
	cliparams      = parser.parse_args()
	return cliparams.__dict__

cliparams = parseCliParams()

print(cliparams)


########start tensorflow#############
tf.reset_default_graph()
tf_config = tf.ConfigProto(
inter_op_parallelism_threads=8,
intra_op_parallelism_threads=8)
session = tf.Session(config=tf_config)
#print("AVAILABLE GPUS: ", get_available_gpus())



environment = "simple_spread"
if environment == "simple_spread":
	num_agents = 3
	obs_dim = 18
elif environment == "simple":
	num_agents = 1
	obs_dim = 4

num_timesteps = 1000000  #rough, todo make real slotframes
print(num_timesteps)
num_iterations = float(num_timesteps)

#lr_multiplier = 10.0
lr_multiplier = cliparams["lr_mult"]
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
		(0, 0.2),
		(num_iterations/2, 0.1),
		(num_iterations, 0.05),
	], outside_value=0.01
)
algs = {}
#spawn multiple q functions
for i in range(0,num_agents):
	algs[i] = dqn.QLearner(env=None,
				q_func=lander_model,
				optimizer_spec=optimizer,
				session=session,
				exploration=exploration_schedule,
				stopping_criterion=stopping_criterion,
				replay_buffer_size=1000000,
				batch_size=cliparams["batch_size"],
				gamma=0.98,
				learning_starts=10000,
				learning_freq=4,
				frame_history_len=4,
				target_update_freq=10000,
				grad_norm_clipping=cliparams["clip"],
				double_q=True,
				num_motes = 3,
				initial_state = {},
				agent = i,
				observation_dim = obs_dim,
				num_actions = 5)

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
	def log_rewards(self, rewards):
		print (rewards)

	def exit(self):
		pass

env = make_env.make_env(environment)

#reset env
obs_n = numpy.array(env.reset())
agent = 0
episode_length = 200
time = 0
episode_num = 0
action_n=[]


print("obs shape after reset", obs_n.shape)
for agent in range(num_agents):

	algs[agent].last_obs[agent] = obs_n[agent]

print (env.action_space)
action_dims = env.action_space[0].n
for i in range(num_agents):
	action_n.append(numpy.zeros(action_dims))
finished = False
#print(action_dims)
rewards_n = None
episode_rewards = []
while not finished:
	episode_reward = []
	render = False
	
	for i in range(episode_length):
		for agent in range(num_agents):
			#action = MyFuncs.step_env(obs,agent) 	# get action from dqn
			action_n[agent] = numpy.zeros(5)
		#	print("")
			
			action = algs[agent].step_env(obs_n[agent],agent)
		#	print("observation before action deciede", obs_n,rewards_n, action)
			action_n[agent][int(action)] = 1

		#convert discrete actions to

		obs_n, rewards_n, done_n, info_n = env.step(action_n) # get observations and rewards from next step of simulator 
		#print("observation rewards after environment step ",obs_n, rewards_n)
		#print("")
		if time > 10000 and episode_num % 100 == 0:
			render = True
		#render = True
		if time % 10000 == 0:
			print(" ")			
			print("Last 10 episode_rewards", numpy.mean(episode_rewards[-100:]))
			print(obs_n)
			print("")
		#print("obs shape after step env", obs_n)
		#MyFuncs.store_effect(obs_n, rewards_n, action, done_n,agent)
		time += 1
		if i == episode_length-1:
			done_n = True
			#print("done")
		for agent in range(num_agents):
			algs[agent].store_effect(obs_n[agent], rewards_n[agent],numpy.array(action),done_n,agent)

			algs[agent].update_model(agent)

		if render:
			env.render()
		
		episode_reward.append(rewards_n[agent])
	
	episode_num += 1
	episode_rewards.append(numpy.sum(episode_reward))
	obs_n = numpy.array(env.reset())
	if time >= num_timesteps:
		finished = True

print(episode_rewards)	

data = pd.DataFrame(episode_rewards)
data.to_csv("gymSim/"+str(uuid.uuid4())+"_b" + str(cliparams["batch_size"])+"_lr" +str(cliparams["lr_mult"]) +"_clip"+str(cliparams["clip"])+ ".csv")	
#plt.plot(numpy.array(episode_rewards))
#plt.plot([0,1,2,3,4,5])









