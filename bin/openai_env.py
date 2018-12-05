import sys
sys.path.insert(0, '/../multiagent-particle-envs-master')

import make_env
import RLServer_open_env


class Environment(object):





	def __init__(self,scenario):
		self.env =make_env(scenario)
		#reset world
		reset_world()

	def step_env(self,action):

		return	self.env.step(action)

