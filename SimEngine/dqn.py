import uuid
import time
import pickle
import sys
import itertools
import numpy as np
import random
import tensorflow                as tf
import tensorflow.contrib.layers as layers
from collections import namedtuple
from dqn_utils import *

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])

class QLearner(object):

  def __init__(
    self,
    env,
    q_func,
    optimizer_spec,
    session,
    exploration=LinearSchedule(1000000, 0.1),
    stopping_criterion=None,
    replay_buffer_size=1000000,
    batch_size=32,
    gamma=0.99,
    learning_starts=50000,
    learning_freq=4,
    frame_history_len=4,
    target_update_freq=10000,
    grad_norm_clipping=10,
    rew_file=None,
    double_q=True,
    lander=False,
    initial_state=None,
    observation_dim = None,
    num_motes = None,
    agent = None):
    """Run Deep Q-learning algorithm.

    You can specify your own convnet using q_func.

    All schedules are w.r.t. total number of steps taken in the environment.

    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            img_in: tf.Tensor
                tensorflow tensor representing the input image
            num_actions: int
                number of actions
            scope: str
                scope in which all the model related variables
                should be created
            reuse: bool
                whether previously created variables should be reused.
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    session: tf.Session
        tensorflow session to use.
    exploration: rl_algs.deepq.utils.schedules.Schedule
        schedule for probability of chosing random action.
    stopping_criterion: (env, t) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    grad_norm_clipping: float or None
        If not None gradients' norms are clipped to this value.
    double_q: bool
        If True, then use double Q-learning to compute target values. Otherwise, use vanilla DQN.
        https://papers.nips.cc/paper/3964-double-q-learning.pdf
    """


    self.target_update_freq = target_update_freq
    self.optimizer_spec = optimizer_spec
    self.batch_size = batch_size
    self.learning_freq = learning_freq
    self.learning_starts = learning_starts
    self.stopping_criterion = stopping_criterion
    self.env = env
    self.session = session
    self.exploration = exploration
    self.rew_file = str(uuid.uuid4()) + '.pkl' if rew_file is None else rew_file
    self.agent = agent
    ###############
    # BUILD MODEL #
    ###############

    input_shape = (2, )
    self.num_actions = 9
    self.curr_reward = 0
    # set up placeholders
    # placeholder for current observation (or state)
    self.obs_t_ph              = tf.placeholder(
        tf.float32 if lander else tf.uint8, [None] + list(input_shape))
    # placeholder for current action
    self.act_t_ph              = tf.placeholder(tf.int32,   [None])
    # placeholder for current reward
    self.rew_t_ph              = tf.placeholder(tf.float32, [None])
    # placeholder for next observation (or state)
    self.obs_tp1_ph            = tf.placeholder(
        tf.float32 if lander else tf.uint8, [None] + list(input_shape))
    # placeholder for end of episode mask
    # this value is 1 if the next state corresponds to the end of an episode,
    # in which case there is no Q-value at the next state; at the end of an
    # episode, only the current state reward contributes to the target, not the
    # next state Q-value (i.e. target is just rew_t_ph, not rew_t_ph + gamma * q_tp1)
    self.done_mask_ph          = tf.placeholder(tf.float32, [None])

    # casting to float on GPU ensures lower data transfer times.
    if lander:
      obs_t_float = self.obs_t_ph
      obs_tp1_float = self.obs_tp1_ph
    else:
      obs_t_float   = tf.cast(self.obs_t_ph,   tf.float32) / 255.0
      obs_tp1_float = tf.cast(self.obs_tp1_ph, tf.float32) / 255.0

    # Here, you should fill in your own code to compute the Bellman error. This requires
    # evaluating the current and next Q-values and constructing the corresponding error.
    # TensorFlow will differentiate this error for you, you just need to pass it to the
    # optimizer. See assignment text for details.
    # Your code should produce one scalar-valued tensor: total_error
    # This will be passed to the optimizer in the provided code below.
    # Your code should also produce two collections of variables:
    # q_func_vars
    # target_q_func_vars
    # These should hold all of the variables of the Q-function network and target network,
    # respectively. A convenient way to get these is to make use of TF's "scope" feature.
    # For example, you can create your Q-function network with the scope "q_func" like this:
    # <something> = q_func(obs_t_float, num_actions, scope="q_func", reuse=False)
    # And then you can obtain the variables like this:
    # q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')
    # Older versions of TensorFlow may require using "VARIABLES" instead of "GLOBAL_VARIABLES"
    # Tip: use huber_loss (from dqn_utils) instead of squared error when defining self.total_error
    ######

    # YOUR CODE HERE

    self.q_vals = q_func(obs_t_float,self.num_actions,str(self.agent)+"/q_func") #predicts actions from observations
    q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=str(self.agent)+"/q_func") # collect parameters
    print q_func_vars

    #create model for target network
    self.targets = q_func(obs_tp1_float,self.num_actions,str(self.agent)+"/target_q_func")
    target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=str(self.agent)+'/target_q_func') # collect parameters


    self.max_targets = tf.reduce_max(self.targets,axis=1) # for debugging 
    if not double_q:
      self.yi = self.rew_t_ph + gamma*tf.reduce_max(self.targets,axis=1)*(1-self.done_mask_ph)
    else:
      self.dbl_targets = q_func(obs_tp1_float,self.num_actions,str(self.agent)+"/q_func",reuse=True)
      self.dbl_one_hot = tf.one_hot(tf.argmax(self.dbl_targets,axis=1),self.num_actions)

      self.yi = self.rew_t_ph + gamma*tf.reduce_sum(self.targets*self.dbl_one_hot,axis=1)*(1-self.done_mask_ph)

    self.one_hot_action = tf.one_hot(self.act_t_ph,self.num_actions)

    self.debug_eval_qs = tf.reduce_sum(self.q_vals*self.one_hot_action,axis=1)

    self.total_error = tf.reduce_sum(huber_loss(tf.reduce_sum(self.q_vals*self.one_hot_action,axis=1)-self.yi))

    ######

    # construct optimization op (with gradient clipping)
    self.learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")
    optimizer = self.optimizer_spec.constructor(learning_rate=self.learning_rate, **self.optimizer_spec.kwargs)
    self.train_fn = minimize_and_clip(optimizer, self.total_error,
                 var_list=q_func_vars, clip_val=grad_norm_clipping)

    # update_target_fn will be called periodically to copy Q network to target Q network
    update_target_fn = []
    for var, var_target in zip(sorted(q_func_vars,        key=lambda v: v.name),
                               sorted(target_q_func_vars, key=lambda v: v.name)):
        update_target_fn.append(var_target.assign(var))
    self.update_target_fn = tf.group(*update_target_fn)

    # construct the replay buffer
    self.replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len, lander=lander)
    self.replay_buffer_idx = None

    ###############
    # RUN ENV     #
    ###############
    self.model_initialized = False
    self.num_param_updates = 0
    self.mean_episode_reward      = -float('nan')
    self.best_mean_episode_reward = -float('inf')
    self.last_obs = initial_state
    self.log_every_n_steps = 1000

    self.start_time = None
    self.t = 0

  def stopping_criterion_met(self):
    return self.stopping_criterion is not None and self.stopping_criterion(self.env, self.t)

  def step_env(self,observations,agent):
    ### 2. Step the env and store the transition
    # At this point, "self.last_obs" contains the latest observation that was
    # recorded from the simulator. Here, your code needs to store this
    # observation and its outcome (reward, next observation, etc.) into
    # the replay buffer while stepping the simulator forward one step.
    # At the end of this block of code, the simulator should have been
    # advanced one step, and the replay buffer should contain one more
    # transition.
    # Specifically, self.last_obs must point to the new latest observation.
    # Usul functions you'll need to call:
    # obs, reward, done, info = env.step(action)
    # this steps the environment forward one step
    # obs = env.reset()
    # this resets the environment if you reached an episode boundary.
    # Don't forget to call env.reset() to get a new observation if done
    # is true!!
    # Note that you cannot use "self.last_obs" directly as input
    # into your network, since it needs to be processed to include context
    # from previous frames. You should check out the replay buffer
    # implementation in dqn_utils.py to see what functionality the replay
    # buffer exposes. The replay buffer has a function called
    # encode_recent_observation that will take the latest observation
    # that you pushed into the buffer and compute the corresponding
    # input that should be given to a Q network by appending some
    # previous frames.
    # Don't forget to include epsilon greedy exploration!
    # And remember that the first time you enter this loop, the model
    # may not yet have been initialized (but of course, the first step
    # might as well be random, since you haven't trained your net...)

    #####

    # YOUR CODE HERE
    #save last_obs in replay buffer
    #print "last ovs",self.last_obs
    #print self.last_obs
    #print self.last_obs
    self.replay_buffer_index = self.replay_buffer.store_frame(self.last_obs[agent]) #returns index of where that obs is stored

   
    obs_frames = self.replay_buffer.encode_recent_observation() #obtain recent frames 
    #print(self.replay_buffer.num_in_buffer,self.replay_buffer.frame_history_len,self.replay_buffer.next_idx)
    #print("before",obs_frames)
    if len(obs_frames.shape) ==1 or len(obs_frames.shape)==3:
      obs_frames=np.expand_dims(obs_frames,axis=0)
    #print("after",obs_frames.shape)
    
    if self.model_initialized:
      #take actions with q and arg_max and greedy exploration
      q_vals = self.session.run(self.q_vals,feed_dict={self.obs_t_ph : obs_frames})
      #print("q_vals shape",q_vals.shape)
      batch_action = []
      #choose based on probabilities
      for itr in q_vals:
        #print(itr)
        argmax = np.argmax(itr)
        p = np.zeros(self.num_actions)
        for ind in range(0,itr.shape[0]):
          if ind == argmax:
            p[ind] = 1-self.exploration.value(self.t)
          else:
            p[ind] = self.exploration.value(self.t)/(self.num_actions-1)
        #print("probabilities",p)
        action = np.random.choice(self.num_actions, p=p)
        #print(action)
        batch_action = np.append(batch_action, action)
      actions = batch_action
      #compute target values 
      
      best_action_idx = np.argmax(q_vals)
      action = actions
    else:
      action = np.random.randint(0,self.num_actions)
    #step environment forward
    return action 

  def store_effect(self,obs,reward,action,done,agent):
    

    #q_targets = self.session.run(self.targets,feed_dict={self.obs_tp1_ph : obs,
    #                                                          self.act_t_ph : actions})
    #store effect of last ob in replay buffer 

    self.replay_buffer.store_effect(idx=self.replay_buffer_index,
                                    action=action,
                                    reward=reward,
                                    done=done
                                    )
    #reset if done
    #if done:
    #  obs = self.env.reset()

    #save obs
    self.last_obs[agent] = obs

  def update_model(self,agent):
    ### 3. Perform experience replay and train the network.
    # note that this is only done if the replay buffer contains enough samples
    # for us to learn something useful -- until then, the model will not be
    # initialized and random actions should be taken
    if (self.t > self.learning_starts and \
        self.t % self.learning_freq == 0 and \
        self.replay_buffer.can_sample(self.batch_size)):
      # Here, you should perform training. Training consists of four steps:
      # 3.a: use the replay buffer to sample a batch of transitions (see the
      # replay buffer code for function definition, each batch that you sample
      # should consist of current observations, current actions, rewards,
      # next observations, and done indicator).
      # 3.b: initialize the model if it has not been initialized yet; to do
      # that, call
      #    initialize_interdependent_variables(self.session, tf.global_variables(), {
      #        self.obs_t_ph: obs_t_batch,
      #        self.obs_tp1_ph: obs_tp1_batch,
      #    })
      # where obs_t_batch and obs_tp1_batch are the batches of observations at
      # the current and next time step. The boolean variable model_initialized
      # indicates whether or not the model has been initialized.
      # Remember that you have to update the target network too (see 3.d)!
      # 3.c: train the model. To do this, you'll need to use the self.train_fn and
      # self.total_error ops that were created earlier: self.total_error is what you
      # created to compute the total Bellman error in a batch, and self.train_fn
      # will actually perform a gradient step and update the network parameters
      # to reduce total_error. When calling self.session.run on these you'll need to
      # populate the following placeholders:
      # self.obs_t_ph
      # self.act_t_ph
      # self.rew_t_ph
      # self.obs_tp1_ph
      # self.done_mask_ph
      # (this is needed for computing self.total_error)
      # self.learning_rate -- you can get this from self.optimizer_spec.lr_schedule.value(t)
      # (this is needed by the optimizer to choose the learning rate)
      # 3.d: periodically update the target network by calling
      # self.session.run(self.update_target_fn)
      # you should update every target_update_freq steps, and you may find the
      # variable self.num_param_updates useful for this (it was initialized to 0)
      #####
      #sample batch from replay buffer
      obs_t_batch, act_batch, rew_batch, obs_tp1_batch, done_mask = self.replay_buffer.sample(self.batch_size)
      #print(obs_t_batch.shape) 
      #initialize
      if not self.model_initialized:
        print(tf.local_variables())
        print(tf.global_variables())
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=str(agent)+'/q_func')
        initialize_interdependent_variables(self.session, tf.global_variables(), {
              self.obs_t_ph: obs_t_batch,
              self.obs_tp1_ph: obs_tp1_batch,
        }) 
        print "model initialized"
        #print(self.session.run(tf.report_uninitialized_variables()))

          
        self.model_initialized = True

      #print self.optimizer_spec.lr_schedule.value(self.t)
      #print obs_t_batch.shape
      self.curr_reward = np.mean(rew_batch)
      error,train,evals,q,action,yi = self.session.run([self.total_error, self.train_fn, self.debug_eval_qs ,self.q_vals,self.act_t_ph, self.yi],feed_dict={self.obs_t_ph : obs_t_batch,
                                                                    self.act_t_ph : act_batch,
                                                                    self.rew_t_ph : rew_batch,
                                                                    self.obs_tp1_ph : obs_tp1_batch,
                                                                    self.done_mask_ph : done_mask,
                                                                    self.learning_rate : self.optimizer_spec.lr_schedule.value(self.t)})
      # YOUR CODE HERE
      if self.t % self.log_every_n_steps == 0 and self.model_initialized:
        print("q",q[0])
        print("yis" , yi[0])
        print("error ", error)
        print("action",action)
        print("evaluated q",evals[0])
        print("exploration %f" % self.exploration.value(self.t))
        print "############################################"

        
      #print(self.num_param_updates,self.target_update_freq,self.num_param_updates % self.target_update_freq )
      if self.num_param_updates % self.target_update_freq ==0: 
        #print("update target")
        self.session.run(self.update_target_fn)
      self.num_param_updates += 1
    self.log_progress()
    self.t += 1
    #print self.t
  def log_progress(self):
    #episode_rewards = get_wrapper_by_name(self.env, "Monitor").get_episode_rewards()

    #if len(episode_rewards) > 0:
      #self.mean_episode_reward = np.mean(episode_rewards[-100:])
    self.mean_episode_reward = self.curr_reward
    #if len(episode_rewards) > 100:
    #  self.best_mean_episode_reward = max(self.best_mean_episode_reward, self.mean_episode_reward)
    self.mean_best_episode_reward = self.curr_reward
    if self.t % self.log_every_n_steps == 0 and self.model_initialized:
      print("Timestep %d" % (self.t,))
      print("mean reward (100 episodes) %f" % self.mean_episode_reward)
      print("best mean reward %f" % self.best_mean_episode_reward)
      #print("episodes %d" % len(episode_rewards))
      print("exploration %f" % self.exploration.value(self.t))
      print("learning_rate %f" % self.optimizer_spec.lr_schedule.value(self.t))
      if self.start_time is not None:
        print("running time %f" % ((time.time() - self.start_time) / 60.))
      else:
        self.logged_data=[]
        self.start_time = time.time()
      self.logged_data.append([self.t,self.mean_episode_reward,self.best_mean_episode_reward])
      sys.stdout.flush()

      #with open(self.rew_file, 'wb') as f:
        #pickle.dump([episode_rewards,self.logged_data], f, pickle.HIGHEST_PROTOCOL)

def learn(*args, **kwargs):
  alg = QLearner(*args, **kwargs)
  while not alg.stopping_criterion_met():
    alg.step_env()
    # at this point, the environment should have been advanced one step (and
    # reset if done was true), and self.last_obs should point to the new latest
    # observation
    alg.update_model()
    alg.log_progress()

