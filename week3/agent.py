import numpy as np

from collections import namedtuple

import torch

import torch.optim as optim

import torch.nn as nn

from network import DQN

from buffers import ReplayBuffer

import time


class DQNAgent():
    """Deep Q-learning agent."""

    # def __init__(self,
                 # env, device=DEVICE, summary_writer=writer,  # noqa
                 # hyperparameters=DQN_HYPERPARAMS):  # noqa

    rewards = []
    total_reward = 0
    birth_time = 0
    n_iter = 0
    n_games = 0
    ts_frame = 0
    ts = time.time()

    # Memory = namedtuple(
        # 'Memory', ['obs', 'action', 'new_obs', 'reward', 'done'],
        # verbose=False, rename=False)
    Memory = namedtuple(
        'Memory', ['obs', 'action', 'new_obs', 'reward', 'done'],
        rename=False)

    def __init__(self, env, hyperparameters, device, summary_writer=None):

        """Set parameters, initialize network."""

        state_space_shape = env.observation_space.shape
        action_space_size = env.action_space.n

        self.env = env

        self.online_network = DQN(
            state_space_shape, action_space_size).to(device)

        self.target_network = DQN(
            state_space_shape, action_space_size).to(device)

        # XXX maybe not really necesary?
        self.update_target_network()

        self.experience_replay = None

        self.accumulated_loss = []
        self.device = device

        self.optimizer = optim.Adam(self.online_network.parameters(),
                                    lr=hyperparameters['learning_rate'])

        # TODO should be parametrized?
        self.double_DQN = True

        # Discount factor
        self.gamma = hyperparameters['gamma']

        # XXX ???
        self.n_multi_step = hyperparameters['n_multi_step']

        self.replay_buffer = ReplayBuffer(
            hyperparameters['buffer_capacity'],
            hyperparameters['n_multi_step'],
            hyperparameters['gamma'])

        self.birth_time = time.time()

        self.iter_update_target = hyperparameters['n_iter_update_target']
        self.buffer_start_size = hyperparameters['buffer_start_size']

        self.summary_writer = summary_writer

        # Greedy search hyperparameters
        self.epsilon_start = hyperparameters['epsilon_start']
        self.epsilon = hyperparameters['epsilon_start']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_final = hyperparameters['epsilon_final']

    def get_max_action(self, obs):
        '''
        Forward pass of the NN to obtain the action of the given observations
        '''
        # convert the observation in tensor
        state_t = torch.tensor(np.array([obs])).to(self.device)
        # forawrd pass
        q_values_t = self.online_network(state_t)
        # get the maximum value of the output (i.e. the best action to take)
        _, act_t = torch.max(q_values_t, dim=1)
        return int(act_t.item())

    def act(self, obs):
        '''
        Greedy action outputted by the NN in the CentralControl
        '''
        return self.get_max_action(obs)

    def act_eps_greedy(self, obs):
        '''
        E-greedy action
        '''

        # In case of a noisy net, it takes a greedy action
        # if self.noisy_net:
            # return self.act(obs)

        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.act(obs)

    def update_target_network(self):
        """Update target network weights with current online network values."""

        self.target_network.load_state_dict(
            self.online_network.state_dict())

    def set_optimizer(self, learning_rate):
        self.optimizer = optim.Adam(
            self.online_network.parameters(), lr=learning_rate)

    def sample_and_optimize(self, batch_size):
        '''
        Sample batch_size memories from the buffer and optimize them
        '''

        # XXX this should be the part where it waits until it has enough
        # experience
        if len(self.replay_buffer) > self.buffer_start_size:
            # sample
            mini_batch = self.replay_buffer.sample(batch_size)
            # optimize
            # l_loss = self.cc.optimize(mini_batch)
            l_loss = self.optimize(mini_batch)
            self.accumulated_loss.append(l_loss)

        # update target NN
        if self.n_iter % self.iter_update_target == 0:
            self.update_target_network()

    def optimize(self, mini_batch):
        '''
        Optimize the NN
        '''
        # reset the grads
        self.optimizer.zero_grad()
        # caluclate the loss of the mini batch
        loss = self._calulate_loss(mini_batch)
        loss_v = loss.item()

        # do backpropagation
        loss.backward()
        # one step of optimization
        self.optimizer.step()

        return loss_v

    def _calulate_loss(self, mini_batch):
        '''
        Calculate mini batch's MSE loss.
        It support also the double DQN version
        '''

        states, actions, next_states, rewards, dones = mini_batch

        # convert the data in tensors
        states_t = torch.as_tensor(states, device=self.device)
        next_states_t = torch.as_tensor(next_states, device=self.device)
        actions_t = torch.as_tensor(actions, device=self.device)
        rewards_t = torch.as_tensor(
            rewards, dtype=torch.float32, device=self.device)

        done_t = torch.as_tensor(dones, dtype=torch.uint8, device=self.device)  # noqa

        # Value of the action taken previously (recorded in actions_v)
        # in state_t
        state_action_values = self.online_network(
            states_t).gather(1, actions_t[:, None]).squeeze(-1)

        # NB gather is a differentiable function

        # Next state value with Double DQN. (i.e. get the value predicted
        # by the target nn, of the best action predicted by the moving nn)
        if self.double_DQN:
            double_max_action = self.online_network(next_states_t).max(1)[1]
            double_max_action = double_max_action.detach()
            target_output = self.target_network(next_states_t)

            # NB: [:,None] add an extra dimension
            next_state_values = torch.gather(
                target_output, 1, double_max_action[:, None]).squeeze(-1)

        # Next state value in the normal configuration
        else:
            next_state_values = self.target_nn(next_states_t).max(1)[0]

        next_state_values = next_state_values.detach()  # No backprop

        # Use the Bellman equation
        expected_state_action_values = rewards_t + \
            (self.gamma**self.n_multi_step) * next_state_values

        # compute the loss
        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def reset_stats(self):
        '''
        Reset the agent's statistics
        '''
        self.rewards.append(self.total_reward)
        self.total_reward = 0
        self.accumulated_loss = []
        self.n_games += 1

    def add_env_feedback(self, obs, action, new_obs, reward, done):
        '''
        Acquire a new feedback from the environment. The feedback is
        constituted by the new observation, the reward and the done boolean.
        '''

        # Create the new memory and update the buffer
        new_memory = self.Memory(
            obs=obs, action=action, new_obs=new_obs, reward=reward, done=done)

        # Append it to the replay buffer
        self.replay_buffer.append(new_memory)

        # update the variables
        self.n_iter += 1

        # TODO check this...
        # decrease epsilon
        self.epsilon = max(
            self.epsilon_final,
            self.epsilon_start - self.n_iter/self.epsilon_decay)

        self.total_reward += reward

    def print_info(self):
        '''
        Print information about the agent
        '''

        fps = (self.n_iter-self.ts_frame)/(time.time()-self.ts)

        # TODO replace with proper logger
        print('%d %d rew:%d mean_rew:%.2f eps:%.2f, fps:%d, loss:%.4f' % (
            self.n_iter, self.n_games, self.total_reward,
            np.mean(self.rewards[-40:]),
            self.epsilon, fps, np.mean(self.accumulated_loss)))

        self.ts_frame = self.n_iter
        self.ts = time.time()

        if self.summary_writer is not None:
            self.summary_writer.add_scalar(
                'reward', self.total_reward, self.n_games)
            self.summary_writer.add_scalar(
                'mean_reward', np.mean(self.rewards[-40:]), self.n_games)
            self.summary_writer.add_scalar(
                '10_mean_reward', np.mean(self.rewards[-10:]), self.n_games)
            self.summary_writer.add_scalar(
                'epsilon', self.epsilon, self.n_games)
            self.summary_writer.add_scalar(
                'loss', np.mean(self.accumulated_loss), self.n_games)
