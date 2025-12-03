import gym
import numpy as np
import torch as th
import torch.nn.functional as F

from modules.critics.discriminator import Discriminator


class ZeroSumEnv(gym.Env):
    """
    The zero-sum environment is an environment to train and evaluate adversarial attacks.

    It encapsulates the AIRL discriminator used to train the adversary policy.
    """

    def __init__(self, state_dims, n_actions, expert_actions, expert_state, args):
        self.D = Discriminator(state_dims + n_actions, args)
        self.optimizer = th.optim.Adam(self.D.parameters(), lr=args.discriminator_learning_rate)
        self.expert_actions = expert_actions
        self.expert_state = expert_state
        self.memory_capacity = args.discriminator_memory_capacity
        self.action_history = np.zeros((self.memory_capacity, n_actions))
        self.state_history = np.zeros((self.memory_capacity, state_dims))
        self.write_ptr = 0
        self.read_ptr = 0
        self.memory_full = False
        self.device = 'cuda' if args.use_cuda else 'cpu'

    def reset(self):
        """
        Train the discriminator
        :return:
        """
        state = th.concat([th.from_numpy(self.expert_state), th.from_numpy(self.state_history[:self.read_ptr + 1])])
        actions = th.concat(
            [th.from_numpy(self.expert_actions), th.from_numpy(self.action_history[:self.read_ptr + 1])])
        target = th.concat(
            [th.ones([len(self.expert_actions), 1]), th.zeros([len(self.action_history[:self.read_ptr + 1]), 1])]).to(
            self.device)
        state_action = th.concat([state, actions], axis=1).to(th.float32).to(self.device)
        output = self.D.forward(state_action)
        loss = F.binary_cross_entropy(output, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def render(self, mode="human"):
        raise NotImplementedError

    def step(self, action: dict):
        """
        Obtain reward from discriminator.
        :param action: Action is expected to be a dict {'agent_a': [], 'agent_s': []}
        :return: Reward
        """
        agent_a, agent_s = action['agent_a'].to(self.device), action['agent_s'].to(self.device)
        state_action = th.concat([agent_s, agent_a])
        reward = self.D.forward(state_action)
        self.action_history[self.write_ptr] = agent_a.cpu()
        self.state_history[self.write_ptr] = agent_s.cpu()
        self.write_ptr = (self.write_ptr + 1) % self.memory_capacity
        self.read_ptr = min(self.read_ptr, self.memory_capacity)
        return reward
