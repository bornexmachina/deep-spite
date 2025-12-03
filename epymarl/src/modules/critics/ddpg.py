import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class DDPG:
    def __init__(self, state_dim, args):
        n_actions = args.n_actions
        self.device = 'cuda' if args.use_cuda else 'cpu'
        self.actor = self.Actor(state_dim, n_actions).to(self.device)
        self.critic = self.Critic(state_dim, n_actions).to(self.device)
        self.target_actor = self.Actor(state_dim, n_actions).to(self.device)
        self.target_critic = self.Critic(state_dim, n_actions).to(self.device)
        self.actor_optimizer = th.optim.Adam(self.actor.parameters(), lr=args.adversary_actor_learning_rate)
        self.critic_optimizer = th.optim.Adam(self.actor.parameters(), lr=args.adversary_critic_learning_rate)
        self.memory = self.Memory(state_dim, args)
        self.noise = self.OrnUhlen(args)
        self.gamma = args.adversary_gamma
        self.tau = args.adversary_tau

    def get_actions(self, state):
        actions_pred = self.actor.forward(state).detach()
        noise = self.noise.sample()
        actions = (actions_pred.data.cpu().numpy() + noise)
        return actions

    def train(self, batch_size=32):
        state, action, new_state, reward, done = self.memory.sample(batch_size)
        done = done.astype(int)
        state = th.from_numpy(state).to(torch.float32).to(self.device)
        action = th.from_numpy(action).to(torch.float32).to(self.device)
        new_state = th.from_numpy(new_state).to(torch.float32).to(self.device)
        reward = th.from_numpy(reward).to(torch.float32).to(self.device)
        done = th.from_numpy(done).to(self.device)

        target_action = self.target_actor(new_state)
        target_q = self.target_critic(new_state, target_action)
        target_y = reward + self.gamma * target_q * (1 - done)
        y = self.critic(state, action)
        critic_loss = F.mse_loss(y, target_y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        action_pred = self.actor(state)
        actor_loss = -1 * th.mean(self.critic(state, action_pred))

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def soft_update(self):
        for target, src in zip(self.target_actor.parameters(), self.actor.parameters()):
            target.data.copy_(target.data * (1.0 - self.tau) + src.data * self.tau)

        for target, src in zip(self.target_critic.parameters(), self.critic.parameters()):
            target.data.copy_(target.data * (1.0 - self.tau) + src.data * self.tau)

    def add_to_memory(self, state, action, reward, new_state, done):
        self.memory.add_to_memory((state, action, reward, new_state, done))

    class Actor(nn.Module):
        def __init__(self, state_dim, action_dim):
            super(DDPG.Actor, self).__init__()

            self.fc1 = nn.Linear(state_dim, 400)
            nn.init.xavier_uniform_(self.fc1.weight)

            self.fc2 = nn.Linear(400, 300)
            nn.init.xavier_uniform_(self.fc2.weight)

            self.fc3 = nn.Linear(300, action_dim)
            nn.init.uniform_(self.fc3.weight, -0.003, 0.003)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.tanh(self.fc3(x))
            return x

    class Critic(nn.Module):
        def __init__(self, state_dim, action_dim):
            super().__init__()
            self.fc1 = nn.Linear(state_dim, 400)
            nn.init.xavier_uniform_(self.fc1.weight)

            self.fc2 = nn.Linear(400 + action_dim, 300)
            nn.init.xavier_uniform_(self.fc2.weight)

            self.fc3 = nn.Linear(300, 1)
            nn.init.uniform_(self.fc3.weight, -0.003, 0.003)

        def forward(self, state, action):
            s = F.relu(self.fc1(state))
            x = th.cat((s, action), dim=1)
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    class OrnUhlen:
        def __init__(self, args):
            self.n_actions = args.n_actions
            self.mu = args.adversary_mu
            self.theta = args.adversary_theta,
            self.sigma = args.adversary_sigma
            self.n_actions = self.n_actions
            self.X = np.ones(self.n_actions) * self.mu

        def sample(self):
            d_x = self.theta * (self.mu - self.X)
            d_x += self.sigma * np.random.randn(self.n_actions)
            self.X += d_x
            return self.X

        def reset(self):
            self.X = np.ones(self.n_actions) * self.mu

    class Memory:
        def __init__(self, state_dim, args):
            self.capacity = args.adversary_memory_capacity
            self.index = None
            self.index_list = list(range(self.capacity))

            self.state = np.zeros((self.capacity, state_dim))
            self.new_state = np.zeros((self.capacity, state_dim))
            self.actions = np.zeros((self.capacity, args.n_actions))
            self.rewards = np.zeros((self.capacity, 1))
            self.done = np.full((self.capacity, 1), True, dtype=bool)

        def add_to_memory(self, experience):
            if self.index is None or self.index == self.capacity - 1:
                self.index = 0
            else:
                self.index += 1

            state, actions, rewards, new_state, done = experience
            self.state[self.index] = state.cpu()
            self.new_state[self.index] = new_state.detach().cpu()
            self.actions[self.index] = actions.cpu()
            self.rewards[self.index] = rewards.detach().cpu()
            self.done[self.index] = done.cpu()

        def sample(self, batch_size):
            indices = np.random.choice(self.index_list, batch_size)
            states = self.state[indices]
            actions = self.actions[indices]
            new_states = self.new_state[indices]
            rewards = self.rewards[indices]
            done = self.done[indices]
            return states, actions, new_states, rewards, done
