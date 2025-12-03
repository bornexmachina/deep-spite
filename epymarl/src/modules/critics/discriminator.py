import torch as th
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    """
    Essentially a binary classifier that aims to distinguish between "real" trajectories
    and trajectories produced by the adversary.
    """

    def __init__(self, input_shape, args):
        super().__init__()
        device = 'cuda' if args.use_cuda else 'cpu'
        self.fc1 = nn.Linear(input_shape, args.discriminator_hidden_dim).to(device)
        self.fc2 = nn.Linear(args.discriminator_hidden_dim, args.discriminator_hidden_dim).to(device)
        self.fc3 = nn.Linear(args.discriminator_hidden_dim, 1).to(device)
        self.prob = None

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        prob = F.softmax(self.fc3(x))
        return prob

    def reward(self, x):
        prob = self.forward(x)
        rewards = th.log(prob) - th.log(1 - prob)
        return rewards
