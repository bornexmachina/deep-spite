from random import random

import numpy as np
import torch as th
import torch.nn.functional as F
from envs import REGISTRY as env_REGISTRY
from envs.zerosum import ZeroSumEnv
from modules.critics.ddpg import DDPG

from .maddpg_controller import MADDPGMAC, gumbel_softmax, gumbel_softmax_sample


class RandomAttackAdversarialController(MADDPGMAC):
    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        self.compromised_agent_index = args.compromised_agent
        self.attack_rate = args.attack_rate

    def select_actions(self, ep_batch, t_ep, t_env=0, test_mode=False):
        if random() < self.attack_rate:  # Attack at random times.
            agent_outputs = self.forward(ep_batch, t_ep)
            actions = gumbel_softmax(agent_outputs, hard=False).argmax(dim=-1)
            y_comp = gumbel_softmax_sample(agent_outputs[0][self.compromised_agent_index], t_ep)
            action_comp = (y_comp == y_comp.min(-1, keepdim=True)[0]).float().argmax()
            actions[self.compromised_agent_index] = action_comp
        else:
            actions = super().select_actions(ep_batch, t_ep)
        return actions


class TimedAttackAdversarialController(MADDPGMAC):
    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        self.compromised_agent_index = args.compromised_agent
        self.attack_threshold = args.attack_threshold

    def select_actions(self, ep_batch, t_ep, t_env=0, test_mode=False):
        agent_outputs = self.forward(ep_batch, t_ep)
        actions = gumbel_softmax(agent_outputs, hard=False).argmax(dim=-1)
        y_comp = gumbel_softmax_sample(agent_outputs[0][self.compromised_agent_index], t_ep)
        if y_comp.max() - y_comp.min() > self.attack_threshold:  # Attack only if the attack has a big impact.
            action_comp = (y_comp == y_comp.min(-1, keepdim=True)[0]).float().argmax()
            actions[self.compromised_agent_index] = action_comp
        return actions


class CounterfactualReasoningAdversarialController(MADDPGMAC):
    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        pass


class KLAdversarialController(MADDPGMAC):
    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        self.hypothetical_obs = None
        self.compromised_agent_index = args.compromised_agent
        self.env_copy = env_REGISTRY[args.env](**args.env_args)
        self.attack_rate = args.attack_rate
        self.adversary_agent = DDPG(scheme['obs']['vshape'] + scheme['actions_onehot']['vshape'][0], args)
        self.adversary_action = None

        obs = self.env_copy.reset()[0][self.compromised_agent_index]
        obs = np.append(obs, np.zeros(scheme['actions_onehot']['vshape']))
        self.device = 'cuda' if args.use_cuda else 'cpu'
        self.adversary_obs = th.from_numpy(obs).to(th.float32).to(self.device)

    def select_actions(self, ep_batch, t_ep, t_env=0, test_mode=False):
        if t_ep > 0:
            actual_obs = ep_batch['obs'][0][t_ep]
            ep_batch['obs'][0][t_ep][self.compromised_agent_index] = self.hypothetical_obs
            next_agent_outputs = self.forward(ep_batch, t_ep)
            ep_batch['obs'][0][t_ep] = actual_obs
            next_agent_outputs_atk = self.forward(ep_batch, t_ep)
            adv_reward = F.kl_div(next_agent_outputs, next_agent_outputs_atk)
            self.adversary_obs = th.concat([actual_obs[0], self.adversary_action]).to(th.float32)
            self.adversary_agent.add_to_memory(
                th.concat([self.hypothetical_obs.to(self.device), self.adversary_action.to(self.device)]),
                self.adversary_action, adv_reward, actual_obs[0][self.compromised_agent_index],
                ep_batch['terminated'][0, t_ep, 0] == 1)
            self.adversary_agent.train()

        agent_outputs = self.forward(ep_batch, t_ep)
        actions = gumbel_softmax(agent_outputs, hard=False).argmax(dim=-1)
        self.env_copy.step(actions[0])
        self.hypothetical_obs = th.from_numpy(self.env_copy.get_obs()[self.compromised_agent_index]).to(th.float32)

        adversarial_agent_output = self.adversary_agent.get_actions(self.adversary_obs)
        agent_outputs[0][self.compromised_agent_index] = th.from_numpy(adversarial_agent_output)
        perturbed_actions = gumbel_softmax(agent_outputs, hard=False).argmax(dim=-1)
        self.adversary_action = th.zeros_like(th.from_numpy(adversarial_agent_output))
        self.adversary_action[perturbed_actions[0][self.compromised_agent_index]] = 1
        self.adversary_action = self.adversary_action.to(self.device)
        if random() < self.attack_rate:  # Attack at random times.
            return perturbed_actions
        return actions


class ZeroSumAdversarialController(MADDPGMAC):
    """
    Zero sum blackbox attack.
    """

    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        self.compromised_agent_index = args.compromised_agent
        expert_storage_path = f'{self.args.expert_storage_path}/{self.args.learner}/{self.args.env_args["key"]}'
        action_storage_path = f'{expert_storage_path}/actions.txt'
        obs_storage_path = f'{expert_storage_path}/obs.txt'
        expert_actions = np.loadtxt(action_storage_path)
        expert_state = np.loadtxt(obs_storage_path)
        expert_actions[0][0] = 1
        self.zero_sum_env = ZeroSumEnv(scheme['obs']['vshape'], scheme['actions_onehot']['vshape'][0], expert_actions,
                                       expert_state, args)
        self.adversary_agent = DDPG(scheme['obs']['vshape'], args)
        self.attack_rate = args.attack_rate
        self.n_actions = args.n_actions

        self.previous_adversarial_action = None

    def select_actions(self, ep_batch, t_ep, t_env=0, test_mode=False):
        if t_ep > 0:
            adv_reward = self.zero_sum_env.step({'agent_a': self.previous_adversarial_action,
                                                 'agent_s': ep_batch["obs"][:, t_ep][0][
                                                     self.compromised_agent_index]}).cpu()
            self.adversary_agent.add_to_memory(ep_batch["obs"][:, t_ep - 1][0][self.compromised_agent_index],
                                               self.previous_adversarial_action, adv_reward,
                                               ep_batch["obs"][:, t_ep][0][self.compromised_agent_index],
                                               ep_batch['terminated'][0, t_ep, 0] == 1)
            self.adversary_agent.train(1)
            self.zero_sum_env.reset()
        if random() < self.attack_rate:  # Attack at random times.
            agent_outputs = self.forward(ep_batch, t_ep)
            action_comp = self.adversary_agent.get_actions(
                ep_batch["obs"][:, t_ep][0][self.compromised_agent_index])
            agent_outputs[0][self.compromised_agent_index] = th.from_numpy(action_comp)
            actions = gumbel_softmax(agent_outputs, hard=False).argmax(dim=-1)
            self.previous_adversarial_action = np.zeros_like(action_comp, dtype=np.float32)
            self.previous_adversarial_action[actions[0][self.compromised_agent_index]] = 1
            self.previous_adversarial_action = th.from_numpy(self.previous_adversarial_action)
        else:
            actions = super().select_actions(ep_batch, t_ep)
            action_comp_onehot = th.zeros(self.n_actions)
            action_comp_onehot[actions[0][self.compromised_agent_index]] = 1
            self.previous_adversarial_action = action_comp_onehot
        return actions
