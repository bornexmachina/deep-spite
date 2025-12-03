from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .actor_critic_learner import ActorCriticLearner
from .maddpg_learner import MADDPGLearner
from .spiteful_maddpg_learner_stage1 import SpitefulMADDPGLearner_stage1
from .spiteful_maddpg_learner_stage2 import SpitefulMADDPGLearner_stage2
from .spiteful_maddpg_learner_stage3 import SpitefulMADDPGLearner_stage3
from .ppo_learner import PPOLearner
REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["actor_critic_learner"] = ActorCriticLearner
REGISTRY["maddpg_learner"] = MADDPGLearner
REGISTRY["spiteful_maddpg_learner_stage1"] = SpitefulMADDPGLearner_stage1
REGISTRY["spiteful_maddpg_learner_stage2"] = SpitefulMADDPGLearner_stage2
REGISTRY["spiteful_maddpg_learner_stage3"] = SpitefulMADDPGLearner_stage3
REGISTRY["ppo_learner"] = PPOLearner
