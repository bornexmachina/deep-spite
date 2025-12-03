from .adversarial_controller import RandomAttackAdversarialController, TimedAttackAdversarialController, \
    ZeroSumAdversarialController, KLAdversarialController

REGISTRY = {}

from .basic_controller import BasicMAC
from .non_shared_controller import NonSharedMAC
from .maddpg_controller import MADDPGMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["non_shared_mac"] = NonSharedMAC
REGISTRY["maddpg_mac"] = MADDPGMAC
REGISTRY["random_attack_maddpg_mac"] = RandomAttackAdversarialController
REGISTRY["timed_attack_maddpg_mac"] = TimedAttackAdversarialController
REGISTRY["zero_sum_maddpg_mac"] = ZeroSumAdversarialController
REGISTRY["kl_maddpg_mac"] = KLAdversarialController
