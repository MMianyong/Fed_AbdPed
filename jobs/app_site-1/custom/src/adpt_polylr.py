from torch.optim.lr_scheduler import _LRScheduler

import yaml


"""
To remain consistent with nnU-Net’s default training strategy, the initial learning rate was fixed at 0.01 for federated learning.
For each federated round, the effective epoch number was accumulated across previous rounds, allowing the learning rate to be adjusted in a manner equivalent to centralized training. Before training started, a randomly initialized model was distributed to both centers.
This same initialization was used for both federated and non-federated training to ensure a fair comparison.
"""

class PolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None, yaml_file: str = "./fed_pedabd/jobs/exp_set.yaml",):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        self.yaml_file = yaml_file
        super().__init__(optimizer, current_step if current_step is not None else -1)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

            
        # This reads the YAML again every step
        with open(self.yaml_file, "r") as f:
            exp_set = yaml.safe_load(f)

        epoch_per_round=exp_set['epoch_per_round']
        current_round=exp_set['current_round']
        epoch_per_round=round(epoch_per_round)
        total_rounds=exp_set['total_rounds']
        current_step=current_round*epoch_per_round+current_step
        self.max_steps=total_rounds*epoch_per_round

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
