import functools
from pathlib import Path
import sys

import torch
from torchvision.ops.poolers import LevelMapper
from tqdm import tqdm
import wandb

from ..training_steps import TrainingSteps


class TrainingLoop:
    def __init__(
        self,
        training_steps: TrainingSteps,
        optimizer,
        device,
        num_epochs,
        dl_val,
        save_unique=False,
        save_last=True,
        save_best=True,
        log_fpn_levels=True,
    ):
        self.training_steps = training_steps
        self.model = self.training_steps.model.to(device)
        self.num_epochs = num_epochs
        self.dl_val = dl_val

        self.max_metrics = {}
        self.ckpt_dir = Path('ckpts')

        self.device = device
        self.optimizer = optimizer
        self.epoch_idx = 0
        self.train_batch_idx = -1
        self.val_batch_idx = -1

        self.save_unique = save_unique
        self.save_last = save_last
        self.save_best = save_best

        if log_fpn_levels:
            self.add_fpn_level_logger()

    def run(self):
        self.max_metrics = {}

        # Training loop
        for self.epoch_idx in tqdm(range(self.num_epochs), leave=True):
            # Training epoch
            self.model.train()
            self.training_steps.on_before_training_epoch()
            self.training_epoch()
            log_dict = self.training_steps.on_after_training_epoch()
            log(log_dict, epoch_idx=self.epoch_idx)

            # Validation epoch
            self.model.eval()
            self.training_steps.on_before_validation_epoch()
            self.validation_epoch()
            log_dict = self.training_steps.on_after_validation_epoch()
            log(log_dict, epoch_idx=self.epoch_idx)

            # Update and log max_metrics
            self.update_max_metrics(log_dict)
            log(self.max_metrics, epoch_idx=self.epoch_idx)

            # Create checkpoints
            self.create_checkpoints(log_dict)

    def training_epoch(self):
        raise NotImplementedError

    def _general_training_epoch(self, dl_train, on_training_step):
        for (self.train_batch_idx, (x, targets)) in enumerate(
            tqdm(dl_train, leave=False), start=self.train_batch_idx + 1
        ):
            x = x.to(self.device)
            targets = [det_target_to_device(t, self.device) for t in targets]
            loss, log_dict = on_training_step(
                (x, targets),
            )
            loss.backward()
            self.optimizer.step()
            self.model.zero_grad()
            log(log_dict, epoch_idx=self.epoch_idx,
                batch_idx=self.train_batch_idx,
                section='TrainLoss')
        if torch.isnan(loss):
            sys.exit('Loss is NaN. Exiting...')

    def validation_epoch(self):
        # Validation loop
        for self.val_batch_idx, (x, targets) in tqdm(
            enumerate(self.dl_val, start=self.val_batch_idx + 1),
            leave=False,
            total=len(self.dl_val),
        ):
            x = x.to(self.device)
            targets = [det_target_to_device(t, self.device) for t in targets]

            with torch.no_grad():
                self.training_steps.on_validation_step((x, targets))

    def add_fpn_level_logger(self):
        self.fpn_levels = []

        map_levels = LevelMapper.__call__
        on_after_training_epoch = self.training_steps.on_after_training_epoch
        on_after_validation_epoch = \
            self.training_steps.on_after_validation_epoch

        @functools.wraps(map_levels)
        def track_map_levels(*args, **kwargs):
            levels = map_levels(*args, **kwargs)
            self.fpn_levels.extend(levels.cpu().numpy())
            return levels

        @functools.wraps(on_after_training_epoch)
        def reset_train_fpn_levels(*args, **kwargs):
            log_dict = on_after_training_epoch(*args, **kwargs)
            set_reset_fpn_levels(log_dict)
            return log_dict

        @functools.wraps(on_after_validation_epoch)
        def reset_val_fpn_levels(*args, **kwargs):
            log_dict = on_after_validation_epoch(*args, **kwargs)
            set_reset_fpn_levels(log_dict)
            return log_dict

        def set_reset_fpn_levels(log_dict):
            log_dict['FPN levels'] = wandb.Histogram(self.fpn_levels)
            self.fpn_levels = []

        LevelMapper.__call__ = track_map_levels
        self.training_steps.on_after_training_epoch = reset_train_fpn_levels
        self.training_steps.on_after_validation_epoch = reset_val_fpn_levels

    def update_max_metrics(self, val_log_dict):
        for k, v in val_log_dict.items():
            if isinstance(v, wandb.Histogram):
                continue

            max_name = f'Max{k}'
            if (
                max_name not in self.max_metrics
                or v > self.max_metrics[max_name]
            ):
                self.max_metrics[max_name] = v

    def create_checkpoints(self, val_log_dict=None):
        file_prefix = f"{wandb.run.id}_" if self.save_unique else ""
        file_suffix = '.pth'

        if not self.ckpt_dir.exists():
            self.ckpt_dir.mkdir(parents=True)

        if (
            self.save_best
            and val_log_dict is not None
            and val_log_dict['COCO_recog/AP@[0.50:0.95]']
            >= self.max_metrics['MaxCOCO_recog/AP@[0.50:0.95]']
        ):
            torch.save(
                self.model.state_dict(),
                self.ckpt_dir / f'{file_prefix}best{file_suffix}'
            )

        if self.save_last:
            torch.save(
                self.model.state_dict(),
                self.ckpt_dir / f'{file_prefix}last{file_suffix}'
            )


def det_target_to_device(det_target, device):
    det_target['boxes'] = det_target['boxes'].to(device)
    det_target['labels'] = det_target['labels'].to(device)
    det_target['product_ids'] = det_target['product_ids'].to(device)
    return det_target


def log(log_dict, epoch_idx, batch_idx=None, section=None):
    def get_key(k):
        if section is None:
            return k
        else:
            return f'{section}/{k}'

    def get_value(v):
        if isinstance(v, torch.Tensor):
            return v.detach().cpu()
        else:
            return v

    for k, v in log_dict.items():
        wandb_dict = {get_key(k): get_value(v),
                      "epoch": epoch_idx}
        if batch_idx is not None:
            wandb_dict['batch_idx'] = batch_idx
        wandb.log(wandb_dict)
