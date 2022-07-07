import torch

from .training_steps import TrainingSteps


class TrainingStepsSameEpoch(TrainingSteps):
    def on_training_step(self, batch):
        loss_dict = self.get_loss_dict(batch)
        log_dict = self.get_log_dict(loss_dict)
        loss_weights = self.get_loss_weights()
        agg_loss = self.aggregate_loss(loss_dict, loss_weights)

        return agg_loss, log_dict

    def get_loss_dict(self, batch):
        x, targets = batch
        loss_dict, detections = self.model.forward(x, targets)
        return loss_dict

    def get_loss_weights(self):
        return {
            'loss_rpn_box_reg': torch.tensor(self.rpn_box_weight),
            'loss_objectness': torch.tensor(self.rpn_objectness_weight),
            'loss_box_reg': torch.tensor(self.roi_box_weight),
            'loss_classifier': torch.tensor(self.roi_classifier_weight),
            'loss_recognition': torch.tensor(self.roi_recognition_weight),
        }

    def get_log_dict(self, loss_dict):
        log_dict = {}
        log_dict['rpn_box_loss'] = loss_dict['loss_rpn_box_reg']
        log_dict['rpn_objectness_loss'] = loss_dict['loss_objectness']
        log_dict['roi_box_loss'] = loss_dict['loss_box_reg']
        log_dict['roi_classifier_loss'] = loss_dict['loss_classifier']
        log_dict['roi_recognition_loss'] = loss_dict['loss_recognition']
        return log_dict
