import torch

from .training_steps import TrainingSteps


class TrainingStepsSplitEpoch(TrainingSteps):
    def on_training_step_detection(self, batch):
        loss_dict = self.get_loss_dict_detection(batch)
        log_dict = self.get_log_dict_detection(loss_dict)
        loss_weights = self.get_loss_weights_detection()
        agg_loss = self.aggregate_loss(loss_dict, loss_weights)

        return agg_loss, log_dict

    def get_loss_dict_detection(self, batch):
        x, targets = batch
        loss_dict, detections = self.model.forward(x, targets)
        if 'loss_recognition' in loss_dict:
            del loss_dict['loss_recognition']
        return loss_dict

    def get_loss_weights_detection(self):
        return {
            'loss_rpn_box_reg': torch.tensor(self.rpn_box_weight),
            'loss_objectness': torch.tensor(self.rpn_objectness_weight),
            'loss_box_reg': torch.tensor(self.roi_box_weight),
            'loss_classifier': torch.tensor(self.roi_classifier_weight),
        }

    def get_log_dict_detection(self, loss_dict):
        log_dict = {}
        log_dict['rpn_box_loss'] = loss_dict['loss_rpn_box_reg']
        log_dict['rpn_objectness_loss'] = loss_dict['loss_objectness']
        log_dict['roi_box_loss'] = loss_dict['loss_box_reg']
        log_dict['roi_classifier_loss'] = loss_dict['loss_classifier']
        return log_dict

    def get_recog_loss(self, x, targets):
        return self.model.get_recog_loss(x, targets)

    def on_training_step_recognition(self, batch):
        x, targets = batch
        recog_loss = self.get_recog_loss(x, targets)
        recog_loss = torch.mul(recog_loss, self.roi_recognition_weight)
        log_dict = {
            'roi_recognition_loss': recog_loss
        }

        return recog_loss, log_dict
