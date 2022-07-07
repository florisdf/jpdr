from .training_loop import TrainingLoop


class TrainingLoopSplitEpoch(TrainingLoop):
    def __init__(
        self,
        *args,
        dl_train_det,
        dl_train_recog,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dl_train_det = dl_train_det
        self.dl_train_recog = dl_train_recog

    def training_epoch(self):
        self.training_detection_epoch()
        self.training_recognition_epoch()

    def training_detection_epoch(self):
        self._general_training_epoch(
            self.dl_train_det,
            self.training_steps.on_training_step_detection
        )

    def training_recognition_epoch(self):
        self._general_training_epoch(
            self.dl_train_recog,
            self.training_steps.on_training_step_recognition
        )
