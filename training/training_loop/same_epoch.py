from .training_loop import TrainingLoop


class TrainingLoopSameEpoch(TrainingLoop):
    def __init__(
        self,
        *args,
        dl_train,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dl_train = dl_train

    def training_epoch(self):
        self._general_training_epoch(
            self.dl_train,
            self.training_steps.on_training_step
        )
