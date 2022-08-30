from pytorch_lightning.callbacks import EarlyStopping


class ThresholdStopping(EarlyStopping):
    """
    Stop training if the monitored quantity is greater than the threshold. Here a model
    is stopped if it reaches 1.0 training accuracy.
    """

    def __init__(self, monitor, threshold):
        super().__init__(monitor=monitor, check_on_train_epoch_end=True)
        self.monitor = monitor
        self.threshold = threshold

    def _run_early_stopping_check(self, trainer):
        logs = trainer.callback_metrics
        if (
            trainer.fast_dev_run
            or not self._validate_condition_metric(  # disable early_stopping with fast_dev_run
                logs
            )
        ):  # short circuit if metric not present
            return
        current = logs[self.monitor].squeeze()
        should_stop = False
        reason = ""
        if current >= self.threshold:
            should_stop = True
            reason = "Training stopped after epoch {}: {} = {}".format(
                trainer.current_epoch, self.monitor, current
            )
        # stop every ddp process if any world process decides to stop
        should_stop = trainer.strategy.reduce_boolean_decision(should_stop)
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop:
            self.stopped_epoch = trainer.current_epoch
        if reason and self.verbose:
            self._log_info(trainer, reason, self.log_rank_zero_only)
