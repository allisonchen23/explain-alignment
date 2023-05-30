import numpy as np
import sys
import torch

sys.path.insert(0, 'src')
from trainer.base_trainer import BaseTrainer
from utils.utils import inf_loop, MetricTracker

from model import metric as module_metric

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size)) * 3

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        train_predictions = []
        train_targets = []
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            # Calculate loss
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            # Obtain predictions and move to numpy
            # Calculate predictions based on argmax
            prediction = torch.argmax(output, dim=1)
            # Move predictions and target to cpu and convert to numpy to calculate metrics
            prediction = prediction.cpu().numpy()
            target = target.cpu().numpy()

            train_predictions.append(prediction)
            train_targets.append(target)

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            # for met in self.metric_ftns:
            #     self.train_metrics.update(met.__name__, met(prediction, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        # log = self.train_metrics.result()

        train_predictions = np.concatenate(train_predictions, axis=0)
        train_targets = np.concatenate(train_targets, axis=0)
        # log = module_metric.compute_metrics(
        #     metric_fns=self.metric_ftns,
        #     prediction=train_predictions,
        #     target=train_targets
        # )
        log = {}
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            log.update({"lr": tuple(self.lr_scheduler.get_last_lr())})
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        val_predictions = []
        val_targets = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Calculate predictions based on argmax
                prediction = torch.argmax(output, dim=1)
                # Move predictions and target to cpu and convert to numpy to calculate metrics
                prediction = prediction.cpu().numpy()
                target = target.cpu().numpy()

                val_predictions.append(prediction)
                val_targets.append(target)

        val_predictions = np.concatenate(val_predictions, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        if len(val_targets.shape) == 2:
            val_targets = np.argmax(val_targets, axis=1)
        assert val_targets.shape == val_predictions.shape

        # Calculate metrics for validation
        val_log = module_metric.compute_metrics(
            metric_fns=self.metric_ftns,
            prediction=val_predictions,
            target=val_targets
        )
        val_log.update({'loss': loss.item()})

        # Add metrics and loss to tensorboard
        self.writer.set_step((epoch - 1) * len(self.data_loader), 'valid')
        for key, val in val_log.items():
            # Only log scalars
            try:
                val = float(val)
            except:
                continue
            self.valid_metrics.update(key, val)

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        # return self.valid_metrics.result()
        return val_log

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
