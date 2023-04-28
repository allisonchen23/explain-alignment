# import torch.nn as nn
import torch
import numpy as np
from abc import abstractmethod


class BaseModel(torch.nn.Module):
    def _initialize_base(self,
                    checkpoint_path="",
                    device=None):
        self.checkpoint_path = checkpoint_path
        self.device = device

        if self.checkpoint_path != "":
            try:
                self.restore_model(checkpoint_path)
            except:
                checkpoint = torch.load(checkpoint_path)
                self.model.load_state_dict(checkpoint)

        # Store parameters
        self.model_parameters = list(filter(lambda p: p.requires_grad, self.parameters()))
        self.n_params = sum([np.prod(p.size()) for p in self.model_parameters])

        # Move model to device
        if device is not None:
            self.device = device
            self.model = self.model.to(self.device)

    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        return super().__str__() + '\nTrainable parameters: {}'.format(self.n_params)

    def save_model(self, save_path, epoch=None, optimizer=None):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        if "Wrapper" in type(self).__name__:
            arch = type(self.model).__name__
            model = self.model
        else:
            arch = type(self).__name__
            model = self
        print("arch: {}".format(arch))
        state = {
            'arch': arch,
            'state_dict': model.state_dict(),
        }
        if epoch is not None:
            state.update({'epoch': epoch})
        if optimizer is not None:
            state.update({'optimizer': optimizer})

        torch.save(state, save_path)


    def restore_model(self, restore_path, optimizer=None):
        """
        Restore model from the given restore_path

        Arg(s):
            restore_path : str
                path to checkpoint
            optimizer : torch.optimizer or None
                optimizer to restore or None

        Returns:
            int, torch.optimizer
                epoch and loaded optimizer
        """
        if "Wrapper" in type(self).__name__:
            # arch = type(self.model).__name__
            model = self.model
        else:
            # arch = type(self).__name__
            model = self

        state = torch.load(restore_path)
        if 'arch' in state.keys():
            assert state['arch'] == type(model).__name__, \
                "'arch in config: {} and model.name: {} do not match".format(state['arch'], type(self.model).__name__)

        if 'state_dict' in state:
            # try:
            model.load_state_dict(state['state_dict'])
            # except:
            #     new_state_dict = {}
            #     for key, val in state['state_dict'].items():
            #         new_state_dict[key.split('model.')[1]] = val
            #     self.model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state)
        if optimizer is not None:
            optimizer.load_state_dict(state['optimizer'])

        if 'epoch' in state.keys():
            return state['epoch'], optimizer
        else:
            return None, optimizer

    def get_parameters(self):
        return self.model_parameters

    def get_n_params(self):
        return self.n_params

    def get_checkpoint_path(self):
        return self.checkpoint_path

