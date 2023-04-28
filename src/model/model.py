import sys, os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

sys.path.insert(0, 'src')
from model.base_model import BaseModel


class LeNetModel(BaseModel):
    def __init__(self,
                 num_classes=10,
                 checkpoint_path="",
                 device=None):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

        self.features = [self.conv1, self.conv2, self.conv2_drop]

        self._initialize_base(
            checkpoint_path=checkpoint_path,
            device=device)


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class WrapperModel(BaseModel):
    '''
    Simple model wrapper for models in external_code/PyTorch_CIFAR10/cifar10_models/state_dicts

    Arg(s):
        type : str
            Name of architecture, must be key in self.all_classifiers

    '''
    def __init__(self,
                 type,
                 checkpoint_path="",
                 device=None):
        super().__init__()
        self.all_classifiers = {

        }
        if type not in self.all_classifiers:
            raise ValueError("Architecture {} not available for pretrained CIFAR-10 models".format(type))
        self.model = self.all_classifiers[type]
        # self.softmax = torch.nn.Softmax(dim=1)

        self._initialize(
            checkpoint_path=checkpoint_path,
            device=device)

        '''
        # MOVED TO BASE_MODEL
        # Restore weights if checkpoint_path is valid
        # self.checkpoint_path = checkpoint_path

        # if self.checkpoint_path != "":
        #     try:
        #         self.restore_model(checkpoint_path)
        #     except:
        #         checkpoint = torch.load(checkpoint_path)
        #         self.model.load_state_dict(checkpoint)

        # Store parameters
        self.model_parameters = list(filter(lambda p: p.requires_grad, self.parameters()))
        self.n_params = sum([np.prod(p.size()) for p in self.model_parameters])

        # Move model to device
        if device is not None:
            self.device = device
            self.model = self.model.to(self.device)
        '''
    def forward(self, x):
        self.logits = self.model(x)
        return self.logits
