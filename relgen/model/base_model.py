from logging import getLogger

import numpy as np
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """Base class for all models"""

    def __init__(self):
        self.logger = getLogger()
        super(BaseModel, self).__init__()

    def calculate_loss(self, x, **kwargs):
        """Calculate the training loss for a batch data.

        Args:
            x (torch.Tensor): Model outputs for a batch data, shaped [batch_size].
            **kwargs: May be condition.

        Returns:
            torch.Tensor: Training loss, shaped [].
        """
        raise NotImplementedError

    def forward(self, inputs, conditions=None):
        """Model forward.

        Args:
            inputs (torch.Tensor): Model inputs for a batch data, shaped [batch_size].
            conditions (torch.Tensor): Additional input used in conditional generation, shaped [batch_size].

        Returns:
            torch.Tensor: Model outputs for a batch data, shaped [batch_size].
        """
        raise NotImplementedError

    def sample(self, sample_num, condition=None):
        """Model sample.

        Args:
            sample_num (int): The number of data to be sampled from Model.
            condition (torch.Tensor): Additional input used in conditional generation, shaped [batch_size].

        Returns:
            torch.Tensor: The data sampled from model, shaped [sample_num].
        """
        raise NotImplementedError

    def other_parameter(self):
        if hasattr(self, "other_parameter_name"):
            return {key: getattr(self, key) for key in self.other_parameter_name}
        return dict()

    def load_other_parameter(self, para):
        if para is None:
            return
        for key, value in para.items():
            setattr(self, key, value)

    def __str__(self):
        """
        model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return (
                super().__str__()
                + f"\nTrainable parameters: {params}"
        )
