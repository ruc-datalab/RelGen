import torch

from relgen.model import BaseModel
from relgen.utils.enum_type import ModelType


class ARModel(BaseModel):
    """
    This is an abstract ar model. All the ar model should implement this class.
    """

    type = ModelType.AR

    def __init__(self):
        super(ARModel, self).__init__()

    def encode_input(self, data, natural_col=None, out=None):
        """Encodes token IDs.

        Warning: this could take up a significant portion of a forward pass.

        Args:
          data (torch.Long): [batch_size, cols_num] or [batch_size, 1].

          natural_col (int): If specified, 'data' has shape [batch_size, 1] corresponding to col-'natural_col'.
          Otherwise, 'data' corresponds to all cols.

          out (torch.Tensor): If specified, assign results into this Tensor storage.

        Returns:
          torch.Tensor: Encoded input.
        """
        raise NotImplementedError

    def forward(self, inputs, conditions=None):
        """Calculates unnormalized logit and outputs logit for (x0, x1|x0, x2|x0,x1, ...).

        Args:
          inputs (torch.Tensor): AR model inputs for a batch data, shaped [batch_size, cols_num].

          conditions (torch.Tensor): Additional input used in conditional generation, shaped [batch_size].

        Returns:
          torch.Tensor: Logit for (x0, x1|x0, x2|x0,x1, ...).
        """
        raise NotImplementedError

    def forward_with_encoded_input(self, inputs, conditions=None):
        """Calculates unnormalized logit with encoded input and outputs logit for (x0, x1|x0, x2|x0,x1, ...).

        Args:
          inputs (torch.Tensor): AR model encoded inputs for a batch data, shaped [batch_size, cols_num].

          conditions (torch.Tensor): Additional input used in conditional generation, shaped [batch_size].

        Returns:
          torch.Tensor: Logit for (x0, x1|x0, x2|x0,x1, ...).
        """
        raise NotImplementedError

    def logit_for_col(self, idx, logit):
        """Returns the logit (vector) corresponding to log p(x_i | x_(<i)).

        Args:
          idx (int): Index in natural (table) ordering.

          logit: Logit for (x0, x1|x0, x2|x0,x1, ...), shaped [batch_size, ...].

        Returns:
          torch.Tensor: [batch_size, domain size for column idx].
        """
        raise NotImplementedError

    def calculate_loss(self, logit, data, **kwargs):
        """Calculate the training loss for a batch data, given logit (the conditionals) and data.

        Args:
          logit: Logit for (x0, x1|x0, x2|x0,x1, ...), shaped [batch_size, ...].

          data: Training data, shaped [batch_size, cols_num].

          **kwargs: May be condition.

        Returns:
          torch.Tensor: Training loss, shaped [].
        """
        raise NotImplementedError
