from torch import Tensor
from torch.nn import Module
from torch.nn.functional import one_hot

from PLSx.autoencoder.architecture import Architecture
from PLSx.utils.seq_utils import add_noise, slide_window, split_input_vals


class Autoencoder(Module):
    """Autoencoder class"""

    def __init__(self, architecture: Architecture):
        """Initialize the Autoencoder."""
        super(Autoencoder, self).__init__()
        # common attributes
        self.architecture = architecture
        self.d0 = architecture.d0
        self.d1 = architecture.d1
        self.dn = architecture.dn
        self.w = architecture.w
        #
        self.ds = architecture.ds

    def transform_input(self, input_vals, device, input_noise=0.0, input_keys="A--"):
        # input_keys = "A--" : sequence, "AC-" sequence:class, "AS-" sequence:ss, "ACS" seq:class:ss, "
        # scans by sliding window of w
        assert isinstance(input_vals, Tensor), f"expected Tensor type, received {type(input_vals)}"
        input_vals = slide_window(input_vals, self.w)
        input_ndx = input_vals[:, : self.w].long()
        target_vals_ss, target_vals_cl = split_input_vals(input_vals, input_keys, self.w)
        # one-hot vec input
        one_hot_input = one_hot(input_ndx, num_classes=self.d0) * 1.0
        if input_noise > 0.0:
            one_hot_input = add_noise(one_hot_input, input_noise, device)
        return input_ndx, target_vals_ss, target_vals_cl, one_hot_input

    def forward(self, one_hot_input):
        """Forward pass."""
        assert isinstance(
            one_hot_input, Tensor
        ), f"expected Tensor type, received {type(one_hot_input)}"
        return self.architecture.forward(one_hot_input)

    def train(self, input_vals: Tensor, device, input_noise=0.0, input_keys="A--", **kwargs):
        """Train the Autoencoder."""
        # transform input
        input_ndx, target_vals_ss, target_vals_cl, one_hot_input = self.transform_input(
            input_vals, device, input_noise, input_keys
        )
        kwargs["input_ndx"] = input_ndx
        kwargs["target_vals_ss"] = target_vals_ss
        kwargs["target_vals_cl"] = target_vals_cl
        kwargs["one_hot_input"] = one_hot_input
        kwargs["device"] = device
        kwargs["input_keys"] = input_keys
        self.architecture.train(**kwargs)
