from typing import Tuple

from numpy.random import choice
from torch import Tensor, cat, device, randperm, tensor
from torch.nn.functional import unfold

KEYS = {"aa_keys": "WYFMILVAGPSTCEDQNHRK*", "ss_keys": "CSTIGHBE*"}


def slide_window(input_vals: Tensor, w: int) -> Tensor:
    """Slide a window of w amino acids over the input sequence."""
    kernel_size = (input_vals.shape[1], w)
    return unfold(input_vals.float().T[None, None, :, :], kernel_size=kernel_size)[0].T


def ndx_to_seq(seq: Tensor, keys: str = "aa_keys") -> str:
    """Convert a sequence of indices to a sequence of amino acids."""
    # TODO use numpy array instead of torch tensor
    if not isinstance(seq, Tensor):
        seq = tensor(seq)
    return "".join([KEYS[keys][i] for i in seq.long()])


def split_input_vals(input_vals: Tensor, input_keys: str, w: int) -> Tuple[Tensor, Tensor]:
    target_vals_ss = None
    target_vals_cl = None
    if len(input_keys) == 3:
        if input_keys[1] == "S":
            target_vals_ss = input_vals[:, w:-w].long()
        elif input_keys[2] == "S":
            target_vals_ss = input_vals[:, -w:].long()
        if input_keys[1] == "C":
            target_vals_cl = input_vals[:, w:-w].mean(axis=1).reshape((-1, 1))
            target_vals_cl = cat((target_vals_cl, 1 - target_vals_cl), 1).float()
        elif input_keys[2] == "C":
            target_vals_cl = input_vals[:, -w:].mean(axis=1).reshape((-1, 1))
            target_vals_cl = cat((target_vals_cl, 1 - target_vals_cl), 1).float()
    return target_vals_ss, target_vals_cl


def add_noise(one_hot_input: Tensor, input_noise: float, device: device) -> Tensor:
    """Add noise to a tensor of one-hot vectors."""
    ndx = randperm(one_hot_input.shape[1])
    size = list(one_hot_input.shape)
    size[-1] = 1
    p = tensor(choice([1, 0], p=[input_noise, 1 - input_noise], size=size)).to(device)
    return (one_hot_input[:, ndx, :] * p) + (one_hot_input * (1 - p))
