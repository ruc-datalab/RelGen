import time
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.profiler import record_function
from inspect import isfunction


def sample_data_no_cond(sample_size, diff_model, data_wrapper):
    sta = time.time()
    samples = []
    sample_row = 0
    while sample_row < sample_size:
        sample_batch = int((sample_size - sample_row) * 1.1)
        sample = diff_model.sample(sample_batch, clip_denoised=True)
        sample = sample.numpy()
        sample = data_wrapper.ReverseToOrdi(sample)

        allow_index, _ = data_wrapper.RejectSample(sample)
        sample = sample[allow_index, :]
        samples.append(sample)
        sample_row += sample.shape[0]
    samples = np.concatenate(samples, axis=0)
    samples = samples[:sample_size, :]
    end = time.time()
    print("Sampling time:", end - sta)

    sample_data = data_wrapper.ReverseToCat(samples)
    sample_data = pd.DataFrame(sample_data, columns=data_wrapper.columns)
    sample_data = data_wrapper.ReOrderColumns(sample_data)
    return sample_data


def sample_data_condition(diff_model, data_wrapper, condition):
    sample_start = time.time()
    sample_index = np.arange(len(condition))
    sample_data = np.zeros([len(condition), data_wrapper.raw_dim])

    while len(sample_index) > 0:
        cond_input = condition[sample_index, :]
        cond_tools = (cond_input, 1.0)

        sample = diff_model.sample_all(len(cond_input), 100000, cond_tools=cond_tools, clip_denoised=True)
        sample = sample.cpu().numpy()
        sample = data_wrapper.ReverseToOrdi(sample)

        allow_index, reject_index = data_wrapper.RejectSample(sample)
        sample_allow_index = sample_index[allow_index] if len(allow_index) > 0 else []
        sample_reject_index = sample_index[reject_index] if len(reject_index) > 0 else []

        if len(sample_allow_index) > 0:
            sample_data[sample_allow_index, :] = sample[allow_index, :]
        sample_index = sample_reject_index
    sample_end = time.time()
    print("Sampling time:", sample_end - sample_start)

    sample_data = data_wrapper.ReverseToCat(sample_data)
    sample_data = pd.DataFrame(sample_data, columns=data_wrapper.columns)
    sample_data = data_wrapper.ReOrderColumns(sample_data)
    return sample_data


def sample_data_control(diff_model, data_wrapper, condition, scorer, weight):
    cond_fn = get_cond_fn(scorer, weight)

    sample_start = time.time()
    sample_index = np.arange(len(condition))
    sample_data = np.zeros([len(condition), data_wrapper.raw_dim])

    while len(sample_index) > 0:
        cond_input = condition[sample_index, :]
        control_tools = (cond_input, cond_fn)

        sample = diff_model.sample_all(len(cond_input), batch_size=200000, clip_denoised=False,
                                       control_tools=control_tools, control_t=200)
        sample = sample.cpu().numpy()
        sample = data_wrapper.ReverseToOrdi(sample)

        allow_index, reject_index = data_wrapper.RejectSample(sample)
        sample_allow_index = sample_index[allow_index] if len(allow_index) > 0 else []
        sample_reject_index = sample_index[reject_index] if len(reject_index) > 0 else []

        if len(sample_allow_index) > 0:
            sample_data[sample_allow_index, :] = sample[allow_index, :]
        sample_index = sample_reject_index
    sample_end = time.time()
    print("Sampling time:", sample_end - sample_start)

    sample_data = data_wrapper.ReverseToCat(sample_data)
    sample_data = pd.DataFrame(sample_data, columns=data_wrapper.columns)
    sample_data = data_wrapper.ReOrderColumns(sample_data)
    return sample_data


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
	Compute the KL divergence between two gaussians.
	Shapes are automatically broadcasted, so batches can be compared to
	scalars, among other use cases.
	"""
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
            -1.0
            + logvar2
            - logvar1
            + torch.exp(logvar1 - logvar2)
            + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
	A fast approximation of the cumulative distribution function of the
	standard normal.
	"""
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
	Compute the log-likelihood of a Gaussian distribution discretizing to a
	given image.
	:param x: the target images. It is assumed that this was uint8 values,
			  rescaled to the range [-1, 1].
	:param means: the Gaussian mean Tensor.
	:param log_scales: the Gaussian log stddev Tensor.
	:return: a tensor like x of log probabilities (in nats).
	"""
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs


def sum_except_batch(x, num_dims=1):
    '''
	Sums all dimensions except the first.
	Args:
		x: Tensor, shape (batch_size, ...)
		num_dims: int, number of batch dims (default=1)
	Returns:
		x_sum: Tensor, shape (batch_size,)
	'''
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)


def mean_flat(tensor):
    """
	Take the mean over all non-batch dimensions.
	"""
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def ohe_to_categories(ohe, K):
    K = torch.from_numpy(K)
    indices = torch.cat([torch.zeros((1,)), K.cumsum(dim=0)], dim=0).int().tolist()
    res = []
    for i in range(len(indices) - 1):
        res.append(ohe[:, indices[i]:indices[i + 1]].argmax(dim=1))
    return torch.stack(res, dim=1)


def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def exists(x):
    return x is not None


def extract(a, t, x_shape):
    b, *_ = t.shape
    t = t.to(a.device)
    out = a.gather(-1, t)
    while len(out.shape) < len(x_shape):
        out = out[..., None]
    return out.expand(x_shape)


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def index_to_log_onehot(x, num_classes):
    onehots = []
    for i in range(len(num_classes)):
        onehots.append(F.one_hot(x[:, i], num_classes[i]))

    x_onehot = torch.cat(onehots, dim=1)
    log_onehot = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_onehot


def index_to_onehot(x, num_classes):
    onehots = []
    for i in range(len(num_classes)):
        onehots.append(F.one_hot(x[:, i], num_classes[i]))

    x_onehot = torch.cat(onehots, dim=1)
    # log_onehot = torch.log(x_onehot.float().clamp(min=1e-30))
    return x_onehot


def log_sum_exp_by_classes(x, slices):
    device = x.device
    res = torch.zeros_like(x)
    for ixs in slices:
        res[:, ixs] = torch.logsumexp(x[:, ixs], dim=1, keepdim=True)

    assert x.size() == res.size()

    return res


@torch.jit.script
def log_sub_exp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m = torch.maximum(a, b)
    return torch.log(torch.exp(a - m) - torch.exp(b - m) + 1e-30) + m


def sliced_logsumexp(x, slices):
    log_sum_p = torch.zeros(x.shape[0], len(slices) - 1).to(x.device)
    for i in range(len(slices) - 1):
        log_sum_p[:, i] = torch.logsumexp(x[:, slices[i]:slices[i + 1]], dim=1)

    slice_lse_repeated = torch.repeat_interleave(
        log_sum_p,
        slices[1:] - slices[:-1],
        dim=-1
    )
    return slice_lse_repeated


# @torch.jit.script
# def sliced_logsumexp(x, slices):
#	 lse = torch.logcumsumexp(
#		 torch.nn.functional.pad(x, [1, 0, 0, 0], value=-float('inf')),
#		 dim=-1)

#	 slice_starts = slices[:-1]
#	 slice_ends = slices[1:]

#	 slice_lse = log_sub_exp(lse[:, slice_ends], lse[:, slice_starts])
#	 slice_lse_repeated = torch.repeat_interleave(
#		 slice_lse,
#		 slice_ends - slice_starts, 
#		 dim=-1
#	 )
#	 return slice_lse_repeated

def log_onehot_to_index(log_x):
    return log_x.argmax(1)


class FoundNANsError(BaseException):
    """Found NANs during sampling"""

    def __init__(self, message='Found NANs during sampling.'):
        super(FoundNANsError, self).__init__(message)
