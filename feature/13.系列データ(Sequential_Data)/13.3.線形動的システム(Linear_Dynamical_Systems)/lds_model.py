"""One-dimensional linear Gaussian state-space example for PRML 13.3."""
import numpy as np

OBS = np.array([0.1, 0.2, 0.5, 3.5, 0.9], dtype=float)
INITIAL_MEAN = 0.0
INITIAL_VARIANCE = 0.64  # P_0 in the official errata
STATE_VARIANCE = 0.04  # Gamma
SENSOR_VARIANCE = 0.64  # Sigma


def infer(observations=OBS, state_variance=STATE_VARIANCE,
          sensor_variance=SENSOR_VARIANCE):
    """Kalman filter and Rauch-Tung-Striebel smoother with A=C=1."""
    x = np.asarray(observations, dtype=float)
    if x.ndim != 1 or len(x) == 0 or state_variance <= 0 or sensor_variance <= 0:
        raise ValueError('Expected a nonempty scalar sequence and positive variances')
    n = len(x)
    prior_mean = np.empty(n)
    prior_var = np.empty(n)
    gain = np.empty(n)
    mean = np.empty(n)
    var = np.empty(n)
    likelihood = np.empty(n)
    for i, observation in enumerate(x):
        prior_mean[i] = INITIAL_MEAN if i == 0 else mean[i-1]
        prior_var[i] = INITIAL_VARIANCE if i == 0 else var[i-1] + state_variance
        gain[i] = prior_var[i] / (prior_var[i] + sensor_variance)
        mean[i] = prior_mean[i] + gain[i] * (observation - prior_mean[i])
        var[i] = (1 - gain[i]) * prior_var[i]
        innovation_var = prior_var[i] + sensor_variance
        likelihood[i] = np.exp(-0.5*(observation-prior_mean[i])**2/innovation_var) / np.sqrt(2*np.pi*innovation_var)
    smooth_mean = mean.copy()
    smooth_var = var.copy()
    smoothing_gain = np.zeros(n)
    for i in range(n-2, -1, -1):
        smoothing_gain[i] = var[i] / (var[i]+state_variance)
        smooth_mean[i] = mean[i] + smoothing_gain[i]*(smooth_mean[i+1]-mean[i])
        smooth_var[i] = var[i] + smoothing_gain[i]**2*(smooth_var[i+1]-var[i]-state_variance)
    return dict(observations=x, prior_mean=prior_mean, prior_var=prior_var,
                gain=gain, mean=mean, var=var, likelihood=likelihood,
                smooth_mean=smooth_mean, smooth_var=smooth_var,
                smoothing_gain=smoothing_gain)
