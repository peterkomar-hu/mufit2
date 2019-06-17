import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

import sys
sys.path.append('../stan/')
from stan_utilities import get_model


def estimate_center_and_width(od):
    """Estimates the center and width of log-OD in main operating region.

    Args:
        od (1D numpy.array): optical density measurements

    Returns:
        float: center of main log-OD domain
        float: width of main log-OD domain
    """
    
    # Remove non-positive OD
    positive_od = od > 0
    od_pos = od[positive_od]
    
    # Fit mxiture of normal and Cauchy to log-OD distribution
    model = get_model('../stan/width_estimation_model.stan', 
                      '../stan/width_estimation_model.pkl')
    log_OD = np.log(od_pos)
    
    data_dict = {
        'N': len(log_OD),
        'x': log_OD
    }
    mu0 = np.median(log_OD)
    sigma0 = 2*(np.percentile(log_OD, 75) - np.percentile(log_OD, 25))
    fit = model.optimizing(data_dict, init=lambda : 
                              {
                                  'mu': mu0,         # normal mean
                                  'sigma': sigma0,   # normal std
                                  'm': mu0,          # Cauchy center
                                  's': 3 * sigma0,   # Cauchy width
                                  'w': [0.8, 0.2]    # mixing weights [normal, Cauchy]
                              }
                            )
    log_od_center = fit['mu']
    log_od_width = np.sqrt(12) * fit['sigma']  # Assuming uniform distribution
    
    return log_od_center, log_od_width


def clean_data(time, od, log_od_center, log_od_width, width_factor=1.5):
    """Selects data points where OD is not too far from the main domain.

    Args:
        time (1D numpy.array): time points
        od (1D numpy.array): optical density measurements (same shape as `time`)
        log_od_center (float): center of log-OD in main region
        log_od_width (float): width of log-OD distribution in main region
        width_factor (float): width of log-OD acceptance boundary is increased by this factor

    Returns:
        numpy.array: cleaned time points
        numpy.array: cleaned log-OD series
    """

    low_od = np.exp(log_od_center - width_factor * log_od_width / 2)
    high_od = np.exp(log_od_center + width_factor * log_od_width / 2)
    
    selection = (low_od < od) & (od < high_od)
    t = time[selection]
    x = np.log(od[selection])

    return t, x


def find_regions(t, x, minimum_datapoints=10, mu_factor=0.7, low_density_factor=0.01):
    """Finds clean regions of gradual growth between jump events.

    Args:
        t (1D numpy.array): clean time points
        x (1D numpy.array): cleaned log-OD series (same shape as `t`)
        minimum_datapoints (int): regions must have at least this many data points
        mu_factor (float): the linear fit is tempered by this factor before isotonic regression
            Low values (0.0..0.5) can result in missing jump events.
            High values (0.8..1.0) can result in oversegmentation, i.e. false jump events

    Returns:
        numpy.array: list of start indexes for the regions
        numpy.array: list of end indexes (inclusive) for the regions
    """
    # find gaps in the data
    avg_dt = (t[-1] - t[0]) / (len(t) - 1)
    gap_start_idexes = np.where(np.diff(t) > avg_dt / low_density_factor)[0]
    
    # build initial set of regions from these gaps
    s_raw = [0]
    e_raw = []
    for gap_idx in gap_start_idexes:
        e_raw.append(gap_idx)
        s_raw.append(gap_idx + 1)
    e_raw.append(len(t) - 1)
    regions_to_investigate = list(zip(s_raw, e_raw))
    
    s = []
    e = []
    while len(regions_to_investigate) > 0:
        
        # pick a new region
        start_idx, end_idx = regions_to_investigate.pop()
        
        # check that there are at least a minimum number of datapoints
        if end_idx - start_idx + 1 < minimum_datapoints:
            continue
        
        # find optimal drift
        t_region = t[start_idx:end_idx+1]
        x_region = x[start_idx:end_idx+1]
        mu_min = LinearRegression(fit_intercept=True) \
                 .fit(t_region.reshape([-1, 1]), 
                     x_region) \
                 .coef_
    
        # fit monotonic function
        x_drifting = x_region - t_region * mu_min * mu_factor
        iso_reg = IsotonicRegression(increasing=False) \
                  .fit(t_region, x_drifting)
        x_segmented = iso_reg.predict(t_region)
        
        # find jumps
        jump_indexes = np.where(np.diff(x_segmented) < 0)[0] + start_idx
        if len(jump_indexes) > 0:
            # if found, add the sub-regions to the list of new regions
            start_indexes = [start_idx]
            end_indexes = []
            for jump_idx in jump_indexes:
                end_indexes.append(jump_idx)
                start_indexes.append(jump_idx + 1)
            end_indexes.append(end_idx)
            for start_idx, end_idx in zip(start_indexes, end_indexes):
                regions_to_investigate.append((start_idx, end_idx))
        else:
            # if no subregions are found, add regions to final set
            s.append(start_idx)
            e.append(end_idx)

    s.sort()
    e.sort()    
    return np.array(s), np.array(e)
