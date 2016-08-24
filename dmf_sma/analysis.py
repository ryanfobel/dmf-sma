"""
Copyright 2014-2016 Ryan Fobel

This file is part of liquid_calibration_plugin.

test_plugin is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

dmf_control_board is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with test_plugin.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import pandas as pd
from matplotlib import mlab
from scipy.optimize import curve_fit
import pkg_resources
from path_helpers import path
import yaml
import arrow

from . import PACKAGE_NAME

# Boltzmann constant
K_B = 1.38064852e-23  # m2 kg s-2 K-1
CACHE_METADATA_FILE_NAME = '.cache-info'


def check_cache(cache_path):
    cache_info_path = path(cache_path.joinpath(CACHE_METADATA_FILE_NAME))
    if cache_info_path.exists():
        return yaml.load(cache_info_path.bytes())
    else:
        return None


def write_cache_info(cache_path):
    cache_info_path = path(cache_path.joinpath(CACHE_METADATA_FILE_NAME))
    version = pkg_resources.get_distribution(PACKAGE_NAME).version
    cache_info = {'package_name': PACKAGE_NAME, 'version': version, 'timestamp': str(arrow.utcnow())}
    cache_info_path.write_bytes(yaml.dump(cache_info))


def feedback_results_series_list_to_velocity_summary_df(fb_results_series_list, cache_path=None):
    df = pd.DataFrame()
    if cache_path:
        velocity_cache_path = cache_path.joinpath('velocity_summary_data.csv')
        if velocity_cache_path.exists():
            df = pd.read_csv(velocity_cache_path, index_col=0)

    for (step_time, step_number, results) in fb_results_series_list:
        # skip steps that are already in the dataframe (as determined by their unique time index)
        # Note that we can't just compare times due to precision (i.e., roundoff); therefore,
        # we consider time indices within 1 us as being the same.
        if 'time' in df.columns and np.any(np.abs(step_time - df['time']) < 1e-6):
            continue

        c_drop = results.data[0].calibration.c_drop(results.frequency[0])

        d = {'step': [],
             'time': [],
             'channel': [],
             'voltage': [],
             'force': [],
             'frequency': [],
             'mean velocity': [],
             'peak velocity': [],
             'c_drop': [],
             'area': [],
             'window size': [],
             'dx': [],
             'dt': [],
             }

        for j, data in enumerate(results.data):
            L = np.sqrt(data.area)
            d['step'] = step_number
            d['time'] = step_time
            d['channel'].append(results.x[j])
            d['voltage'].append(np.mean(data.V_actuation()))
            d['force'].append(np.mean(data.force(Ly=1.0)))
            d['frequency'].append(data.frequency)
            d['area'].append(data.area)

            window_size = data._get_window_size()
            d['window size'].append(window_size)

            velocity_results = data.mean_velocity(Lx=L)
            mean_velocity = None
            peak_velocity = None
            dx = 0
            dt = 0
            if velocity_results:
                mean_velocity = velocity_results['p'][0]

                dx = velocity_results['dx']
                dt = velocity_results['dt']

                order = None
                if window_size and window_size < len(data.time) / 2 and window_size > 3:
                    order = 3

                t, dxdt = data.dxdt(filter_order=order, Lx=L)
                dxdt = np.ma.masked_invalid(dxdt)
                peak_velocity = np.max(dxdt)

            d['dx'].append(dx)
            d['dt'].append(dt)
            d['mean velocity'].append(mean_velocity)
            d['peak velocity'].append(peak_velocity)
            d['c_drop'].append(c_drop)
        df = df.append(pd.DataFrame(d), ignore_index=True)

    if len(df.index) and cache_path:
        if not velocity_cache_path.parent.isdir():
            velocity_cache_path.parent.makedirs_p()
        df.to_csv(velocity_cache_path)

        # update the cache info
        write_cache_info(cache_path)
    return df


def fit_velocity_vs_force_and_find_outliers(dxdt, f):
    """
    Identify outliers in the data, where outliers are defined as
    residuals that are > 2 standard deviations from the mean.

    Parameters
    ----------
      dxdt (np.array):  drop velocity
      f (np.array):     applied electrostatic force
    """

    outliers_mask = np.zeros(dxdt.shape, dtype=bool)
    n_outliers = 0
    p, info = fit_velocity_vs_force(f, dxdt, full=True)

    while True:
        # calculate the new outliers mask
        outliers_mask[~outliers_mask] = abs(info['r'] / np.sqrt(np.mean(info['r'] ** 2))) > 2.0

        # If the number of outliers has increased since the previous iteration,
        # re-fit the data and recalculate the outlier mask
        if len(mlab.find(outliers_mask)) > n_outliers:
            n_outliers = len(mlab.find(outliers_mask))

            # re-fit the data (excluding outliers)
            p, info = fit_velocity_vs_force(f[~outliers_mask], dxdt[~outliers_mask], full=True)
        else:
            return p, info, outliers_mask


def find_saturation_force(dxdt, f):
    """
    Iterate through the data points by dividing it into 2 data sets
    (i.e., pre-satuation and post-saturation) and fit a line to each data set.
    Define the saturation force as the intersection between these two lines.
    """
    f_sat = []

    for n in range(4, len(dxdt) - 3):
        pre_sat_mask = np.zeros(dxdt.shape, dtype=bool)
        pre_sat_mask[:n] = True

        post_sat_mask = np.zeros(dxdt.shape, dtype=bool)
        post_sat_mask[n:] = True

        p_pre, info_pre, outliers_mask_pre = \
            fit_velocity_vs_force_and_find_outliers(dxdt[pre_sat_mask], f[pre_sat_mask])
        f_th_pre, k_df_pre = p_pre
        f_th_error_pre, k_df_error_pre = info_pre['perr']

        p_post, info_post, outliers_mask_post = \
            fit_velocity_vs_force_and_find_outliers(dxdt[post_sat_mask], f[post_sat_mask])
        f_th_post, k_df_post = p_post
        f_th_error_post, k_df_error_post = info_post['perr']

        # find the intersection of the 2 lines
        f_int = (f_th_pre * k_df_post - f_th_post * k_df_pre) / (k_df_post - k_df_pre)

        if (f_int > 0 and k_df_pre > 0 and (k_df_post > 0 and k_df_post > k_df_pre) or
                    k_df_post < 0):
            # We expect the intersection of the line (i.e., f_sat) to be > 0 and k_df > 0.
            # k_df_post should be greater than k_df_pre (i.e., more friction) if both are positive.
            # k_df_post may also be negative if drops are slowing down with increasing force.
            # If these conditions are met, append the intersection to the f_sat list.
            f_sat.append(f_int)
        else:
            f_sat.append(np.nan)

    # Each entry in the calculated f_sat array represents the intersection
    # point of two lines (fit to the pre- and post-saturation data). Find
    # the value in this array that is closest the to the force at the index
    # n used in it's computation (i.e., the index used to divide the dataset
    # into pre- and post-saturation values).
    ind_sat = []
    f_sat = np.array(f_sat)
    diff = np.ma.masked_invalid(np.abs(f_sat - f[3:-4]))
    if len(diff) and False in diff.mask:
        ind_sat = mlab.find(diff == np.min(diff))

    # Re-fit the post-saturation data according to this new saturation value
    if len(ind_sat):
        pre_sat_mask = np.zeros(dxdt.shape, dtype=bool)
        pre_sat_mask[:ind_sat[0]] = True

        post_sat_mask = np.zeros(dxdt.shape, dtype=bool)
        post_sat_mask[ind_sat[0]:] = True

        p_pre, info_pre, outliers_mask_pre = \
            fit_velocity_vs_force_and_find_outliers(dxdt[pre_sat_mask], f[pre_sat_mask])
        f_th_pre, k_df_pre = p_pre
        f_th_error_pre, k_df_error_pre = info_pre['perr']

        p_post, info_post, outliers_mask_post = \
            fit_velocity_vs_force_and_find_outliers(dxdt[post_sat_mask], f[post_sat_mask])
        f_th_post, k_df_post = p_post
        f_th_error_post, k_df_error_post = info_post['perr']

    # if we found a saturation index
    if len(ind_sat):
        n = ind_sat[0] + 4
        outliers_mask = np.zeros(dxdt.shape, dtype=bool)
        outliers_mask[pre_sat_mask] = outliers_mask_pre
        outliers_mask[post_sat_mask] = outliers_mask_post
        return n, f_sat[ind_sat[0]], outliers_mask
    else:
        # fit all data (assuming no saturation)
        p_no_sat, info_no_sat, outliers_mask_no_sat = \
            fit_velocity_vs_force_and_find_outliers(dxdt, f)
        return None, None, outliers_mask_no_sat


def f_dxdt_mkt(f, f_th, Lambda, k0):
    """
    Drop velocity as a function of applied force f, threshold force f_th,
    Lambda, and k0 (parameters defined by the Molecular Kinetic Theory).
    """
    return 2 * k0 * Lambda * np.sinh(Lambda ** 2 / (K_B * 293) * 1e3 * np.abs(f - f_th))  # T=20C=293K


def f_dxdt_linear(f, f_th, k_df):
    return 1. / k_df * (f - f_th)


def fit_velocity_vs_force(f, dxdt, nonlin=False, full=False):
    """
    Fit velocity vs force data.

    Parameters:
        f: array of floats
            Applied forces.
        dxdt: array of floats
            Drop velocities.
        nonlin: bool, optional
            If true, apply a nonlinear (hyperbolic sine function) according to the
            Molecular Kinetic Theory.
        full : bool, optional
            Switch determining nature of the return value. When it is False (default)
            just the fitted parameters are returned; when True, a dictionary is also
            returned containing additional info.

    Returns:
        p: array of fit paramters (f_th, Lambda, k0) if nonlin is True,
            (f_th, k_df) if nonlin is False.
        info: Dictionary containing error estimates, residuals, and R^2 values are returned.
            (only returned if full is set to True).
    """

    pcov = None
    try:
        p, pcov = np.polyfit(f, dxdt, 1, cov=True)
    except ValueError:
        p = np.polyfit(f, dxdt, 1)
    f_th = -p[1] / p[0]
    k_df = 1.0 / p[0]

    if not nonlin:
        if full:
            r = dxdt - f_dxdt_linear(f, f_th, k_df)
            R2 = 1 - np.sum(r ** 2) / (len(dxdt) * np.var(dxdt))
            perr = (np.nan, np.nan)

            if pcov is not None:
                # calculate uncertainties
                f_th_error = np.sqrt(pcov[0, 0] / p[0] ** 2 +
                                     pcov[1, 1] / p[1] ** 2 -
                                     2 * pcov[0, 1] / np.prod(p)) * f_th
                k_df_error = np.sqrt(np.diag(pcov))[0] / np.abs(p[0]) * k_df
                perr = (f_th_error, k_df_error)

            info = {'perr': perr,
                    'r': r,
                    'R2': R2}
            return (f_th, k_df), info
        else:
            return (f_th, k_df)
    else:
        # initial guesses for a and b
        b = 1.0 / np.max(f - f_th)
        a = (1 / k_df) / b

        Lambda = np.sqrt((b / 1000 * (K_B * 293)))  # m (T=20C=293K)
        k0 = a / (2 * Lambda)  # 1/s
        try:
            p, pcov = curve_fit(f_dxdt_mkt, f, dxdt, p0=[f_th, Lambda, k0])
            p = np.abs(p)
            f_th, Lambda, k0 = p
            perr = np.sqrt(np.diag(pcov))
            r = dxdt - f_dxdt_mkt(f, f_th, Lambda, k0)
            R2 = 1 - np.sum(r ** 2) / (len(dxdt) * np.var(dxdt))
        except:
            p = np.nan * np.ones(3)
            perr = np.nan * np.ones(3)
            r = np.nan * np.ones(len(dxdt))
            R2 = np.nan

        if full:
            info = {'perr': perr,
                    'r': r,
                    'R2': R2}
            return p, info
        else:
            return p


def fit_parameters_to_velocity_data(velocity_df, eft, cache_path=None):
    df = pd.DataFrame()
    outliers_df = pd.DataFrame()

    if cache_path:
        fitted_params_path = cache_path.joinpath('fitted_params.csv')
        if fitted_params_path.exists():
            df = pd.read_csv(fitted_params_path, index_col=0)

        outliers_path = cache_path.joinpath('outliers.csv')
        if outliers_path.exists():
            outliers_df = pd.read_csv(outliers_path, index_col=0)

    for (step_time, step), group in velocity_df.groupby(['time', 'step']):

        # if a row with this time already exists, skip it
        if ('time' in df.columns and
            len(df[df['time'] == step_time])):
            continue

        L = np.sqrt(group['area'].values)
        dx = group['dx'].values
        include_mask = dx > L * eft

        f = group['force'].values[include_mask]
        dxdt = np.array(group['peak velocity'].values,
                        dtype=np.float)[include_mask]

        # if there's not enough data to fit, continue
        if len(dxdt) < 2:
            continue

        ind_sat, f_sat, outliers_mask = find_saturation_force(dxdt, f)

        if ind_sat:
            sat_mask = np.zeros(dxdt.shape, dtype=bool)
            sat_mask[ind_sat:] = True
            mask = np.logical_and(~sat_mask, ~outliers_mask)
        else:
            mask = ~outliers_mask

        # if there's not enough data to fit, continue
        if len(mask) < 2:
            continue

        # fit the pre-saturation data
        p, info = fit_velocity_vs_force(f[mask], dxdt[mask], full=True)

        # calculate the parameter estimates and uncertainties
        f_th, k_df = p
        f_th_error, k_df_error = info['perr']

        # fit the MKT model to the pre-saturation data
        p_mkt, info_mkt = fit_velocity_vs_force(f[mask], dxdt[mask], nonlin=True, full=True)

        # calculate the parameter estimates and uncertainties
        f_th_mkt, Lambda, k0 = p_mkt
        f_th_mkt_error, Lambda_error, k0_error = info_mkt['perr']
        max_sinh_arg = Lambda ** 2 / (K_B * 293) * 1e3 * np.max(f[mask] - f_th_mkt)  # T=20C=293K

        df = df.append({
            'step': step,
            'time': step_time,
            'f_th_linear': f_th,
            'f_th_linear_error': f_th_error,
            'k_df_linear': k_df * 1e3,
            'k_df_linear_error': k_df_error * 1e3,
            'R2_linear': info['R2'],
            'f_sat': f_sat,
            'f_th_mkt': f_th_mkt,
            'f_th_mkt_error': f_th_mkt_error,
            'Lambda': Lambda,
            'Lambda_error': Lambda_error,
            'k0': k0,
            'k0_error': k0_error,
            'R2_mkt': info_mkt['R2'],
            'max_sinh_arg': max_sinh_arg,
        }, ignore_index=True)

        try:
            outliers_df.loc[group.index[include_mask], 'outlier'] = outliers_mask
        except KeyError:
            outliers_df = outliers_df.append(pd.DataFrame(data={'outlier': outliers_mask},
                                                          index=group.index[include_mask]))
    if cache_path:
        # save the outliers mask
        outliers_df.sort_index(inplace=True)
        outliers_df.to_csv(outliers_path)

        # reorder columns and save the fitted parameters
        df = df[[u'step', u'time', u'f_th_linear', u'f_th_linear_error', u'k_df_linear',
                 u'k_df_linear_error', u'R2_linear', u'f_sat', u'f_th_mkt', u'f_th_mkt_error',
                 u'Lambda', u'Lambda_error', u'k0', u'k0_error', u'R2_mkt', u'max_sinh_arg']]
        df.to_csv(fitted_params_path)

    return df, outliers_df
