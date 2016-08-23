'''
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
'''

import numpy as np
import pandas as pd
from matplotlib import mlab
from scipy.optimize import curve_fit


# Boltzmann constant
K_B = 1.38064852e-23 # m2 kg s-2 K-1

def feedback_results_series_list_to_velocity_summary_df(fb_results_series_list, data_path):
    velocity_data_path = data_path.joinpath('velocity_summary_data.csv')

    if velocity_data_path.exists():
        df = pd.read_csv(velocity_data_path, index_col=0)
    else:
        df = pd.DataFrame()

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

    if len(df.index):
        if not velocity_data_path.parent.isdir():
            velocity_data_path.parent.makedirs_p()
        df.to_csv(velocity_data_path)
    return df

def find_outliers(dxdt, f, n, r):
    '''
    Identify outliers in the data, where outliers are defined as
    residuals that are > 2 standard deviations from the mean
    '''
    sat_mask = np.zeros(dxdt.shape, dtype=bool)
    sat_mask[n:] = True
    outliers_mask = np.zeros(dxdt.shape, dtype=bool)
    n_outliers = 0
    while True:
        outliers_mask[~sat_mask] = \
            abs(r[~sat_mask] / np.sqrt(np.mean(r[~sat_mask]**2))) > 2.0
        outliers_mask[sat_mask] = \
            abs(r[sat_mask] / np.sqrt(np.mean(r[sat_mask]**2))) > 2.0

        if len(mlab.find(outliers_mask)) > n_outliers:
            n_outliers = len(mlab.find(outliers_mask))

            # refit (excluding outliers)
            mask = np.logical_and(~sat_mask, ~outliers_mask)
            p, cov = np.polyfit(f[mask], dxdt[mask], 1, cov=True)
            r[~sat_mask] = p[0]*f[~sat_mask] + p[1] - dxdt[~sat_mask]

            mask = np.logical_and(sat_mask, ~outliers_mask)
            p_sat, cov_sat = np.polyfit(f[mask], dxdt[mask], 1, cov=True)
            r[sat_mask] = p_sat[0]*f[sat_mask] + p_sat[1] - dxdt[sat_mask]
        else:
            return outliers_mask

def find_saturation_force(dxdt, f):
    f_sat = []
    # Iterate through the data points by dividing it into 2 data sets
    # pre-satuation and post-saturation and fit a line to each data set
    for n in range(4, len(dxdt)-3):
        sat_mask = np.zeros(dxdt.shape, dtype=bool)
        sat_mask[n:] = True

        # fit the pre-saturation data
        p, cov = np.polyfit(f[~sat_mask], dxdt[~sat_mask], 1, cov=True)
        r = np.zeros(dxdt.shape)
        r[~sat_mask] = p[0] * f[~sat_mask] + p[1] - dxdt[~sat_mask]

        # fit the post-saturation data
        p_sat, cov_sat = np.polyfit(f[sat_mask], dxdt[sat_mask], 1, cov=True)
        r[sat_mask] = p[0] * f[sat_mask] + p[1] - dxdt[sat_mask]

        outliers_mask = find_outliers(dxdt, f, n, r)

        # fit all data (assuming no saturation)
        p_no_sat, cov_no_sat = np.polyfit(f[~outliers_mask], dxdt[~outliers_mask], 1, cov=True)
        r_no_sat = p_no_sat[0] * f + p_no_sat[1] - dxdt

        # The post-saturation data should have a higher k_df
        # (k_df = 1. / p[0]). If the intersection of the two lines is a
        # force > 0, append it to the f_sat list.
        if p_sat[0] < p[0] and p[1] < p_sat[1]:
            f_sat.append((p_sat[1] - p[1]) / (p[0] - p_sat[0]))
        else:
            f_sat.append(np.nan)

    f_sat = np.array(f_sat)
    f_min = np.ma.masked_invalid(np.abs(f_sat - f[3:-4]))

    ind_sat = []
    if len(f_min) and False in f_min.mask:
        ind_sat = mlab.find(f_min == np.min(f_min))

    # if we found a saturation index and the relative difference in k_df
    # is greater than our uncertainty in measuring it
    if len(ind_sat) and (
        np.abs((p[0] - p_sat[0]) / p[0]) >
        np.sqrt(np.diag(cov))[0] / np.abs(p[0])
    ):
        n = ind_sat[0] + 4
        return n, f_sat[ind_sat[0]], find_outliers(dxdt, f, n, r)
    else:
        outliers_mask = np.zeros(dxdt.shape, dtype=bool)
        if len(f_sat):
            outliers_mask = \
                abs(r_no_sat / np.sqrt(np.mean(r_no_sat**2))) > 2.0
        return None, None, outliers_mask

def f_dxdt_mkt(f, f_th, Lambda, k0):
    '''
    Drop velocity as a function of applied force f, threshold force f_th,
    Lambda, and k0 (parameters defined by the Molecular Kinetic Theory).
    '''
    return 2 * k0 * Lambda * np.sinh(Lambda**2 / (K_B * 293) * 1e3 * np.abs(f - f_th)) # T=20C=293K

def f_dxdt_linear(f, f_th, k_df):
    return 1. / k_df * np.abs(f - f_th)

def fit_velocity_vs_force(f, dxdt, nonlin=False, full=False):
    '''
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
    '''

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
            R2 = 1 - np.sum(r**2) / (len(dxdt) * np.var(dxdt))
            perr = (np.nan, np.nan)

            if pcov is not None:
                # calculate uncertainties
                f_th_error = np.sqrt(pcov[0, 0] / p[0]**2 +
                                     pcov[1, 1] / p[1]**2 -
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

        Lambda = np.sqrt((b / 1000 * (K_B * 293))) # m (T=20C=293K)
        k0 = a / (2 * Lambda) # 1/s
        try:
            p, pcov = curve_fit(f_dxdt_mkt, f, dxdt, p0=[f_th, Lambda, k0])
            p = np.abs(p)
            f_th, Lambda, k0 = p
            perr = np.sqrt(np.diag(pcov))
        except:
            p = np.nan * np.ones(3)
            perr = np.nan * np.ones(3)

        if full:
            r = dxdt - f_dxdt_mkt(f, f_th, Lambda, k0)
            R2 = 1 - np.sum(r**2) / (len(dxdt) * np.var(dxdt))
            info = {'perr': perr,
                    'r': r,
                    'R2': R2}
            return p, info
        else:
            return p

def fit_parameters_to_velocity_data(velocity_df, data_path, eft):
    fitted_params_path = data_path.joinpath('fitted_params.csv')

    if fitted_params_path.exists():
        df = pd.read_csv(fitted_params_path, index_col=0)
    else:
        df = pd.DataFrame()

    outliers_path = data_path.joinpath('outliers.csv')
    if outliers_path.exists():
        outliers_df = pd.read_csv(outliers_path, index_col=0)
    else:
        outliers_df = pd.DataFrame()

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

        ind_sat, f_sat, outliers_mask = find_saturation_force(dxdt, f)

        if ind_sat:
            sat_mask = np.zeros(dxdt.shape, dtype=bool)
            sat_mask[ind_sat:] = True

            mask = np.logical_and(sat_mask, ~outliers_mask)
            # fit the post-saturation data
            p_sat, cov_sat = np.polyfit(f[mask], dxdt[mask], 1,
                                        cov=True)

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
        max_sinh_arg = Lambda**2 / (K_B * 293) * 1e3 * np.max(f[mask] - f_th_mkt) # T=20C=293K

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
    # save the outliers mask
    outliers_df.sort_index(inplace=True)
    outliers_df.to_csv(outliers_path)

    # reorder columns and save the fitted parameters
    df = df[[u'step', u'time', u'f_th_linear', u'f_th_linear_error', u'k_df_linear',
             u'k_df_linear_error', u'R2_linear', u'f_sat', u'f_th_mkt', u'f_th_mkt_error',
             u'Lambda', u'Lambda_error', u'k0', u'k0_error', u'R2_mkt', u'max_sinh_arg']]
    df.to_csv(fitted_params_path)
    return df, outliers_df