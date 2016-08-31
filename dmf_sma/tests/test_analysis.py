import pandas as pd
import numpy as np
from path_helpers import path
from dmf_sma.analysis import fit_parameters_to_velocity_data


def test_find_outliers():
    velocity_df = pd.read_csv(path(__file__).parent / path('data_0/velocity_summary_data.csv'), index_col=0)
    fit_df, outliers_df = fit_parameters_to_velocity_data(velocity_df, eft=0.3, cache_path=None)
    ref_outliers_df = pd.read_csv(path(__file__).parent / path('data_0/outliers.csv'), index_col=0)

    # check that the outlier masks match
    assert(np.all(outliers_df == ref_outliers_df))


def test_fit_parameters_to_velocity_data():
    velocity_df = pd.read_csv(path(__file__).parent / path('data_0/velocity_summary_data.csv'), index_col=0)
    fit_df, outliers_df = fit_parameters_to_velocity_data(velocity_df, eft=0.3, cache_path=None)
    ref_df = pd.read_csv(path(__file__).parent / path('data_0/fitted_params.csv'), index_col=0)

    test_dict = {}
    test_dict['f_th_linear'] = 1e-10
    test_dict['f_th_linear_error'] = 1e-10
    test_dict['k_df_linear'] = 1e-10
    test_dict['k_df_linear_error'] = 1e-10
    test_dict['f_sat'] = 1e-10
    test_dict['f_th_post_sat'] = 1e-10
    test_dict['f_th_post_sat_error'] = 1e-10
    test_dict['k_df_post_sat'] = 1e-10
    test_dict['k_df_post_sat_error'] = 1e-10
    test_dict['R2_post_sat'] = 1e-10
    test_dict['R2_mkt'] = 1e-10
    test_dict['f_th_mkt'] = 1e-6
    test_dict['f_th_mkt_error'] = 1e-6
    test_dict['Lambda'] = 1e-4
    test_dict['Lambda_error'] = 1e-4
    test_dict['k0'] = 1e-4
    test_dict['k0_error'] = 1e-4
    test_dict['max_sinh_arg'] = 1e-4

    for name, tol in test_dict.items():
        yield _check_param, fit_df, ref_df, name, tol


def _check_param(fit_df, ref_df, name, tol):
    nan_mask = np.isnan(ref_df[name])
    relative_diff = np.max(np.abs((fit_df.loc[~nan_mask, name] -
                                   ref_df.loc[~nan_mask, name]) /
                                  ref_df.loc[~nan_mask, name]))

    print 'relative difference in param="%s" is %e (tol=%e)' % (name, relative_diff, tol)
    assert(relative_diff < tol)

    # check that any values that were NaN in the reference are also NaN in the
    # new fit
    assert(np.all(np.isnan(fit_df.loc[nan_mask, name])))
