#!/usr/bin/env python
"""Fit expression data for soybean flowering network.

Functions:
    flower_ode: Right-hand side of the flowering network ODE.
    get_params: Generate lmfit.Parameters object.
    fit_expression: Optimize the parameters.
    plot_fit: Plot fitting result.
    res_flower_ode: Calculate residual.
    saturation: Saturation on production function.
    read_flower_data: Read expression data from file.
    solve_flower_ode: Solve ODE for flowering network.
    fit_soybean_flower: Fit five-gene model to soybean data.
    fit_soybean_flower_from_pickle: Read optimized parameters
        for data fitting.
    repressilator_ode: Right-hand side of an arbitrary
        repressilator network ODE.
    normalize: Normalize data.

Classes:
    FitArgs: Fitting arguments.
"""
import pandas as pd
import numpy as np
from scipy.integrate import odeint
from lmfit import Parameters, Minimizer
import pickle
# Plotting module.
import sys
if sys.platform == 'darwin':
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
elif sys.platform in ['linux', 'linux2']:
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
else:
    print("No support for Windows.")
    exit(1)


def flower_ode(x, t, param_dict, sat, module_type,
               col1a_activates_e1):
    """Right-hand side function of the ODE-based flowering model.

    Args:
        x: array
            First half are the mRNA concentrations.  Second half
            are the protein concentrations.
        t: float
            Time.
        param_dict: dict
            Parameters.
        sat: str
            Saturation of production rate.
        module_type: array
            Modules types of genes 3 and 5.
        col1a_activates_e1: bool
            COL1a gene activates E1 if True.
            COL1a gene represses E1 if False.

    Returns: array
        Time-derivatives of x.
    """
    v = param_dict
    # Independent binding enhancer.
    if module_type[0] == 0:
        module_effect_3 = (
                (x[5]/v['mm_13'])**v['hill_13']
                / (1+(x[5]/v['mm_13'])**v['hill_13'])
                )*(
                (x[6]/v['mm_23'])**v['hill_23']
                / (1+(x[6]/v['mm_23'])**v['hill_23'])
                )*(
                v['beta_12_3']
                )
    # Complex binding enhancer.
    elif module_type[0] == 1:
        module_effect_3 = (
            (x[5]/v['mm_13'])**v['hill_13']
            *(x[6]/v['mm_23'])**v['hill_23']
            )/(
            1+(x[5]/v['mm_13'])**v['hill_13']
            *(x[6]/v['mm_23'])**v['hill_23']
            )*(
            v['beta_12_3']
            )
    # Two enhancer modules.
    elif module_type[0] == 2:
        module_effect_3 = (
            (x[5]/v['mm_13'])**v['hill_13']
            /(1+(x[5]/v['mm_13'])**v['hill_13'])
            *v['beta_1_3']
            )+(
            (x[6]/v['mm_23'])**v['hill_23']
            /(1+(x[6]/v['mm_23'])**v['hill_23'])
            *v['beta_2_3']
            )
    else:
        raise Exception
    # Independent binding enhancer.
    if module_type[1] == 0:
        module_effect_5 = (
                1/(1+(x[7]/v['mm_35'])**v['hill_35'])
                )*(
                (x[8]/v['mm_45'])**v['hill_45']
                /(1+(x[8]/v['mm_45'])**v['hill_45'])
                )*(
                v['beta_34_5']
                )
    # Independent binding silencer.
    elif module_type[1] == 1:
        module_effect_5 = -(
                (x[7]/v['mm_35'])**v['hill_35']
                /(1+(x[7]/v['mm_35'])**v['hill_35'])
                )*(
                1/(1+(x[8]/v['mm_45'])**v['hill_45'])
                )*(
                v['beta_34_5']
                )
    # Complex binding enhancer.
    elif module_type[1] == 2:
        module_effect_5 = (
            (x[8]/v['mm_45'])**v['hill_45']
            /(
                1 + (x[8]/v['mm_45'])**v['hill_45']
                + (x[7]/v['mm_35'])**v['hill_35']
                *(x[8]/v['mm_45'])**v['hill_45']
                )*(
                v['beta_34_5']
                )
            )
    # Complex binding silencer.
    elif module_type[1] == 3:
        module_effect_5 = -(
            (x[7]/v['mm_35'])**v['hill_35']
            /(
                1 + (x[7]/v['mm_35'])**v['hill_35']
                + (x[7]/v['mm_35'])**v['hill_35']
                *(x[8]/v['mm_45'])**v['hill_45']
                )*(
                v['beta_34_5']
                )
            )
    # Two modules.
    elif module_type[1] == 4:
        module_effect_5 = (
            - (x[7]/v['mm_35'])**v['hill_35']
            /(1+(x[7]/v['mm_35'])**v['hill_35'])
            *v['beta_3_5']
            + (x[8]/v['mm_45'])**v['hill_45']
            /(1+(x[8]/v['mm_45'])**v['hill_45'])
            *v['beta_4_5']
            )
    if col1a_activates_e1:
        beta_sign = 1
    else:
        beta_sign = -1
    return np.asarray([
        saturation(
            v['alpha_1']
            + beta_sign*(x[6]/v['mm_21'])**v['hill_21']
            /(1+(x[6]/v['mm_21'])**v['hill_21'])*v['beta_2_1'],
            sat) - v['delta_1']*x[0],
        saturation(
            v['alpha_2']
            + (x[5]/v['mm_12'])**v['hill_12']
            /(1+(x[5]/v['mm_12'])**v['hill_12'])*v['beta_1_2'],
            sat) - v['delta_2']*x[1],
        saturation(
            v['alpha_3']+module_effect_3, sat
            ) - v['delta_3']*x[2],
        saturation(
            v['alpha_4']
            - 1/(1+(x[6]/v['mm_24'])**v['hill_24'])
            *v['beta_2_4'],
            sat) - v['delta_4']*x[3],
        saturation(
            v['alpha_5']+module_effect_5, sat
            ) - v['delta_5']*x[4],
        v['lambda_1']*(x[0]-x[5]),
        v['lambda_2']*(x[1]-x[6]),
        v['lambda_3']*(x[2]-x[7]),
        v['lambda_4']*(x[3]-x[8]),
        v['lambda_5']*(x[4]-x[9])
        ])


def get_params(hill_dict, span, module_type, ode_func=flower_ode):
    """Get Parameters object.

    Args:
        hill_dict: dict
            Hill coefficients.
        span: float
            Time span.
        module_type: array
            Modules types of genes 3 and 5.
        ode_func: callable, optional
            Right-hand side of the ODE.  The calling signature
            is ode_func(x, t, *args).  Here x is an array and
            t is a scalar.  For flower_ode args are (param_dict,
            sat, module_type, col1a_activates_e1).  For other
            callables args are (param_dict, sat).

    Returns: lmfit.Parameter
        ODE parameters.
    """
    fit_params = Parameters()
    fit_params.add('alpha_1', value=np.random.rand(),
                   min=0, max=1)
    fit_params.add('alpha_2', value=np.random.rand(),
                   min=0, max=1)
    fit_params.add('alpha_3', value=np.random.rand(),
                   min=0, max=1)
    fit_params.add('alpha_4', value=np.random.rand(),
                   min=0, max=1)
    fit_params.add('alpha_5', value=np.random.rand(),
                   min=0, max=1)
    if ode_func == flower_ode:
        fit_params.add('mm_12', value=0.01+0.99*np.random.rand(),
                       min=0.01, max=1)
        fit_params.add('mm_21', value=0.01+0.99*np.random.rand(),
                       min=0.01, max=1)
        fit_params.add('mm_13', value=0.01+0.99*np.random.rand(),
                       min=0.01, max=1)
        fit_params.add('mm_23', value=0.01+0.99*np.random.rand(),
                       min=0.01, max=1)
        fit_params.add('mm_24', value=0.01+0.99*np.random.rand(),
                       min=0.01, max=1)
        fit_params.add('mm_35', value=0.01+0.99*np.random.rand(),
                       min=0.01, max=1)
        fit_params.add('mm_45', value=0.01+0.99*np.random.rand(),
                       min=0.01, max=1)
        fit_params.add('hill_12', value=hill_dict['12'], min=1,
                       max=3, vary=False)
        fit_params.add('hill_21', value=hill_dict['21'], min=1,
                       max=3, vary=False)
        fit_params.add('hill_13', value=hill_dict['13'], min=1,
                       max=3, vary=False)
        fit_params.add('hill_23', value=hill_dict['23'], min=1,
                       max=3, vary=False)
        fit_params.add('hill_24', value=hill_dict['24'], min=1,
                       max=3, vary=False)
        fit_params.add('hill_35', value=hill_dict['35'], min=1,
                       max=3, vary=False)
        fit_params.add('hill_45', value=hill_dict['45'], min=1,
                       max=3, vary=False)
        fit_params.add('beta_1_2', value=np.random.rand(), min=0,
                       max=1)
        fit_params.add('beta_2_1', value=np.random.rand(), min=0,
                       max=1)
        if module_type[0] < 2:
            fit_params.add('beta_12_3', value=np.random.rand(),
                           min=0, max=1)
        else:
            fit_params.add('beta_1_3', value=np.random.rand(),
                           min=0, max=1)
            fit_params.add('beta_2_3', value=np.random.rand(),
                           min=0, max=1)
        fit_params.add('beta_2_4', value=np.random.rand(), min=0,
                       max=1)
        if module_type[1] < 4:
            fit_params.add('beta_34_5', value=np.random.rand(),
                           min=0, max=1)
        else:
            fit_params.add('beta_3_5', value=np.random.rand(),
                           min=0, max=1)
            fit_params.add('beta_4_5', value=np.random.rand(),
                           min=0, max=1)
    elif ode_func == repressilator_ode:
        fit_params.add('mm_12', value=0.01+0.99*np.random.rand(),
                       min=0.01, max=1)
        fit_params.add('mm_25', value=0.01+0.99*np.random.rand(),
                       min=0.01, max=1)
        fit_params.add('mm_54', value=0.01+0.99*np.random.rand(),
                       min=0.01, max=1)
        fit_params.add('mm_43', value=0.01+0.99*np.random.rand(),
                       min=0.01, max=1)
        fit_params.add('mm_31', value=0.01+0.99*np.random.rand(),
                       min=0.01, max=1)
        fit_params.add('hill_12', value=hill_dict['12'], min=1,
                       max=3, vary=False)
        fit_params.add('hill_25', value=hill_dict['25'], min=1,
                       max=3, vary=False)
        fit_params.add('hill_54', value=hill_dict['54'], min=1,
                       max=3, vary=False)
        fit_params.add('hill_43', value=hill_dict['43'], min=1,
                       max=3, vary=False)
        fit_params.add('hill_31', value=hill_dict['31'], min=1,
                       max=3, vary=False)
        fit_params.add('beta_1_2', value=np.random.rand(), min=0,
                       max=1)
        fit_params.add('beta_2_5', value=np.random.rand(), min=0,
                       max=1)
        fit_params.add('beta_5_4', value=np.random.rand(), min=0,
                       max=1)
        fit_params.add('beta_4_3', value=np.random.rand(), min=0,
                       max=1)
        fit_params.add('beta_3_1', value=np.random.rand(), min=0,
                       max=1)
    else:
        raise ValueError
    fit_params.add('delta_1', value=0.2*np.random.rand(), min=0,
                   max=0.2)
    fit_params.add('delta_2', value=0.2*np.random.rand(), min=0,
                   max=0.2)
    fit_params.add('delta_3', value=0.2*np.random.rand(), min=0,
                   max=0.2)
    fit_params.add('delta_4', value=0.2*np.random.rand(), min=0,
                   max=0.2)
    fit_params.add('delta_5', value=0.2*np.random.rand(), min=0,
                   max=0.2)
    fit_params.add('lambda_1', value=0.2*np.random.rand(), min=0,
                   max=0.2)
    fit_params.add('lambda_2', value=0.2*np.random.rand(), min=0,
                   max=0.2)
    fit_params.add('lambda_3', value=0.2*np.random.rand(), min=0,
                   max=0.2)
    fit_params.add('lambda_4', value=0.2*np.random.rand(), min=0,
                   max=0.2)
    fit_params.add('lambda_5', value=0.2*np.random.rand(), min=0,
                   max=0.2)
    fit_params.add('x_1', value=np.random.rand(), min=0, max=1)
    fit_params.add('y_1', value=np.random.rand(), min=0, max=1)
    fit_params.add('x_2', value=np.random.rand(), min=0, max=1)
    fit_params.add('y_2', value=np.random.rand(), min=0, max=1)
    fit_params.add('x_3', value=np.random.rand(), min=0, max=1)
    fit_params.add('y_3', value=np.random.rand(), min=0, max=1)
    fit_params.add('x_4', value=np.random.rand(), min=0, max=1)
    fit_params.add('y_4', value=np.random.rand(), min=0, max=1)
    fit_params.add('x_5', value=np.random.rand(), min=0, max=1)
    fit_params.add('y_5', value=np.random.rand(), min=0, max=1)
    fit_params.add('span', value=span, vary=False)
    return fit_params


def fit_expression(hill_dict, sat, niter, x_data, output, tol,
                   disp, show_legend, span, module_type,
                   col1a_activates_e1, ode_func=flower_ode,
                   show_protein=False):
    """Fit the flowering model to data.

    Args:
        hill_dict: dict
            Dictionary of Hill coefficients.
        sat: str
            Saturation.
        niter: int
            Number of iterations.
        x_data: array
            2D array of mRNA concentration data.
        output: str
            Output filename.
        tol: float
            Tolerance for basinhopping.
        disp: bool
            Display optimization result.
        show_legend: bool
            Show legend.
        span: float
            Time span.
        module_type: array
            Modules types of genes 3 and 5.
        col1a_activates_e1: bool
            COL1a gene activates E1 if True.
            COL1a gene represses E1 if False.
        ode_func: callable, optional
            Right-hand side of the ODE.  The calling signature
            is ode_func(x, t, *args).  Here x is an array and
            t is a scalar.  For flower_ode args are (param_dict,
            sat, module_type, col1a_activates_e1).  For other
            callables args are (param_dict, sat).

        show_protein: bool
            Indicator to show proteins as well as mRNAs.

    Returns: lmfit.minimizer.MinimizerResult
        Display and return optimization result.
    """
    x_data_normalized = normalize(x_data, 'sqrt-quad')
    # Optimize parameters.
    fit_params = get_params(hill_dict, span, module_type, ode_func)
    minner = Minimizer(
        res_flower_ode, fit_params, fcn_args=(
            x_data_normalized, sat, module_type,
            col1a_activates_e1, ode_func
            )
        )
    result = minner.minimize(method='basinhopping', niter=niter,
                             minimizer_kwargs={'tol': tol})
    num_genes = x_data.shape[1]
    plot_fit(num_genes, result, sat, module_type, show_legend,
             disp, output, x_data_normalized, col1a_activates_e1,
             ode_func=ode_func, show_protein=show_protein)
    return result


def plot_fit(num_genes, result, sat, module_type, show_legend,
             disp, output, x_data_normalized, col1a_activates_e1,
             show_protein=False, ode_func=flower_ode):
    """Plot fitting result.

    Args:
        num_genes: int
            Number of genes.
        result: lmfit.minimizer.MinimizerResult
            Optimization result.
        sat: str
            Saturation.
        module_type: array
            Modules types of genes 3 and 5.
        show_legend: bool
            Show legend.
        disp: bool
            Display optimization result.
        output: str
            Output filename.
        x_data_normalized: array
            2D array of normalized mRNA concentration data.
        col1a_activates_e1: bool
            COL1a gene activates E1 if True.
            COL1a gene represses E1 if False.
        show_protein: bool
            Indicator to show proteins as well as mRNAs.
        ode_func: callable, optional
            Right-hand side of the ODE.  The calling signature
            is ode_func(x, t, *args).  Here x is an array and
            t is a scalar.  For flower_ode args are (param_dict,
            sat, module_type, col1a_activates_e1).  For other
            callables args are (param_dict, sat).

    Returns: None
        Plots fitting figure and saves to file.
    """
    fig, ax = plt.subplots()
    x_opt = solve_flower_ode(
        result.params, sat, 101, num_genes, module_type,
        col1a_activates_e1, ode_func
        )
    for i in range(num_genes):
        ax.plot(np.linspace(0, result.params['span'], 101),
                x_opt[:, i],
                label='sim, gene {} mRNA'.format(i+1))
    if show_protein:
        ax.set_prop_cycle(None)
        for i in range(num_genes):
            ax.plot(np.linspace(0, result.params['span'], 101),
                    x_opt[:, i+num_genes], '--',
                    label='sim, gene {} protein'.format(i+1))
    ax.set_prop_cycle(None)
    for i in range(num_genes):
        ax.plot(np.linspace(0, result.params['span'], 7),
                x_data_normalized[:, i], 'o',
                label='data, gene {} mRNA (normalized)'.format(
                    i+1
                    ))
    if show_legend:
        lgd = ax.legend(bbox_to_anchor=(0., 1.02, 1., .102),
                        loc=3, ncol=1, borderaxespad=0.)
    else:
        lgd = None
    if disp:
        print('Number of function evaluations:', result.nfev)
        print('AIC:', result.aic)
        print('Square root of average square of difference:',
              np.sqrt(np.mean(result.residual**2)))
    if output:
        fig.savefig(output, bbox_extra_artists=(lgd,),
                    bbox_inches='tight')


def res_flower_ode(params, x_data, sat, module_type,
                   col1a_activates_e1, ode_func=flower_ode):
    """Residual of ODE solution with respect to data.

    Args:
        param: lmfit.Parameters
            All parameters.
        x_data: array
            2D array of mRNA expression in data.
        sat: str
            Saturation of production rate.
        col1a_activates_e1: bool
            COL1a gene activates E1 if True.
            COL1a gene represses E1 if False.
        ode_func: callable, optional
            Right-hand side of the ODE.  The calling signature
            is ode_func(x, t, *args).  Here x is an array and
            t is a scalar.  For flower_ode args are (param_dict,
            sat, module_type, col1a_activates_e1).  For other
            callables args are (param_dict, sat).

    Returns: array
        The 2-D array of the residual.
    """
    num_times = x_data.shape[0]
    if len(x_data.shape) == 1:
        x_data = np.reshape(x_data, (x_data.shape[0], 1))
        num_genes = 1
    else:
        num_genes = x_data.shape[1]
    sol = solve_flower_ode(
        params.valuesdict(), sat, num_times, num_genes,
        module_type, col1a_activates_e1, ode_func
        )[:, :num_genes]
    return sol-x_data


def saturation(x, sat):
    """The saturation function for production rate.

    Args:
        x: float
            Unsaturated production rate.
        sat: str
            Saturation type.  Can be 'relu', 'truncate' or
            'none'.

    Returns: float
        Saturated production rate.
    """
    if sat == 'truncate':
        return min(max(x, 0), 1)
    elif sat == 'relu':
        return max(x, 0)
    elif sat == 'none':
        return x
    else:
        raise ValueError
        return x


def read_flower_data(photoperiod, temperature, genotype,
                     gene_list, exp_file):
    """Read the 7-day expression data for the flowering network.

    Args:
        photoperiod: str
            Photoperiod condition.  Can be 'LD', 'SD', or 'Sh'.
        temperature: str
            Temperature condition.  Can be '16', '25', or '32'.
        genotype: str
            Genotype.  Can be '1', '2', '3', '4', or '5'.
        gene_list: array
            List of gene IDs.
        exp_file: str
            Expression file.

    Returns: array
        The 2D <time points>-by-<mRNA and protein> array of
        concentrations.
    """
    df_tpm = pd.read_csv(exp_file, index_col=0,
                         header=[0, 1, 2, 3, 4, 5])
    df_tpm.sort_index(axis=1, inplace=True)
    time_points = ['D1', 'D2', 'D3', 'D4', 'II2', 'D6', 'D7']
    flower_exp = np.empty((len(time_points), len(gene_list)))
    for idx_t, time in enumerate(time_points):
        for idx_g, gene in enumerate(gene_list):
            flower_exp[idx_t, idx_g] = np.mean(df_tpm.loc[gene, (
                slice(None), temperature, photoperiod, genotype,
                time
                )])
    return flower_exp


def solve_flower_ode(param_dict, sat, num_times, num_genes,
                     module_type, col1a_activates_e1, ode_func=flower_ode):
    """Time series as solution of the flowering ODE.

    Args:
        param_dict: dict
            All parameters, including the time span.
        sat: str
            Saturation of production rate.
        num_times: int
            Number of time points.
        num_genes: int
            Number of genes.
        col1a_activates_e1: bool
            COL1a gene activates E1 if True.
            COL1a gene represses E1 if False.
        ode_func: callable, optional
            Right-hand side of the ODE.  The calling signature
            is ode_func(x, t, *args).  Here x is an array and
            t is a scalar.  For flower_ode args are (param_dict,
            sat, module_type, col1a_activates_e1).  For other
            callables args are (param_dict, sat).

    Returns: array
        The 2D <time points>-by-<mRNA and protein> array
        of concentrations.
    """
    x_init = []
    for i in range(1, num_genes+1):
        x_init.append(param_dict['x_{}'.format(i)])
    for i in range(1, num_genes+1):
        x_init.append(param_dict['y_{}'.format(i)])
    span = param_dict['span']
    times = np.linspace(0, span, num_times)
    if ode_func == flower_ode:
        x = odeint(ode_func, x_init, times,
                   args=(param_dict, sat, module_type,
                         col1a_activates_e1))
    else:
        x = odeint(ode_func, x_init, times,
                   args=(param_dict, sat))
    return x


def fit_soybean_flower(span, niter, tol, rand_seed,
                       col1a_activates_e1=True,
                       ode_func=flower_ode, show_protein=False,
                       module_type=[0, 0], photoperiod='LD', temperature='25'):
    """Fit the five-gene model to soybean RNA-seq data.

    Args:
        span: float
            Total time span of the 7 time points.
        niter: int
            Number of iterations.
        tol: float
            Tolerance for the basinhopping algorithm.
        rand_seed: int or None
            Random seed for the lmfit optimization.  If
            rand_seed is None, it tries to read data from
            /dev/urandom or the clock.
        col1a_activates_e1: bool, optional
            COL1a gene activates E1 if True.
            COL1a gene represses E1 if False.
            For ode_func=flower_ode only.
        ode_func: callable, optional
            Right-hand side of the ODE.  The calling signature
            is ode_func(x, t, *args).  Here x is an array and
            t is a scalar.  For flower_ode args are (param_dict,
            sat, module_type, col1a_activates_e1).  For other
            callables args are (param_dict, sat).
        show_protein: bool, optional
            Indicator to show proteins as well as mRNAs.
        module_type: array, optional
            Indices of the module types of gene 3 (FT4) and
            gene 5 (AP1a).  For ode_func=flower_ode only.
        photoperiod: str
            Photoperiod.
        temperature: str
            Temperature.

    Returns: None
        Saves the optimization result in pickle file, and
        the figure in EPS.
    """
    np.random.seed(rand_seed)
    args = FitArgs(span, module_type, niter, tol, ode_func,
                   photoperiod, temperature)
    x_data = read_flower_data(
        args.photoperiod, args.temperature, args.genotype,
        args.gene_list, args.exp_file
        )
    result = fit_expression(
        args.hill_dict, args.sat, args.niter, x_data,
        args.output+'.eps', args.tol,
        args.disp, args.show_legend, args.span,
        args.module_type, col1a_activates_e1, ode_func,
        show_protein=show_protein
        )
    pickle.dump({'result': result},
                open(args.output+'.pkl', 'wb'))


class FitArgs:
    """Fitting arguments."""

    def __init__(self, span, module_type, niter, tol,
                 ode_func=flower_ode, photoperiod='LD',
                 temperature='25'):
        """Initialization.

        Args:
            span: float
                Total time span of the 7 time points.
            module_type: array
                Indices of the module types of gene 3 (FT4)
                and gene 5 (AP1a).
            niter: int
                Number of iterations.
            tol: float
                Tolerance for the basinhopping algorithm.
            ode_func: callable, optional
                Right-hand side of the ODE.  The calling signature is
                ode_func(x, t, *args).  Here x is an array and t is a
                scalar.  For flower_ode args are (param_dict, sat,
                module_type, col1a_activates_e1).  For other callables
                args are (param_dict, sat).
            photoperiod: str
                Photoperiod.
            temperature: str
                Temperature.
        """
        if ode_func == flower_ode:
            self.hill_dict = {'12': 2, '21': 2, '13': 2, '23': 2,
                              '24': 2, '35': 2, '45': 2}
        elif ode_func == repressilator_ode:
            self.hill_dict = {'12': 2, '25': 2, '54': 2, '43': 2,
                              '31': 2}
        self.sat = 'relu'
        self.photoperiod = photoperiod
        self.temperature = temperature
        self.genotype = '4'
        self.span = span
        self.module_type = module_type
        self.niter = niter
        self.tol = tol
        self.output = 'flower-p{}-t{}-s{}-m{}{}'.format(
            self.photoperiod, self.temperature, self.span,
            *self.module_type
            )
        self.gene_list = [
            'Glyma.06G207800', 'Glyma.08G255200',
            'Glyma.08G363100',
            'Glyma.16G150700', 'Glyma.16G091300'
            ]
        self.disp = True
        self.show_legend = True
        self.exp_file = 'expression-2017-multicol-flowering.csv'


def fit_soybean_flower_from_pickle(module_type, pickle_file,
                                   col1a_activates_e1,
                                   show_protein,
                                   ode_func=flower_ode,
                                   output_file='', photoperiod='LD', temperature='25'):
    """Fit the five-gene model to soybean RNA-seq data from
    pickle data.

    Note the span is in the parameters of the pickle file.

    Args:
        module_type: array
            Indices of the module types of gene 3 (FT4) and
            gene 5 (AP1a).
        pickle_file: str
            Pickle file that stores the optimization result.
        col1a_activates_e1: bool
            COL1a gene activates E1 if True.
            COL1a gene represses E1 if False.
        show_protein: bool
            Indicator to show proteins as well as mRNAs.
        ode_func: callable, optional
            Right-hand side of the ODE.  The calling signature is
            ode_func(x, t, *args).  Here x is an array and t is a
            scalar.  For flower_ode args are (param_dict, sat,
            module_type, col1a_activates_e1).  For other callables
            args are (param_dict, sat).
        output_file: str, optional
            Output file.
        photoperiod: str
            Photoperiod.
        temperature: str
            Temperature.

    Returns: None
        Plots the data fitting figure.
    """
    data = pickle.load(open(pickle_file, 'rb'))
    span = data['result'].params['span'].value
    # niter and tol are not used, so we set them arbitrarily.
    args = FitArgs(span, module_type, 10, 0.1, ode_func,
                   photoperiod, temperature)
    num_genes = len(args.gene_list)
    x_data = read_flower_data(
        args.photoperiod, args.temperature, args.genotype,
        args.gene_list, args.exp_file
        )
    x_data_normalized = x_data/np.linalg.norm(x_data, axis=0)
    plot_fit(num_genes, data['result'], args.sat, args.module_type,
             args.show_legend, args.disp, output_file,
             x_data_normalized, col1a_activates_e1, show_protein,
             ode_func)


def repressilator_ode(x, t, param_dict, sat):
    """Right-hand side function of an arbitrary repressilator model.

    Args:
        x: array
            First half are the mRNA concentrations.  Second half
            are the protein concentrations.
        t: float
            Time.
        param_dict: dict
            Parameters.
        sat: str
            Saturation of production rate.

    Returns: array
        Time-derivatives of x.
    """
    v = param_dict    
    return np.asarray([
        saturation(
            v['alpha_1']
            +1/(1+(x[7]/v['mm_31'])**v['hill_31'])*v['beta_3_1'],
            sat) - v['delta_1']*x[0],
        saturation(
            v['alpha_2']
            +1/(1+(x[5]/v['mm_12'])**v['hill_12'])*v['beta_1_2'],
            sat) - v['delta_2']*x[1],
        saturation(
            v['alpha_3']
            +1/(1+(x[8]/v['mm_43'])**v['hill_43'])*v['beta_4_3'],
            sat) - v['delta_3']*x[2],
        saturation(
            v['alpha_4']
            +1/(1+(x[9]/v['mm_54'])**v['hill_54'])*v['beta_5_4'],
            sat) - v['delta_4']*x[3],
        saturation(
            v['alpha_5']
            +1/(1+(x[6]/v['mm_25'])**v['hill_25'])*v['beta_2_5'],
            sat) - v['delta_5']*x[4],
        v['lambda_1']*(x[0]-x[5]),
        v['lambda_2']*(x[1]-x[6]),
        v['lambda_3']*(x[2]-x[7]),
        v['lambda_4']*(x[3]-x[8]),
        v['lambda_5']*(x[4]-x[9])
        ])


def normalize(data, method='two-norm-no-rescale'):
    """Normalize data.

    Args:
        data: array
            2-D array of data whose columns are to be normalized.
        method: str, optional
            Normalization method.  Can be 'two-norm-no-rescale',
            'two-norm', 'sqrt-arit', or 'sqrt-quad'.

    Returns: array
        Normalized array of the same shape.
    """
    if method == 'two-norm-no-rescale':
        return data/np.linalg.norm(data, axis=0)
    if method == 'two-norm':
        data_normalized_by_gene = data/np.linalg.norm(data,
                                                      axis=0)
    elif method == 'sqrt-arit':
        data_normalized_by_gene = data/np.sqrt(
            np.mean(np.abs(data), axis=0)
            )
    elif method == 'sqrt-quad':
        data_normalized_by_gene = data/np.sqrt(
            np.sqrt(np.mean(data*data, axis=0))
            )
    else:
        raise
    data_normalized_global = (
        data_normalized_by_gene/1.2
        /np.abs(data_normalized_by_gene).max()
        )
    return data_normalized_global
