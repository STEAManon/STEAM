from generation import *
from metrics import *

from synthcity.metrics.eval_statistical import AlphaPrecision, InverseKLDivergence, MaximumMeanDiscrepancy, KolmogorovSmirnovTest, JensenShannonDistance, WassersteinDistance
import matplotlib.pyplot as plt
from synthcity.plugins.core.dataloader import GenericDataLoader
import random
from sklearn.metrics import mean_squared_error
from catenets.models.torch import *
from catenets.experiment_utils.simulation_utils import simulate_treatment_setup
from sklearn.linear_model import LogisticRegression
import numpy as np
import torch
import pandas as pd
from typing import Optional, Union
from scipy.special import expit

def seq_test(real, gen, treatment_col, outcome_col, n_iter, private=False, epsilon=None, delta=None, binary_y=False, save=False, fp=''):
    results = pd.DataFrame(columns = ['method', 'p_alpha_x', 'r_beta_x', 'jsd_pi', 'u_pehe'])
    for _ in range(n_iter):
        stand = generate_standard(real, gen, private=private, epsilon=epsilon, delta=delta)
        steam = generate_sequentially(real, gen, treatment_col, outcome_col, private=private, epsilon=epsilon, delta=delta,binary_y=binary_y)
        n_units = len(real.drop([treatment_col, outcome_col], axis=1).columns)

        results.loc[len(results)] = ['standard', evaluate_f(real, stand, treatment_col, outcome_col), evaluate_c(real, stand, treatment_col, outcome_col), 
                                     evaluate_jsd(real, stand, treatment_col, outcome_col), evaluate_average_u_pehe(real, stand, treatment_col, outcome_col, n_units,binary_y)]
        results.loc[len(results)] = ['STEAM', evaluate_f(real, steam, treatment_col, outcome_col), evaluate_c(real, steam, treatment_col, outcome_col), 
                                     evaluate_jsd(real, steam, treatment_col, outcome_col), evaluate_average_u_pehe(real, steam, treatment_col, outcome_col, n_units,binary_y)]
        if save:
            results.to_csv(fp, index=False)
    
    return results

#metric failure experiments

#extreme failure (appendix)
def illustrative_failure():
    alpha = AlphaPrecision()
    kl = InverseKLDivergence()
    mmd = MaximumMeanDiscrepancy()
    res = pd.DataFrame(columns=['failure', 'alpha', 'beta', 'kl', 'mmd'])

    # Faiure to model $P_X$
    X, y, w, p, t = simulate_treatment_setup(1000, 1)
    X_df = pd.DataFrame(X)
    y_df = pd.DataFrame(y, columns = ['y'])
    w_df = pd.DataFrame(w, columns = ['w'])

    d_real = pd.concat([X_df,w_df,y_df], axis=1)
    c = d_real.drop(['w', 'y'], axis=1).columns

    # %%
    X, y, w, p, t = simulate_treatment_setup(1000, 1,seed=1)
    X_df = pd.DataFrame(X)
    y_df = pd.DataFrame(y, columns = ['y'])
    w_df = pd.DataFrame(w, columns = ['w'])

    d_syn = pd.concat([X_df,w_df,y_df], axis=1)
    d_syn[c] = 0

    res.loc[len(res)] = ['p_x', alpha.evaluate(GenericDataLoader(d_real), GenericDataLoader(d_syn))['delta_precision_alpha_OC'],
                         alpha.evaluate(GenericDataLoader(d_real), GenericDataLoader(d_syn))['delta_coverage_beta_OC'], 
                         kl.evaluate(GenericDataLoader(d_real), GenericDataLoader(d_syn))['marginal'],
                         mmd.evaluate(GenericDataLoader(d_real), GenericDataLoader(d_syn))['joint']]
    
    # Faiure to model $P_W|X$
    d = 1
    X, y, w, p, t = simulate_treatment_setup(1000, d)
    X_df = pd.DataFrame(X)
    y_df = pd.DataFrame(y, columns = ['y'])
    w_df = pd.DataFrame(w, columns = ['w'])

    d_real = pd.concat([X_df,w_df,y_df], axis=1)
    #d_real[c] *= 10

    # %%
    X, y, w, p, t = simulate_treatment_setup(1000, d, seed=1)
    X_df = pd.DataFrame(X)
    y_df = pd.DataFrame(y, columns = ['y'])
    w_df = pd.DataFrame(w, columns = ['w'])

    d_syn = pd.concat([X_df,w_df,y_df], axis=1)
    d_syn['w'] = 0

    res.loc[len(res)] = ['p_w|x', alpha.evaluate(GenericDataLoader(d_real), GenericDataLoader(d_syn))['delta_precision_alpha_OC'],
                         alpha.evaluate(GenericDataLoader(d_real), GenericDataLoader(d_syn))['delta_coverage_beta_OC'], 
                         kl.evaluate(GenericDataLoader(d_real), GenericDataLoader(d_syn))['marginal'],
                         mmd.evaluate(GenericDataLoader(d_real), GenericDataLoader(d_syn))['joint']]
    
    # Faiure to model $P_Y|W, X$

    d=1
    X, y, w, p, t = simulate_treatment_setup(1000, d, n_t=d)
    X_df = pd.DataFrame(X)
    y_df = pd.DataFrame(y, columns = ['y'])
    w_df = pd.DataFrame(w, columns = ['w'])

    d_real = pd.concat([X_df,w_df,y_df], axis=1)
    #d_real[c] *= 10

    # %%
    X, y, w, p, t = simulate_treatment_setup(1000, d, n_t=d, seed=1)
    X_df = pd.DataFrame(X)
    y_df = pd.DataFrame(y, columns = ['y'])
    w_df = pd.DataFrame(w, columns = ['w'])

    d_syn = pd.concat([X_df,w_df,y_df], axis=1)
    d_syn['y'] = np.random.normal(loc = 0, size=(1000,1))

    res.loc[len(res)] = ['p_y|w,x', alpha.evaluate(GenericDataLoader(d_real), GenericDataLoader(d_syn))['delta_precision_alpha_OC'],
                         alpha.evaluate(GenericDataLoader(d_real), GenericDataLoader(d_syn))['delta_coverage_beta_OC'], 
                         kl.evaluate(GenericDataLoader(d_real), GenericDataLoader(d_syn))['marginal'],
                         mmd.evaluate(GenericDataLoader(d_real), GenericDataLoader(d_syn))['joint']]
    
    res.to_csv('results/current_metric_failure.csv', index=False)

#treatment assignment failure (appendix)

def prop_test(
    X: np.ndarray,
    n_c: int = 0,
    n_w: int = 0,
    xi: float = 0.5,
    nonlinear: bool = True,
    offset: float = 0,
    target_prop: Optional[np.ndarray] = None,
) -> np.ndarray:
    if n_c + n_w == 0:
        # constant propensity
        return xi * np.ones(X.shape[0])
    else:
        coefs = np.ones(n_c + n_w)

        if nonlinear:
            z = np.dot(X[:, : (n_c + n_w)] , coefs) / (n_c + n_w)
        else:
            z = np.dot(X[:, : (n_c + n_w)], coefs) / (n_c + n_w)

        if type(offset) is float or type(offset) is int:
            prop = expit(xi * z + offset)
            if target_prop is not None:
                avg_prop = np.average(prop)
                prop = target_prop / avg_prop * prop
            return prop
        elif offset == "center":
            # center the propensity scores to median 0.5
            prop = expit(xi * (z - np.median(z)))
            if target_prop is not None:
                avg_prop = np.average(prop)
                prop = target_prop / avg_prop * prop
            return prop
        else:
            raise ValueError("Not a valid value for offset")

def selection_test_treatment_assignment(n_iter, save=False, fp=''):
    seed = random.randint(1, 1000000)
    np.random.seed(seed)
    alpha = AlphaPrecision()
    kl = InverseKLDivergence()
    ks = KolmogorovSmirnovTest()
    jsd = JensenShannonDistance()
    was = WassersteinDistance()
    results = pd.DataFrame(columns=['num correct confounders', 'alpha', 'beta', 'kl', 'was', 'ks', 'jsd', 'JSD_pi'])
    for _ in range(n_iter):
        i = np.random.randint(100)
        n=1000;d=5;n_w=5;n_t=0
        X,y,w,p,cate = simulate_treatment_setup(n, d=d, n_t=n_t, n_w=n_w, seed=0+i, propensity_model=prop_test)
        X_df = pd.DataFrame(X)
        X_df = pd.DataFrame(X)
        y_df = pd.DataFrame(y, columns = ['y'])
        w_df = pd.DataFrame(w, columns = ['w'])
        d_real = pd.concat([X_df,w_df,y_df], axis=1)

        X_syn,y_syn,w_syn,_,_ = simulate_treatment_setup(n, d=d, n_t=n_t, n_w=n_w, seed=1+i, propensity_model=prop_test)
        X_syn_df = pd.DataFrame(X_syn)
        w_syn_df = pd.DataFrame(w_syn, columns = ['w'])
        y_syn_df = pd.DataFrame(y_syn, columns = ['y'])
        d_syn_1 = pd.concat([X_syn_df,w_syn_df,y_syn_df], axis=1)

        X_syn,y_syn,w_syn,_,_ = simulate_treatment_setup(n, d=d, n_t=n_t, n_w=n_w-2, seed=3+i, propensity_model=prop_test)
        X_syn_df = pd.DataFrame(X_syn)
        w_syn_df = pd.DataFrame(w_syn, columns = ['w'])
        y_syn_df = pd.DataFrame(y_syn, columns = ['y'])
        d_syn_3 = pd.concat([X_syn_df,w_syn_df,y_syn_df], axis=1)

        X_syn,y_syn,w_syn,_,_ = simulate_treatment_setup(n, d=d, n_t=n_t, n_w=n_w-4, seed=5+i, propensity_model=prop_test)
        X_syn_df = pd.DataFrame(X_syn)
        w_syn_df = pd.DataFrame(w_syn, columns = ['w'])
        y_syn_df = pd.DataFrame(y_syn, columns = ['y'])
        d_syn_5 = pd.concat([X_syn_df,w_syn_df,y_syn_df], axis=1)

        
        results.loc[len(results)] = [5, alpha.evaluate(GenericDataLoader(d_real), GenericDataLoader(d_syn_1))['delta_precision_alpha_OC'], alpha.evaluate(GenericDataLoader(d_real), GenericDataLoader(d_syn_1))['delta_coverage_beta_OC'],
                                     kl.evaluate(GenericDataLoader(d_real), GenericDataLoader(d_syn_1))['marginal'], was.evaluate(GenericDataLoader(d_real), GenericDataLoader(d_syn_1))['joint'], 
                                     ks.evaluate(GenericDataLoader(d_real), GenericDataLoader(d_syn_1))['marginal'], jsd.evaluate(GenericDataLoader(d_real), GenericDataLoader(d_syn_1))['marginal'], evaluate_jsd(d_real, d_syn_1, 'w', 'y')]

       
        results.loc[len(results)] = [3, alpha.evaluate(GenericDataLoader(d_real), GenericDataLoader(d_syn_3))['delta_precision_alpha_OC'], alpha.evaluate(GenericDataLoader(d_real), GenericDataLoader(d_syn_3))['delta_coverage_beta_OC'],
                                     kl.evaluate(GenericDataLoader(d_real), GenericDataLoader(d_syn_3))['marginal'], was.evaluate(GenericDataLoader(d_real), GenericDataLoader(d_syn_3))['joint'], 
                                     ks.evaluate(GenericDataLoader(d_real), GenericDataLoader(d_syn_3))['marginal'], jsd.evaluate(GenericDataLoader(d_real), GenericDataLoader(d_syn_3))['marginal'], evaluate_jsd(d_real, d_syn_3, 'w', 'y')]

        results.loc[len(results)] = [1, alpha.evaluate(GenericDataLoader(d_real), GenericDataLoader(d_syn_5))['delta_precision_alpha_OC'], alpha.evaluate(GenericDataLoader(d_real), GenericDataLoader(d_syn_5))['delta_coverage_beta_OC'],
                                     kl.evaluate(GenericDataLoader(d_real), GenericDataLoader(d_syn_5))['marginal'], was.evaluate(GenericDataLoader(d_real), GenericDataLoader(d_syn_5))['joint'], 
                                     ks.evaluate(GenericDataLoader(d_real), GenericDataLoader(d_syn_5))['marginal'], jsd.evaluate(GenericDataLoader(d_real), GenericDataLoader(d_syn_5))['marginal'], evaluate_jsd(d_real, d_syn_5, 'w', 'y')]

        if save:
            results.to_csv(fp, index=False)
    return results

def selection_treatment_assignment_exp():
    results = selection_test_treatment_assignment(10)
    results.to_csv('results/model_selection_treatment_assignment.csv', index=False)

#outcome generation failure

def selection_test_outcome_generation(n_iter, save=False, fp=''):
    alpha = AlphaPrecision()
    kl = InverseKLDivergence()
    ks = KolmogorovSmirnovTest()
    jsd = JensenShannonDistance()
    was = WassersteinDistance()
    results = pd.DataFrame(columns=['outcome learner', 'alpha', 'beta', 'kl', 'was', 'ks', 'jsd', 'u_pehe', 'oracle'])
    for _ in range(n_iter):
        n=1000;d=10;n_c=0;n_t=1
        X,y,w,p,cate = simulate_treatment_setup(n, d=d, n_c=n_c, n_t=n_t)
        X_df = pd.DataFrame(X)
        y_df = pd.DataFrame(y, columns = ['y'])
        w_df = pd.DataFrame(w, columns = ['w'])
        d_real = pd.concat([X_df,w_df,y_df], axis=1)

        X_syn,y_syn,w_syn,_,_ = simulate_treatment_setup(n, d=d, n_c=n_c, n_t=n_t)
        X_syn_df = pd.DataFrame(X_syn)
        w_syn_df = pd.DataFrame(w_syn, columns = ['w'])
        synth_cov_with_prop = pd.concat([X_syn_df,w_syn_df], axis=1)

        t = synth_cov_with_prop.copy()
        l = TLearner(n_unit_in=d, binary_y=False, batch_norm=False)
        l.fit(X, y, w)
        _, y0, y1 = l.predict(X_syn_df, return_po=True)
        outcomes = []
        for index, value in synth_cov_with_prop['w'].iteritems():
            if value == 0:
                outcomes.append(y0[index].item())
            else:
                outcomes.append(y1[index].item())
        t['y'] = outcomes
        X_t = np.array(t.drop(['w', 'y'], axis=1))
        y_t = np.array(t['y'])
        w_t = np.array(t['w'])
        l_t = TLearner(n_unit_in=d, binary_y=False)
        l_t.fit(X_t, y_t, w_t)
        
        pred_t = l_t.predict(X)
        #return synth_cov_with_prop, t, pred_t
        oracle_t = mean_squared_error(cate, pred_t.detach().cpu().numpy(), squared=False)
        print(oracle_t)
        s = synth_cov_with_prop.copy()
        l = SLearner(n_unit_in=d, binary_y=False, batch_norm=False)
        l.fit(X, y, w)
        _, y0, y1 = l.predict(X_syn_df, return_po=True)
        outcomes = []
        for index, value in synth_cov_with_prop['w'].iteritems():
            if value == 0:
                outcomes.append(y0[index].item())
            else:
                outcomes.append(y1[index].item())
        s['y'] = outcomes
        X_s = np.array(s.drop(['w', 'y'], axis=1))
        y_s = np.array(s['y'])
        w_s = np.array(s['w'])
        l_s = TLearner(n_unit_in=d, binary_y=False)
        l_s.fit(X_s, y_s, w_s)
        pred_s= l_s.predict(X)
        oracle_s = mean_squared_error(cate, pred_s.detach().cpu().numpy(), squared=False)
        print(oracle_s)
        dr = synth_cov_with_prop.copy()
        l = DragonNet(n_unit_in=d, binary_y=False, batch_norm=False)
        l.fit(X, y, w)
        _, y0, y1 = l.predict(X_syn_df, return_po=True)
        outcomes = []
        for index, value in synth_cov_with_prop['w'].iteritems():
            if value == 0:
                outcomes.append(y0[index].item())
            else:
                outcomes.append(y1[index].item())
        dr['y'] = outcomes
        X_dr = np.array(dr.drop(['w', 'y'], axis=1))
        y_dr = np.array(dr['y'])
        w_dr = np.array(dr['w'])
        l_dr = TLearner(n_unit_in=d, binary_y=False)
        l_dr.fit(X_dr, y_dr, w_dr)
        pred_dr= l_dr.predict(X)
        oracle_dr = mean_squared_error(cate, pred_dr.detach().cpu().numpy(), squared=False)
        print(oracle_dr)
        tar = synth_cov_with_prop.copy()
        l = TARNet(n_unit_in=d, binary_y=False, batch_norm=False)
        l.fit(X, y, w)
        _, y0, y1 = l.predict(X_syn_df, return_po=True)
        outcomes = []
        for index, value in synth_cov_with_prop['w'].iteritems():
            if value == 0:
                outcomes.append(y0[index].item())
            else:
                outcomes.append(y1[index].item())
        tar['y'] = outcomes
        X_tar = np.array(tar.drop(['w', 'y'], axis=1))
        y_tar = np.array(tar['y'])
        w_tar = np.array(tar['w'])
        l_tar = TLearner(n_unit_in=d, binary_y=False)
        l_tar.fit(X_tar, y_tar, w_tar)
        pred_tar= l_tar.predict(X)
        oracle_tar = mean_squared_error(cate, pred_tar.detach().cpu().numpy(), squared=False)
        print(oracle_tar)
        
        results.loc[len(results)] = ['t', alpha.evaluate(GenericDataLoader(d_real), GenericDataLoader(t))['delta_precision_alpha_OC'], alpha.evaluate(GenericDataLoader(d_real), GenericDataLoader(t))['delta_coverage_beta_OC'],
                                     kl.evaluate(GenericDataLoader(d_real), GenericDataLoader(t))['marginal'], was.evaluate(GenericDataLoader(d_real), GenericDataLoader(t))['joint'], 
                                     ks.evaluate(GenericDataLoader(d_real), GenericDataLoader(t))['marginal'], jsd.evaluate(GenericDataLoader(d_real), GenericDataLoader(t))['marginal'], evaluate_average_u_pehe(d_real, t, 'w', 'y', d), oracle_t]
        results.loc[len(results)] = ['s', alpha.evaluate(GenericDataLoader(d_real), GenericDataLoader(s))['delta_precision_alpha_OC'], alpha.evaluate(GenericDataLoader(d_real), GenericDataLoader(s))['delta_coverage_beta_OC'],
                                     kl.evaluate(GenericDataLoader(d_real), GenericDataLoader(s))['marginal'], was.evaluate(GenericDataLoader(d_real), GenericDataLoader(s))['joint'], 
                                     ks.evaluate(GenericDataLoader(d_real), GenericDataLoader(s))['marginal'], jsd.evaluate(GenericDataLoader(d_real), GenericDataLoader(s))['marginal'], evaluate_average_u_pehe(d_real, s, 'w', 'y', d), oracle_s]
        results.loc[len(results)] = ['dr', alpha.evaluate(GenericDataLoader(d_real), GenericDataLoader(dr))['delta_precision_alpha_OC'], alpha.evaluate(GenericDataLoader(d_real), GenericDataLoader(dr))['delta_coverage_beta_OC'],
                                     kl.evaluate(GenericDataLoader(d_real), GenericDataLoader(dr))['marginal'], was.evaluate(GenericDataLoader(d_real), GenericDataLoader(dr))['joint'], 
                                     ks.evaluate(GenericDataLoader(d_real), GenericDataLoader(dr))['marginal'], jsd.evaluate(GenericDataLoader(d_real), GenericDataLoader(dr))['marginal'], evaluate_average_u_pehe(d_real, dr, 'w', 'y', d), oracle_dr]
        results.loc[len(results)] = ['tar', alpha.evaluate(GenericDataLoader(d_real), GenericDataLoader(tar))['delta_precision_alpha_OC'], alpha.evaluate(GenericDataLoader(d_real), GenericDataLoader(tar))['delta_coverage_beta_OC'],
                                     kl.evaluate(GenericDataLoader(d_real), GenericDataLoader(tar))['marginal'], was.evaluate(GenericDataLoader(d_real), GenericDataLoader(tar))['joint'], 
                                     ks.evaluate(GenericDataLoader(d_real), GenericDataLoader(tar))['marginal'], jsd.evaluate(GenericDataLoader(d_real), GenericDataLoader(tar))['marginal'], evaluate_average_u_pehe(d_real, tar, 'w', 'y', d), oracle_tar]
        if save:
            results.to_csv(fp, index=False)
    return results

def selection_outcome_gen_exp():
    results = selection_test_outcome_generation(10)
    results.to_csv('results/model_selection_outcome_generation.csv', index=False)


#steam medical data performance
def steam_v_standard(real, gen, treatment_col, outcome_col, n_iter, d_name):
    results = seq_test(real, gen, treatment_col, outcome_col, n_iter)
    results.to_csv(f'results/{d_name}_{gen}_steam_v_standard.csv', index=False)

def full_stream_vs_standard_exp(aids, ihdp, acic_encoded, jobs):
    steam_v_standard(ihdp, 'ddpm', 'treatment', 'y_factual', 20, 'ihdp')
    steam_v_standard(ihdp, 'ctgan', 'treatment', 'y_factual', 20, 'ihdp')
    steam_v_standard(ihdp, 'tvae', 'treatment', 'y_factual', 20, 'ihdp')
    steam_v_standard(ihdp, 'arf', 'treatment', 'y_factual', 20, 'ihdp')
    steam_v_standard(ihdp, 'nflow', 'treatment', 'y_factual', 20, 'ihdp')

    steam_v_standard(jobs, 'ddpm', 'training', 're78', 20, 'jobs')
    steam_v_standard(jobs, 'ctgan', 'training', 're78', 20, 'jobs')
    steam_v_standard(jobs, 'tvae', 'training', 're78', 20, 'jobs')
    steam_v_standard(jobs, 'arf', 'training', 're78', 20, 'jobs')
    steam_v_standard(jobs, 'nflow', 'training', 're78', 20, 'jobs')

    steam_v_standard(acic_encoded, 'ddpm', 'z', 'y', 20, 'acic')
    steam_v_standard(acic_encoded, 'ctgan', 'z', 'y', 20, 'acic')
    steam_v_standard(acic_encoded, 'tvae', 'z', 'y', 20, 'acic')
    steam_v_standard(acic_encoded, 'arf', 'z', 'y', 20, 'acic')
    steam_v_standard(acic_encoded, 'nflow', 'z', 'y', 20, 'acic')

    steam_v_standard(aids, 'ddpm', 't', 'y', 20, 'aids')
    steam_v_standard(aids, 'ctgan', 't', 'y', 20, 'aids')
    steam_v_standard(aids, 'tvae', 't', 'y', 20, 'aids')
    steam_v_standard(aids, 'arf', 't', 'y', 20, 'aids')
    steam_v_standard(aids, 'nflow', 't', 'y', 20, 'aids')


#insight exps

#covariate insight

def num_cov_insight_exp(n, ds, n_t, n_c, gen, n_iter, save=False, fp=''):
    results = pd.DataFrame(columns=['method', 'p_alpha_x', 'r_beta_x', 'jsd_pi', 'u_pehe', 'num_cov'])
    for d in ds:
        X,y,w,p,t = simulate_treatment_setup(n, d=d, n_c=n_c, n_t=n_t)
        X_df = pd.DataFrame(X)
        y_df = pd.DataFrame(y, columns = ['y'])
        w_df = pd.DataFrame(w, columns = ['w'])
        d_real = pd.concat([X_df,w_df,y_df], axis=1)

        r = seq_test(d_real, gen, 'w', 'y', n_iter)
        r['num_cov'] = d
        results = pd.concat([results, r])
        print(f'Tested num. covariates = {d}')

        if save:
            results.to_csv(fp, index=False)
        
    return results

def run_cov_insight_exp():
    results = num_cov_insight_exp(1000, [5,10,20,50], 2, 2, 'ddpm', 10)
    results.to_csv('results/num_cov_insight.csv', index=False)

    x = [5,10,20,50]
    d_seq = results[results['method'] == 'STEAM'].groupby(['num_cov']).mean()['u_pehe']
    d_stand = results[results['method'] == 'standard'].groupby(['num_cov']).mean()['u_pehe']

    d_seq_std = results[results['method']=='STEAM'].groupby(['num_cov']).std()['u_pehe'] / np.sqrt(10) * 1.96
    d_stand_std = results[results['method']=='standard'].groupby(['num_cov']).std()['u_pehe']/ np.sqrt(10) * 1.96

    plt.rcParams.update({'font.size': 30})
    plt.figure(figsize=(8, 6))
    plt.plot(x, d_stand, 'o-', label = 'Standard')
    plt.plot(x, d_seq, 'o-', label = 'STEAM')


    plt.fill_between(x, d_stand - d_stand_std, d_stand + d_stand_std, alpha=0.2)
    plt.fill_between(x, d_seq - d_seq_std, d_seq + d_seq_std, alpha=0.2)
    plt.ylabel('$U_{PEHE}$')
    plt.xticks(x)
    plt.savefig('plots/covariate_dimension_insight_U.pdf', bbox_inches='tight')

    d_seq = results[results['method'] == 'STEAM'].groupby(['num_cov']).mean()['jsd_pi']
    d_stand = results[results['method'] == 'standard'].groupby(['num_cov']).mean()['jsd_pi']

    d_seq_std = results[results['method']=='STEAM'].groupby(['num_cov']).std()['jsd_pi'] / np.sqrt(10) * 1.96
    d_stand_std = results[results['method']=='standard'].groupby(['num_cov']).std()['jsd_pi']/ np.sqrt(10) * 1.96

    plt.rcParams.update({'font.size': 30})
    plt.figure(figsize=(8, 6))
    plt.plot(x, d_stand, 'o-', label = 'Standard')
    plt.plot(x, d_seq, 'o-', label = 'STEAM')

    plt.fill_between(x, d_stand - d_stand_std, d_stand + d_stand_std, alpha=0.2)
    plt.fill_between(x, d_seq - d_seq_std, d_seq + d_seq_std, alpha=0.2)
    plt.legend()
    plt.xlabel('$d$')
    plt.ylabel('$JSD_\pi$')
    plt.xticks(x)
    plt.savefig('plots/covariate_dimension_insight_jsd.pdf', bbox_inches='tight')


#treatment assignment insight

def confounding_insight(n, d, n_t, n_cs, gen, n_iter, save=False, fp=''):
    results = pd.DataFrame(columns=['method', 'p_alpha_x', 'r_beta_x', 'jsd_pi', 'u_pehe', 'n_c'])
    for n_c in n_cs:
        X,y,w,p,t = simulate_treatment_setup(n, d=d, n_c=n_c, n_t=n_t)
        X_df = pd.DataFrame(X)
        y_df = pd.DataFrame(y, columns = ['y'])
        w_df = pd.DataFrame(w, columns = ['w'])
        d_real = pd.concat([X_df,w_df,y_df], axis=1)

        r = seq_test(d_real, gen, 'w', 'y', n_iter)
        r['n_c'] = n_c
        results = pd.concat([results, r])
        print(f'Tested n_c = {n_c}')

    if save:
        results.to_csv(fp, index=False)
        
    return results

def run_treatment_assignment_insight_exp():
    results = confounding_insight(1000, 10, 2, [1,2,3,4,5], 'ddpm', 10)
    results.to_csv('results/treatment_assignment_insight.csv', index=False)

    x = [1,2,3,4,5]
    d_seq = results[results['method']=='STEAM'].groupby(['n_c']).mean()['jsd_pi']
    d_stand = results[results['method']=='standard'].groupby(['n_c']).mean()['jsd_pi']

    d_seq_std = results[results['method']=='STEAM'].groupby(['n_c']).std()['jsd_pi'] / np.sqrt(10) * 1.96
    d_stand_std = results[results['method']=='standard'].groupby(['n_c']).std()['jsd_pi']/ np.sqrt(10) * 1.96

    plt.rcParams.update({'font.size': 30})
    plt.figure(figsize=(8, 6))
    plt.plot(x, d_stand, 'o-', label = 'Standard')
    plt.plot(x, d_seq, 'o-', label = 'STEAM')

    plt.fill_between(x, d_stand - d_stand_std, d_stand + d_stand_std, alpha=0.2)
    plt.fill_between(x, d_seq - d_seq_std, d_seq + d_seq_std, alpha=0.2)
    plt.legend()
    plt.xlabel('K')
    plt.ylabel('$JSD_\pi$')
    plt.ylim([0.8,1])
    plt.xticks([1,2,3,4,5])
    plt.savefig('plots/treatment_assignment_insight.pdf', bbox_inches='tight')

#outcome generation insight

def predictive_insight(n, d, n_c, n_ts, gen, n_iter, save=False, fp=''):
    results = pd.DataFrame(columns=['method', 'p_alpha_x', 'r_beta_x', 'jsd_pi', 'u_pehe', 'n_t'])
    for n_t in n_ts:
        X,y,w,p,t = simulate_treatment_setup(n, d=d, n_c=n_c, n_t=n_t)
        X_df = pd.DataFrame(X)
        y_df = pd.DataFrame(y, columns = ['y'])
        w_df = pd.DataFrame(w, columns = ['w'])
        d_real = pd.concat([X_df,w_df,y_df], axis=1)

        r = seq_test(d_real, gen, 'w', 'y', n_iter)
        r['n_t'] = n_t
        results = pd.concat([results, r])

        print(f'Tested n_t = {n_t}')

        if save:
            results.to_csv(fp, index=False)
        
    return results

def run_outcome_gen_insight_exp():
    results = predictive_insight(1000, 10, 2, [1,2,3,4,5], 'ddpm', 10)
    results.to_csv('results/outcome_generation_insight.csv', index=False)

    x = [1,2,3,4,5]
    u_seq = results[results['method']=='STEAM'].groupby(['n_t']).mean()['u_pehe']
    u_stand = results[results['method']=='standard'].groupby(['n_t']).mean()['u_pehe']

    u_seq_std = results[results['method']=='STEAM'].groupby(['n_t']).std()['u_pehe'] / np.sqrt(10) * 1.96
    u_stand_std = results[results['method']=='standard'].groupby(['n_t']).std()['u_pehe']/ np.sqrt(10) * 1.96

    plt.rcParams.update({'font.size': 30})
    plt.figure(figsize=(8, 6))
    plt.plot(x, u_stand, 'o-', label = 'Standard')
    plt.plot(x, u_seq, 'o-', label = 'STEAM')

    plt.fill_between(x, u_stand - u_stand_std, u_stand + u_stand_std, alpha=0.2)
    plt.fill_between(x, u_seq - u_seq_std, u_seq + u_seq_std, alpha=0.2)
    plt.legend()
    plt.xlabel('$K$')
    plt.ylabel('$U_{PEHE}$')
    plt.xticks([1, 2, 3, 4, 5], [3, 4, 5, 6, 7])
    plt.savefig('plots/predictive_insight_plot.pdf', bbox_inches='tight')


#privacy exp
def privacy_comparison(real, gen, treatment_col, outcome_col, delta, epsilons, n_iter, save=False, fp=''):
    results = pd.DataFrame(columns=['method', 'epsilon', 'p_alpha_x', 'r_beta_x', 'jsd_pi', 'u_pehe',])
    n_units = len(real.drop([treatment_col, outcome_col], axis=1).columns)
    for epsilon in epsilons:
        for _ in range(n_iter):
            synth = generate_standard(real, gen, private=True, epsilon=epsilon, delta=delta)

            results.loc[len(results)] = ['standard', epsilon, evaluate_f(real, synth, treatment_col, outcome_col), evaluate_c(real, synth, treatment_col, outcome_col), 
                                     evaluate_jsd(real, synth, treatment_col, outcome_col), evaluate_average_u_pehe(real, synth, treatment_col, outcome_col, n_units)]

            synth_seq = generate_sequentially(real, gen, treatment_col, outcome_col, private=True, epsilon = epsilon/3, delta=delta/3)

            results.loc[len(results)] = ['STEAM', epsilon, evaluate_f(real, synth_seq, treatment_col, outcome_col), evaluate_c(real, synth_seq, treatment_col, outcome_col), 
                                     evaluate_jsd(real, synth_seq, treatment_col, outcome_col), evaluate_average_u_pehe(real, synth_seq, treatment_col, outcome_col, n_units)]
    
            if save:
                print('saving results')
                results.to_csv(fp, index=False)
    return results

def privacy_exp():
    X,y,w,p,t = simulate_treatment_setup(1000, d=5, n_t=2, n_c=2, seed=2)
    X_df = pd.DataFrame(X)
    y_df = pd.DataFrame(y, columns = ['y'])
    w_df = pd.DataFrame(w, columns = ['w'])
    d_real = pd.concat([X_df,w_df,y_df], axis=1)

    results = privacy_comparison(d_real, 'dpgan', 'w', 'y', delta = 1e-3, epsilons = [1,2,3,5,10,15], n_iter = 10)
    results.to_csv('results/privacy_results.csv', index=False)

    results_standard = results[results['method']=='standard']
    results_seq = results[results['method']=='STEAM']

    x = [1,2,3,5,10,15]
    d_stand = results_standard.groupby('epsilon').mean()['jsd_pi']
    d_seq = results_seq.groupby('epsilon').mean()['jsd_pi']
    f_stand = results_standard.groupby('epsilon').mean()['p_alpha_x']
    f_seq = results_seq.groupby('epsilon').mean()['p_alpha_x']
    c_stand = results_standard.groupby('epsilon').mean()['r_beta_x']
    c_seq = results_seq.groupby('epsilon').mean()['r_beta_x']
    u_stand = results_standard.groupby('epsilon').mean()['u_pehe']
    u_seq = results_seq.groupby('epsilon').mean()['u_pehe']


    d_stand_std = results_standard.groupby('epsilon').std()['jsd_pi'] / np.sqrt(10) * 1.96
    d_seq_std = results_seq.groupby('epsilon').std()['jsd_pi']/ np.sqrt(10)* 1.96
    f_stand_std = results_standard.groupby('epsilon').std()['p_alpha_x']/ np.sqrt(10)* 1.96
    f_seq_std = results_seq.groupby('epsilon').std()['p_alpha_x']/ np.sqrt(10)* 1.96
    c_stand_std = results_standard.groupby('epsilon').std()['r_beta_x']/ np.sqrt(10)* 1.96
    c_seq_std = results_seq.groupby('epsilon').std()['r_beta_x']/ np.sqrt(10)* 1.96
    u_stand_std = results_standard.groupby('epsilon').std()['u_pehe']/ np.sqrt(10)* 1.96
    u_seq_std = results_seq.groupby('epsilon').std()['u_pehe']/ np.sqrt(10)* 1.96

    plt.rcParams.update({'font.size': 30})
    plt.figure(figsize=(8, 6))
    plt.plot(x, d_stand, 'o-', label = 'Standard')
    plt.plot(x, d_seq, 'o-', label = 'STEAM')
    plt.fill_between(x, d_stand - d_stand_std, d_stand + d_stand_std, alpha=0.2)
    plt.fill_between(x, d_seq - d_seq_std, d_seq + d_seq_std, alpha=0.2)
    plt.ylim([0,1])
    plt.ylabel(r'$JSD_\pi$')
    plt.xlabel('$\epsilon$')
    plt.xscale('log')
    plt.savefig('plots/D_with_epsilon.pdf', bbox_inches='tight')

    plt.figure(figsize=(8, 6))
    plt.plot(x, f_stand, 'o-', label = 'Standard')
    plt.plot(x, f_seq, 'o-', label = 'STEAM')
    plt.fill_between(x, f_stand - f_stand_std, f_stand + f_stand_std, alpha=0.2)
    plt.fill_between(x, f_seq - f_seq_std, f_seq + f_seq_std, alpha=0.2)
    #plt.legend()
    plt.ylim([0,1])
    plt.ylabel(r'$P_{\alpha, X}$')
    plt.xlabel('$\epsilon$')
    plt.xscale('log')
    plt.legend()
    plt.savefig('plots/F_with_epsilon.pdf', bbox_inches='tight')

    plt.figure(figsize=(8, 6))
    plt.plot(x, c_stand, 'o-', label = 'Standard')
    plt.plot(x, c_seq, 'o-', label = 'STEAM')
    plt.fill_between(x, c_stand - c_stand_std, c_stand + c_stand_std, alpha=0.2)
    plt.fill_between(x, c_seq - c_seq_std, c_seq + c_seq_std, alpha=0.2)
    #plt.legend()
    plt.ylim([0,1])
    plt.ylabel(r'$R_{\beta, X}$')
    plt.xlabel('$\epsilon$')
    plt.xscale('log')
    plt.savefig('plots/C_with_epsilon.pdf', bbox_inches='tight')

    plt.figure(figsize=(8, 6))
    plt.plot(x, u_stand, 'o-', label = 'Standard')
    plt.plot(x, u_seq, 'o-', label = 'STEAM')
    plt.fill_between(x, u_stand - u_stand_std, u_stand + u_stand_std, alpha=0.2)
    plt.fill_between(x, u_seq - u_seq_std, u_seq + u_seq_std, alpha=0.2)
    #plt.legend()
    plt.ylabel('$U_{PEHE}$')
    plt.xlabel('$\epsilon$')
    plt.xscale('log')
    plt.savefig('plots/U_with_epsilon.pdf', bbox_inches='tight')


#ablation exp

def generate_sequential_no_prop(real, gen, treatment_col, outcome_col):
    #generate X and W
    random.seed()
    g = Plugins().get(gen, random_state = random.randint(0, 1000000))
    real_X_W = real.drop([outcome_col], axis=1)
    print(f'Fitting {gen} X and W model')
    g.fit(real_X_W)
    print(f'Generating {gen} synthetic X and W')
    synth = g.generate(count = len(real)).dataframe()
    synth[outcome_col] = 0

    #generate Y
    X = np.array(real.drop([treatment_col, outcome_col], axis=1))
    y = np.array(real[outcome_col])
    w = np.array(real[treatment_col])
    n_units = len(real.drop([treatment_col, outcome_col], axis=1).columns)
    
    l = TLearner(n_unit_in=n_units, binary_y=False, seed=random.randint(0,1000000))
    print('Fitting CATE learner')
    l.fit(X, y, w)

    seq_X = np.array(synth.drop([treatment_col, outcome_col], axis=1))
    print('Generating POs')
    cate, y0, y1 = l.predict(seq_X, return_po=True)

    outcomes = []
    for index, value in synth[treatment_col].items():
        if value == 0:
            outcomes.append(y0[index].item())
        else:
            outcomes.append(y1[index].item())

    synth[outcome_col] = outcomes
    return synth

def ablation(real, gen, treatment_col, outcome_col, n_iter, binary_y=False, save=False, fp=''):
    results = pd.DataFrame(columns = ['method', 'p_alpha_x', 'r_beta_x', 'jsd_pi', 'u_pehe'])
    for _ in range(n_iter):
        ablation = generate_sequential_no_prop(real, gen, treatment_col, outcome_col)
        n_units = len(real.drop([treatment_col, outcome_col], axis=1).columns)

        results.loc[len(results)] = ['ablation', evaluate_f(real, ablation, treatment_col, outcome_col), evaluate_c(real, ablation, treatment_col, outcome_col), 
                                     evaluate_jsd(real, ablation, treatment_col, outcome_col), evaluate_average_u_pehe(real, ablation, treatment_col, outcome_col, n_units,binary_y)]

        if save:
            results.to_csv(fp, index=False)
    
    return results

def ablation_exp(real, gen, treatment_col, outcome_col, n_iter, d_name):
    results = ablation(real, gen, treatment_col, outcome_col, n_iter)
    results.to_csv(f'results/{d_name}_{gen}_ablation.csv', index=False)

def full_ablation_exp(aids, ihdp, acic_encoded):
    ablation_exp(aids, 'tvae', 't', 'y', 10, 'aids')
    ablation_exp(ihdp, 'ctgan', 'treatment', 'y_factual', 10, 'ihdp')
    ablation_exp(acic_encoded, 'tvae', 'z', 'y', 10, 'acic')

#hyperparam stability exp

#n_units
def generate_stand_with_hyperparams_n_units(real, gen, hyperparam = 500):
    g = Plugins().get(gen, random_state = random.randint(0, 1000000), generator_n_units_hidden = hyperparam)
    print(f'Fitting {gen} model')
    g.fit(real)
    print(f'Generating {gen} synthetic dataset')
    synth = g.generate(count = len(real)).dataframe()
    return synth

def generate_steam_with_hyperparams_n_units(real, gen, treatment_col, outcome_col, hyperparam = 500):
    g = Plugins().get(gen, random_state = random.randint(0, 1000000), generator_n_units_hidden = hyperparam)
    real_cov = real.drop([treatment_col, outcome_col], axis=1)
    print(f'Fitting {gen} covariate model')
    g.fit(real_cov)
    print(f'Generating {gen} synthetic covariates')
    synth_cov = g.generate(count = len(real)).dataframe()

    #generate propensities
    X = np.array(real.drop([treatment_col, outcome_col], axis=1))
    y = np.array(real[treatment_col])
    
    classifier = LogisticRegression(random_state = random.randint(0, 1000000))
    print('Fitting propensity model')
    classifier.fit(X, y)
    print('Generating propensities')
    probabilities = classifier.predict_proba(np.array(synth_cov))
    prob_class_1 = probabilities[:, 1]
    binary_outcomes = np.random.binomial(n=1, p=prob_class_1)

    synth_cov_with_prop = synth_cov
    synth_cov_with_prop[treatment_col] = pd.Series(binary_outcomes)

    synth_cov_with_prop[outcome_col] = 0


    synth = synth_cov_with_prop
    X = np.array(real.drop([treatment_col, outcome_col], axis=1))
    y = np.array(real[outcome_col])
    w = np.array(real[treatment_col])
    n_units = len(real.drop([treatment_col, outcome_col], axis=1).columns)
    
    l = TLearner(n_unit_in=n_units, binary_y=False, seed=random.randint(0,1000000))
    print('Fitting CATE learner')
    l.fit(X, y, w)

    seq_X = np.array(synth.drop([treatment_col, outcome_col], axis=1))
    print('Generating POs')
    cate, y0, y1 = l.predict(seq_X, return_po=True)

    outcomes = []
    for index, value in synth[treatment_col].items():
        if value == 0:
            outcomes.append(y0[index].item())
        else:
            outcomes.append(y1[index].item())

    synth[outcome_col] = outcomes
    return synth

def hyperparam_test_n_units(real, gen, treatment_col, outcome_col, hyperparams, n_iter, binary_y=False, save=False, fp=''):
    results = pd.DataFrame(columns = ['method', 'hyperparam', 'p_alpha_x', 'r_beta_x', 'jsd_pi', 'u_pehe'])
    for h in hyperparams:
        for _ in range(n_iter):
            stand = generate_stand_with_hyperparams_n_units(real, gen, h)
            steam = generate_steam_with_hyperparams_n_units(real, gen, treatment_col, outcome_col, h)
            n_units = len(real.drop([treatment_col, outcome_col], axis=1).columns)

            results.loc[len(results)] = ['standard', h, evaluate_f(real, stand, treatment_col, outcome_col), evaluate_c(real, stand, treatment_col, outcome_col), 
                                        evaluate_jsd(real, stand, treatment_col, outcome_col), evaluate_average_u_pehe(real, stand, treatment_col, outcome_col, n_units,binary_y)]
            results.loc[len(results)] = ['steam', h, evaluate_f(real, steam, treatment_col, outcome_col), evaluate_c(real, steam, treatment_col, outcome_col), 
                                        evaluate_jsd(real, steam, treatment_col, outcome_col), evaluate_average_u_pehe(real, steam, treatment_col, outcome_col, n_units,binary_y)]
            if save:
                results.to_csv(fp, index=False)
    
    return results

def hyperparam_exp_n_units(real):
    results = hyperparam_test_n_units(real, 'ctgan', 'treatment', 'y_factual', [5,50,100,300,500], 5)
    results.to_csv('results/hyperparam_n_units.csv', index=False)


#n_layers
def generate_stand_with_hyperparams_n_layers(real, gen, hyperparam = 500):
    g = Plugins().get(gen, random_state = random.randint(0, 1000000), generator_n_layers_hidden = hyperparam)
    print(f'Fitting {gen} model')
    g.fit(real)
    print(f'Generating {gen} synthetic dataset')
    synth = g.generate(count = len(real)).dataframe()
    return synth

def generate_steam_with_hyperparams_n_layers(real, gen, treatment_col, outcome_col, hyperparam = 500):
    g = Plugins().get(gen, random_state = random.randint(0, 1000000), generator_n_layers_hidden = hyperparam)
    real_cov = real.drop([treatment_col, outcome_col], axis=1)
    print(f'Fitting {gen} covariate model')
    g.fit(real_cov)
    print(f'Generating {gen} synthetic covariates')
    synth_cov = g.generate(count = len(real)).dataframe()

    #generate propensities
    X = np.array(real.drop([treatment_col, outcome_col], axis=1))
    y = np.array(real[treatment_col])
    
    classifier = LogisticRegression(random_state = random.randint(0, 1000000))
    print('Fitting propensity model')
    classifier.fit(X, y)
    print('Generating propensities')
    probabilities = classifier.predict_proba(np.array(synth_cov))
    prob_class_1 = probabilities[:, 1]
    binary_outcomes = np.random.binomial(n=1, p=prob_class_1)

    synth_cov_with_prop = synth_cov
    synth_cov_with_prop[treatment_col] = pd.Series(binary_outcomes)

    synth_cov_with_prop[outcome_col] = 0


    synth = synth_cov_with_prop
    X = np.array(real.drop([treatment_col, outcome_col], axis=1))
    y = np.array(real[outcome_col])
    w = np.array(real[treatment_col])
    n_units = len(real.drop([treatment_col, outcome_col], axis=1).columns)
    
    l = TLearner(n_unit_in=n_units, binary_y=False, seed=random.randint(0,1000000))
    print('Fitting CATE learner')
    l.fit(X, y, w)

    seq_X = np.array(synth.drop([treatment_col, outcome_col], axis=1))
    print('Generating POs')
    cate, y0, y1 = l.predict(seq_X, return_po=True)

    outcomes = []
    for index, value in synth[treatment_col].items():
        if value == 0:
            outcomes.append(y0[index].item())
        else:
            outcomes.append(y1[index].item())

    synth[outcome_col] = outcomes
    return synth

def hyperparam_test_n_layers(real, gen, treatment_col, outcome_col, hyperparams, n_iter, binary_y=False, save=False, fp=''):
    results = pd.DataFrame(columns = ['method', 'hyperparam', 'p_alpha_x', 'r_beta_x', 'jsd_pi', 'u_pehe'])
    for h in hyperparams:
        for _ in range(n_iter):
            stand = generate_stand_with_hyperparams_n_layers(real, gen, h)
            steam = generate_steam_with_hyperparams_n_layers(real, gen, treatment_col, outcome_col, h)
            n_units = len(real.drop([treatment_col, outcome_col], axis=1).columns)

            results.loc[len(results)] = ['standard', h, evaluate_f(real, stand, treatment_col, outcome_col), evaluate_c(real, stand, treatment_col, outcome_col), 
                                        evaluate_jsd(real, stand, treatment_col, outcome_col), evaluate_average_u_pehe(real, stand, treatment_col, outcome_col, n_units,binary_y)]
            results.loc[len(results)] = ['steam', h, evaluate_f(real, steam, treatment_col, outcome_col), evaluate_c(real, steam, treatment_col, outcome_col), 
                                        evaluate_jsd(real, steam, treatment_col, outcome_col), evaluate_average_u_pehe(real, steam, treatment_col, outcome_col, n_units,binary_y)]
            if save:
                results.to_csv(fp, index=False)
    
    return results

def hyperparam_exp_n_layers(real):
    results = hyperparam_test_n_layers(real, 'ctgan', 'treatment', 'y_factual', [2,3,4,5], 5)
    results.to_csv('results/hyperparam_n_layers.csv', index=False)


#activation function
def generate_stand_with_hyperparams_nonlin(real, gen, hyperparam = 500):
    g = Plugins().get(gen, random_state = random.randint(0, 1000000), generator_nonlin = hyperparam)
    print(f'Fitting {gen} model')
    g.fit(real)
    print(f'Generating {gen} synthetic dataset')
    synth = g.generate(count = len(real)).dataframe()
    return synth

def generate_steam_with_hyperparams_nonlin(real, gen, treatment_col, outcome_col, hyperparam = 500):
    g = Plugins().get(gen, random_state = random.randint(0, 1000000), generator_nonlin = hyperparam)
    real_cov = real.drop([treatment_col, outcome_col], axis=1)
    print(f'Fitting {gen} covariate model')
    g.fit(real_cov)
    print(f'Generating {gen} synthetic covariates')
    synth_cov = g.generate(count = len(real)).dataframe()

    #generate propensities
    X = np.array(real.drop([treatment_col, outcome_col], axis=1))
    y = np.array(real[treatment_col])
    
    classifier = LogisticRegression(random_state = random.randint(0, 1000000))
    print('Fitting propensity model')
    classifier.fit(X, y)
    print('Generating propensities')
    probabilities = classifier.predict_proba(np.array(synth_cov))
    prob_class_1 = probabilities[:, 1]
    binary_outcomes = np.random.binomial(n=1, p=prob_class_1)

    synth_cov_with_prop = synth_cov
    synth_cov_with_prop[treatment_col] = pd.Series(binary_outcomes)

    synth_cov_with_prop[outcome_col] = 0


    synth = synth_cov_with_prop
    X = np.array(real.drop([treatment_col, outcome_col], axis=1))
    y = np.array(real[outcome_col])
    w = np.array(real[treatment_col])
    n_units = len(real.drop([treatment_col, outcome_col], axis=1).columns)
    
    l = TLearner(n_unit_in=n_units, binary_y=False, seed=random.randint(0,1000000))
    print('Fitting CATE learner')
    l.fit(X, y, w)

    seq_X = np.array(synth.drop([treatment_col, outcome_col], axis=1))
    print('Generating POs')
    cate, y0, y1 = l.predict(seq_X, return_po=True)

    outcomes = []
    for index, value in synth[treatment_col].items():
        if value == 0:
            outcomes.append(y0[index].item())
        else:
            outcomes.append(y1[index].item())

    synth[outcome_col] = outcomes
    return synth

def hyperparam_test_nonlin(real, gen, treatment_col, outcome_col, hyperparams, n_iter, binary_y=False, save=False, fp=''):
    results = pd.DataFrame(columns = ['method', 'hyperparam', 'p_alpha_x', 'r_beta_x', 'jsd_pi', 'u_pehe'])
    for h in hyperparams:
        for _ in range(n_iter):
            stand = generate_stand_with_hyperparams_nonlin(real, gen, h)
            steam = generate_steam_with_hyperparams_nonlin(real, gen, treatment_col, outcome_col, h)
            n_units = len(real.drop([treatment_col, outcome_col], axis=1).columns)

            results.loc[len(results)] = ['standard', h, evaluate_f(real, stand, treatment_col, outcome_col), evaluate_c(real, stand, treatment_col, outcome_col), 
                                        evaluate_jsd(real, stand, treatment_col, outcome_col), evaluate_average_u_pehe(real, stand, treatment_col, outcome_col, n_units,binary_y)]
            results.loc[len(results)] = ['steam', h, evaluate_f(real, steam, treatment_col, outcome_col), evaluate_c(real, steam, treatment_col, outcome_col), 
                                        evaluate_jsd(real, steam, treatment_col, outcome_col), evaluate_average_u_pehe(real, steam, treatment_col, outcome_col, n_units,binary_y)]
            if save:
                results.to_csv(fp, index=False)
    
    return results

def hyperparam_exp_nonlin(real):
    results = hyperparam_test_nonlin(real, 'ctgan', 'treatment', 'y_factual', ['relu', 'leaky_relu', 'selu'], 5)
    results.to_csv('results/hyperparam_nonlin.csv', index=False)


#covariate shift exp

def cov_shift_example(mus, ncov, gen, n_iter):
    alpha = AlphaPrecision()
    kl = InverseKLDivergence()
    mmd = MaximumMeanDiscrepancy()
    results = pd.DataFrame(columns=['mu', 'alpha', 'beta', 'kl', 'mmd'])
    for mu in mus:
        for _ in range(n_iter):
            np.random.seed(_)
            X_1 = np.random.normal(loc = mu, size = (900, ncov))
            X_2 = np.random.normal(loc = -mu, size = (100, ncov))
            X = pd.DataFrame(np.row_stack([X_1, X_2]))
            g = Plugins().get(gen)
            g.fit(X)
            X_syn = g.generate(count = 1000)
            results.loc[len(results)] = [mu, alpha.evaluate(GenericDataLoader(X), X_syn)['delta_precision_alpha_OC'], alpha.evaluate(GenericDataLoader(X), X_syn)['delta_coverage_beta_OC'], 
                                        kl.evaluate(GenericDataLoader(X), X_syn)['marginal'], mmd.evaluate(GenericDataLoader(X), X_syn)['joint']]
    return results

def cov_shift_exp():
    results = cov_shift_example([2,3,4,5], 50, 'ddpm', n_iter = 5)
    results.to_csv('covariate_shift_results.csv', index=False)
    alpha = results.groupby('mu').mean()['alpha']
    beta = results.groupby('mu').mean()['beta']
    alpha_std = results.groupby('mu').std()['alpha']/np.sqrt(5) * 1.96
    beta_std = results.groupby('mu').std()['beta']/np.sqrt(5) * 1.96
    x = [2,3,4,5]
    plt.plot(x, alpha, label = 'Alpha precision')
    plt.plot(x, beta, label = 'Beta recall')

    plt.fill_between(x, alpha - alpha_std, alpha + alpha_std, alpha=0.2)
    plt.fill_between(x, beta - beta_std, beta + beta_std, alpha=0.2)
    plt.legend()
    plt.xlabel('$\mu$')
    plt.savefig('plots/covariate_shift_example.pdf', bbox_inches='tight')