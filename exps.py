from generation import *
from metrics import *

from synthcity.metrics.eval_statistical import AlphaPrecision, InverseKLDivergence, MaximumMeanDiscrepancy
import matplotlib.pyplot as plt
from synthcity.metrics.eval_statistical import AlphaPrecision, InverseKLDivergence, MaximumMeanDiscrepancy
from synthcity.plugins.core.dataloader import GenericDataLoader
import random
from sklearn.metrics import mean_squared_error
from catenets.models.torch import *
from catenets.experiment_utils.simulation_utils import simulate_treatment_setup
from sklearn.linear_model import LogisticRegression
import numpy as np
import torch
import pandas as pd

def seq_test(real, gen, treatment_col, outcome_col, n_iter, private=False, epsilon=None, delta=None, binary_y=False, save=False, fp=''):
    results = pd.DataFrame(columns = ['method', 'f', 'c', 'd', 'u_pehe'])
    for _ in range(n_iter):
        stand = generate_standard(real, gen, private=private, epsilon=epsilon, delta=delta)
        seq_new = generate_sequentially(real, gen, treatment_col, outcome_col, private=private, epsilon=epsilon, delta=delta,binary_y=binary_y)
        n_units = len(real.drop([treatment_col, outcome_col], axis=1).columns)

        results.loc[len(results)] = ['standard', evaluate_f(real, stand, treatment_col, outcome_col), evaluate_c(real, stand, treatment_col, outcome_col), 
                                     evaluate_d(real, stand, treatment_col, outcome_col), evaluate_average_u_pehe(real, stand, treatment_col, outcome_col, n_units,binary_y)]
        results.loc[len(results)] = ['seq_new', evaluate_f(real, seq_new, treatment_col, outcome_col), evaluate_c(real, seq_new, treatment_col, outcome_col), 
                                     evaluate_d(real, seq_new, treatment_col, outcome_col), evaluate_average_u_pehe(real, seq_new, treatment_col, outcome_col, n_units,binary_y)]
        if save:
            results.to_csv(fp, index=False)
    
    return results


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
    
    res.to_csv('current_metric_failure.csv', index=False)


#our metrics work
def generate_propensities_datasets(n, correct):
    random.seed()
    X, y, w, p, t = simulate_treatment_setup(n, 20, seed=random.randint(0,1000000))
    X_df = pd.DataFrame(X)
    y_df = pd.DataFrame(y, columns = ['y'])
    w_df = pd.DataFrame(w, columns = ['w'])
    d = pd.concat([X_df,w_df,y_df], axis=1)

    X, y, w, p, t = simulate_treatment_setup(n, 20,seed=random.randint(0,1000000))
    X_df = pd.DataFrame(X)
    y_df = pd.DataFrame(y, columns = ['y'])
    w_df = pd.DataFrame(w, columns = ['w'])
    d_adv = pd.concat([X_df,w_df,y_df], axis=1)
    
    n_correct = round(correct*n)
    d_adv['w'] = 0
    d_adv.loc[:n_correct,'w'] = 1

    return d, d_adv

# %%
def new_metrics_propensity_exp(n, corrects, n_iter):
    results = pd.DataFrame(columns=['treated %', 'd'])
    alpha = AlphaPrecision()
    kl = InverseKLDivergence()

    for c in corrects:
        for _ in range(n_iter):
            d_real, d_adv = generate_propensities_datasets(n, c)

            d = evaluate_d(d_real,d_adv,'w','y')
            results.loc[len(results)] = [c,d]
            print('done')
    return results

def propensity_plot():
    results_prop = new_metrics_propensity_exp(10000, [0.01, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9, 0.99], 5)
    results_prop.to_csv('propensity_metric_results.csv', index=False)

    x = [0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]
    d = results_prop.groupby('treated %').mean()['d']
    d_std = results_prop.groupby('treated %').std()['d'] / np.sqrt(5) * 1.96
    plt.rcParams.update({'font.size': 30})
    plt.figure(figsize=(8, 6))
    plt.plot(x, d, 'o-', label = 'd')
    plt.fill_between(x, d - d_std, d + d_std, alpha=0.2)
    plt.ylim([0,1])
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.9)
    plt.ylabel('$D_{\pi}$')
    plt.xlabel('$\pi_{synth}$')
    plt.savefig('plots/propensity_metric_plot.pdf', bbox_inches='tight')

def create_simulated_datasets(n, d, n_o, n_t, n_known, error_sd=0):
    random.seed()
    X,y,w,p,t = simulate_treatment_setup(n, d=d, n_o=n_o, n_t=n_t, error_sd=error_sd, seed=random.randint(0,1000000))
    X_df = pd.DataFrame(X)
    y_df = pd.DataFrame(y, columns = ['y'])
    w_df = pd.DataFrame(w, columns = ['w'])
    d_real = pd.concat([X_df,w_df,y_df], axis=1)

    X,y,w,p,t = simulate_treatment_setup(n, d=d, n_o=n_o, n_t=n_t, error_sd=error_sd, seed=random.randint(0,1000000))
    X_df = pd.DataFrame(X)
    w_df = pd.DataFrame(w, columns = ['w'])
    d_synth = pd.concat([X_df,w_df], axis=1)
    
    mu1_coefs = np.ones(n_known)
    X_sel = X[:, n_o : n_o + n_known]
    mu1 = np.dot(X_sel**2, mu1_coefs)

    new_column_values = []
    for index, value in d_synth['w'].iteritems():
        if value == 0:
            new_column_values.append(0+ np.random.normal(scale=error_sd))
        else:
            new_column_values.append(mu1[index]+ np.random.normal(scale=error_sd)) 

    d_synth['y'] = new_column_values

    return d_real, d_synth

# %%
def run_utility_exp(n, d, n_o, n_t, n_knowns, n_iter, error_sd=0):
    results = pd.DataFrame(columns=['n_known', 'u_pehe'])
    for n_known in n_knowns:
        for _ in range(n_iter):
            d_real, d_synth = create_simulated_datasets(n, d,n_o, n_t, n_known)
            u_pehe = evaluate_average_u_pehe(d_real, d_synth, 'w', 'y', n_units=d)

            results.loc[len(results)] = [n_known, u_pehe]
            print(f'done {n_known} {_}')

    return results

def utility_plot():
    results = run_utility_exp(1000, 5, 0, 5, [1,2,3,4,5], n_iter = 5)
    results.to_csv('utility_metric_results.csv', index=False)

    x = [1,2,3,4,5]
    u_pehe = results.groupby('n_known').mean()['u_pehe']
    u_pehe_std = results.groupby('n_known').std()['u_pehe'] / np.sqrt(5) * 1.96

    plt.figure(figsize=(8, 6))
    plt.plot(x, u_pehe, 'o-', label = 'PEHE', color='g')
    plt.ylabel('$U_{PEHE}$')
    plt.fill_between(x, u_pehe - u_pehe_std, u_pehe + u_pehe_std, alpha=0.2, color='g')
    plt.xlabel('Correct predictive covariates')
    plt.xticks([1,2,3,4,5])
    plt.savefig('plots/utility_metric_plot.pdf', bbox_inches='tight')

#selection
def selection_test(n_iter, save=False, fp=''):
    kl = InverseKLDivergence()
    mmd = MaximumMeanDiscrepancy()
    alpha = AlphaPrecision()
    results = pd.DataFrame(columns=['outcome learner', 'alpha', 'beta', 'kl', 'mmd', 'u', 'oracle'])
    for _ in range(n_iter):
        n=1000;d=5;n_c=0;n_t=1
        X,y,w,p,cate = simulate_treatment_setup(n, d=d, n_c=n_c, n_t=n_t)
        X_df = pd.DataFrame(X)
        y_df = pd.DataFrame(y, columns = ['y'])
        w_df = pd.DataFrame(w, columns = ['w'])
        d_real = pd.concat([X_df,w_df,y_df], axis=1)
        synth_cov_with_prop = generate_sequentially_to_w(d_real, 'ddpm', 'w', 'y')
        seq_X = np.array(synth_cov_with_prop.drop(['w', 'y'], axis=1))

        t = synth_cov_with_prop.copy()
        l = TLearner(n_unit_in=d, binary_y=False, batch_norm=False)
        l.fit(X, y, w)
        _, y0, y1 = l.predict(seq_X, return_po=True)
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
        _, y0, y1 = l.predict(seq_X, return_po=True)
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
        _, y0, y1 = l.predict(seq_X, return_po=True)
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
        _, y0, y1 = l.predict(seq_X, return_po=True)
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
                                     kl.evaluate(GenericDataLoader(d_real), GenericDataLoader(t))['marginal'], mmd.evaluate(GenericDataLoader(d_real), GenericDataLoader(t))['joint'], 
                                     evaluate_average_u_pehe(d_real, t, 'w', 'y', d), oracle_t]
        results.loc[len(results)] = ['s', alpha.evaluate(GenericDataLoader(d_real), GenericDataLoader(s))['delta_precision_alpha_OC'], alpha.evaluate(GenericDataLoader(d_real), GenericDataLoader(s))['delta_coverage_beta_OC'],
                                     kl.evaluate(GenericDataLoader(d_real), GenericDataLoader(s))['marginal'], mmd.evaluate(GenericDataLoader(d_real), GenericDataLoader(s))['joint'], 
                                     evaluate_average_u_pehe(d_real, s, 'w', 'y', d), oracle_s]
        results.loc[len(results)] = ['dr', alpha.evaluate(GenericDataLoader(d_real), GenericDataLoader(dr))['delta_precision_alpha_OC'], alpha.evaluate(GenericDataLoader(d_real), GenericDataLoader(dr))['delta_coverage_beta_OC'],
                                     kl.evaluate(GenericDataLoader(d_real), GenericDataLoader(dr))['marginal'], mmd.evaluate(GenericDataLoader(d_real), GenericDataLoader(dr))['joint'], 
                                     evaluate_average_u_pehe(d_real, dr, 'w', 'y', d), oracle_dr]
        results.loc[len(results)] = ['tar', alpha.evaluate(GenericDataLoader(d_real), GenericDataLoader(tar))['delta_precision_alpha_OC'], alpha.evaluate(GenericDataLoader(d_real), GenericDataLoader(tar))['delta_coverage_beta_OC'],
                                     kl.evaluate(GenericDataLoader(d_real), GenericDataLoader(tar))['marginal'], mmd.evaluate(GenericDataLoader(d_real), GenericDataLoader(tar))['joint'], 
                                     evaluate_average_u_pehe(d_real, tar, 'w', 'y', d), oracle_tar]
        if save:
            results.to_csv(fp, index=False)
    return results

def selection_exp():
    results = selection_test(10)
    results.to_csv('model_selection.csv', index=False)

#steam performance
def steam_v_standard(real, gen, treatment_col, outcome_col, n_iter, d_name):
    results = seq_test(real, gen, treatment_col, outcome_col, n_iter)
    results.to_csv(f'{d_name}_{gen}_steam_v_standard.csv', index=False)

def full_stream_vs_standard_exp(ihdp, jobs, acic):
    steam_v_standard(ihdp, 'ddpm', 'treatment', 'y_factual', 20, 'ihdp')
    steam_v_standard(ihdp, 'ctgan', 'treatment', 'y_factual', 20, 'ihdp')
    steam_v_standard(ihdp, 'tvae', 'treatment', 'y_factual', 20, 'ihdp')

    steam_v_standard(jobs, 'ddpm', 'training', 're78', 20, 'jobs')
    steam_v_standard(jobs, 'ctgan', 'training', 're78', 20, 'jobs')
    steam_v_standard(jobs, 'tvae', 'training', 're78', 20, 'jobs')

    steam_v_standard(acic, 'ddpm', 'z', 'y', 20, 'acic')
    steam_v_standard(acic, 'ctgan', 'z', 'y', 20, 'acic')
    steam_v_standard(acic, 'tvae', 'z', 'y', 20, 'acic')

#insights

def confounding_insight(n, d, n_t, n_cs, gen, n_iter, save=False, fp=''):
    results = pd.DataFrame(columns=['method', 'f', 'c', 'd', 'u_pehe', 'n_c'])
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


def confounding_insight_exp():
    conf_results = confounding_insight(1000, 10, 5, [1,2,3,4,5], 'ddpm', 10)
    conf_results.to_csv('confounding_insight.csv', index=False)
    x = [1,2,3,4,5]
    d_seq = conf_results[conf_results['method']=='seq_new'].groupby(['n_c']).mean()['d']
    d_stand = conf_results[conf_results['method']=='standard'].groupby(['n_c']).mean()['d']

    d_seq_std = conf_results[conf_results['method']=='seq_new'].groupby(['n_c']).std()['d'] / np.sqrt(10) * 1.96
    d_stand_std = conf_results[conf_results['method']=='standard'].groupby(['n_c']).std()['d']/ np.sqrt(10) * 1.96

    plt.rcParams.update({'font.size': 30})
    plt.figure(figsize=(8, 6))
    plt.plot(x, d_stand, 'o-', label = 'Standard')
    plt.plot(x, d_seq, 'o-', label = 'STEAM')


    plt.fill_between(x, d_stand - d_stand_std, d_stand + d_stand_std, alpha=0.2)
    plt.fill_between(x, d_seq - d_seq_std, d_seq + d_seq_std, alpha=0.2)
    plt.legend()
    plt.xlabel('# confounding covariates')
    plt.ylabel('$D_\pi$')
    plt.ylim([0.6,1])
    plt.xticks([1,2,3,4,5])
    plt.savefig('plots/confounding_complexity_plot.pdf', bbox_inches='tight')


def predictive_insight(n, d, n_o, n_ts, gen, n_iter, save=False, fp=''):
    results = pd.DataFrame(columns=['method', 'f', 'c', 'd', 'u_pehe', 'n_t'])
    for n_t in n_ts:
        X,y,w,p,t = simulate_treatment_setup(n, d=d, n_o=n_o, n_t=n_t)
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

def predictive_insight_exp():
    pred_results = predictive_insight(1000, 10, 5, [1,2,3,4,5], 'ddpm', 10)
    pred_results.to_csv('predictive_insight.csv', index=False)

    x = [1,2,3,4,5]
    u_seq = pred_results[pred_results['method']=='seq_new'].groupby(['n_t']).mean()['u_pehe']
    u_stand = pred_results[pred_results['method']=='standard'].groupby(['n_t']).mean()['u_pehe']

    u_seq_std = pred_results[pred_results['method']=='seq_new'].groupby(['n_t']).std()['u_pehe'] / np.sqrt(10) * 1.96
    u_stand_std = pred_results[pred_results['method']=='standard'].groupby(['n_t']).std()['u_pehe']/ np.sqrt(10) * 1.96

    plt.figure(figsize=(8, 6))
    plt.plot(x, u_stand, 'o-', label = 'Standard')
    plt.plot(x, u_seq, 'o-', label = 'STEAM')
    plt.fill_between(x, u_stand - u_stand_std, u_stand + u_stand_std, alpha=0.2)
    plt.fill_between(x, u_seq - u_seq_std, u_seq + u_seq_std, alpha=0.2)
    plt.xlabel('# predictive covariates')
    plt.ylabel('$U_{PEHE}$')
    plt.xticks([1,2,3,4,5])
    plt.savefig('plots/CATE_complexity_plot.pdf', bbox_inches='tight')


#privacy
def privacy_comparison(real, gen, treatment_col, outcome_col, delta, epsilons, n_iter, save=False, fp=''):
    results = pd.DataFrame(columns=['method', 'epsilon', 'f', 'c', 'd', 'avg_u_pehe'])
    n_units = len(real.drop([treatment_col, outcome_col], axis=1).columns)
    for epsilon in epsilons:
        for _ in range(n_iter):
            synth = generate_standard(real, gen, private=True, epsilon=epsilon, delta=delta)

            results.loc[len(results)] = ['standard', epsilon, evaluate_f(real, synth, treatment_col, outcome_col), evaluate_c(real, synth, treatment_col, outcome_col), 
                                     evaluate_d(real, synth, treatment_col, outcome_col), evaluate_average_u_pehe(real, synth, treatment_col, outcome_col, n_units)]

            synth_seq = generate_sequentially(real, gen, treatment_col, outcome_col, private=True, epsilon = epsilon/3, delta=delta/3)

            results.loc[len(results)] = ['sequential', epsilon, evaluate_f(real, synth_seq, treatment_col, outcome_col), evaluate_c(real, synth_seq, treatment_col, outcome_col), 
                                     evaluate_d(real, synth_seq, treatment_col, outcome_col), evaluate_average_u_pehe(real, synth_seq, treatment_col, outcome_col, n_units)]
    
            if save:
                print('saving results')
                results.to_csv(fp, index=False)
    return results

def privacy_exp():
    X,y,w,p,t = simulate_treatment_setup(1000, d=5, n_t=3, n_c=2)
    X_df = pd.DataFrame(X)
    y_df = pd.DataFrame(y, columns = ['y'])
    w_df = pd.DataFrame(w, columns = ['w'])
    d_real = pd.concat([X_df,w_df,y_df], axis=1)

    # %%
    results = privacy_comparison(d_real, 'dpgan', 'w', 'y', delta = 1e-3, epsilons = [1,2,3,4,5,10,15], n_iter = 10)
    results.to_csv('privacy_results.csv', index=False)

    results_standard = results[results['method']=='standard']

    # %%
    results_seq = results[results['method']=='sequential']

    # %%
    x = [1,2,3,4,5,10,15]
    d_stand = results_standard.groupby('epsilon').mean()['d']
    d_seq = results_seq.groupby('epsilon').mean()['d']
    f_stand = results_standard.groupby('epsilon').mean()['f']
    f_seq = results_seq.groupby('epsilon').mean()['f']
    c_stand = results_standard.groupby('epsilon').mean()['c']
    c_seq = results_seq.groupby('epsilon').mean()['c']
    u_stand = results_standard.groupby('epsilon').mean()['avg_u_pehe']
    u_seq = results_seq.groupby('epsilon').mean()['avg_u_pehe']


    d_stand_std = results_standard.groupby('epsilon').std()['d'] / np.sqrt(10) * 1.96
    d_seq_std = results_seq.groupby('epsilon').std()['d']/ np.sqrt(10)* 1.96
    f_stand_std = results_standard.groupby('epsilon').std()['f']/ np.sqrt(10)* 1.96
    f_seq_std = results_seq.groupby('epsilon').std()['f']/ np.sqrt(10)* 1.96
    c_stand_std = results_standard.groupby('epsilon').std()['c']/ np.sqrt(10)* 1.96
    c_seq_std = results_seq.groupby('epsilon').std()['c']/ np.sqrt(10)* 1.96
    u_stand_std = results_standard.groupby('epsilon').std()['avg_u_pehe']/ np.sqrt(10)* 1.96
    u_seq_std = results_seq.groupby('epsilon').std()['avg_u_pehe']/ np.sqrt(10)* 1.96

    # %%
    plt.figure(figsize=(8, 6))
    plt.plot(x, d_stand, 'o-', label = 'Standard')
    plt.plot(x, d_seq, 'o-', label = 'STEAM')
    plt.fill_between(x, d_stand - d_stand_std, d_stand + d_stand_std, alpha=0.2)
    plt.fill_between(x, d_seq - d_seq_std, d_seq + d_seq_std, alpha=0.2)
    plt.ylim([0,1])
    plt.ylabel('$D_\pi$')
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
    plt.ylabel('$F$')
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
    plt.ylabel('$C$')
    plt.xlabel('$\epsilon$')
    plt.xscale('log')
    plt.savefig('plots/C_with_epsilon.pdf', bbox_inches='tight')

    # %%
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

    

#covariate shift

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
