# %%
from synthcity.metrics.eval_statistical import AlphaPrecision, InverseKLDivergence, MaximumMeanDiscrepancy
from synthcity.plugins.core.dataloader import GenericDataLoader
import random
from sklearn.metrics import mean_squared_error
from catenets.models.torch import *
from catenets.experiment_utils.simulation_utils import simulate_treatment_setup
from sklearn.linear_model import LogisticRegression
import numpy as np
import torch
from scipy.spatial.distance import jensenshannon

def evaluate_f(real, synth, treatment_col, outcome_col):
    alpha = AlphaPrecision(random_state = random.randint(0, 1000000))

    real_cov = real.drop([treatment_col, outcome_col], axis=1)
    synth_cov = synth.drop([treatment_col, outcome_col], axis=1)

    f = alpha.evaluate(GenericDataLoader(real_cov), GenericDataLoader(synth_cov))['delta_precision_alpha_OC']

    return f

# %%
def evaluate_c(real, synth, treatment_col, outcome_col):
    alpha = AlphaPrecision(random_state = random.randint(0, 1000000))

    real_cov = real.drop([treatment_col, outcome_col], axis=1)
    synth_cov = synth.drop([treatment_col, outcome_col], axis=1)

    c = alpha.evaluate(GenericDataLoader(real_cov), GenericDataLoader(synth_cov))['delta_coverage_beta_OC']

    return c


# %%
def train_propensity_function(real, treatment_col, outcome_col, clf, avg=False):
    X = np.array(real.drop([treatment_col, outcome_col], axis=1))
    y = np.array(real[treatment_col])
    
    if not avg:
        clf.fit(X, y)
        return clf
    
    trained_clfs = []
    for i in clf:
        i.fit(X,y)
        trained_clfs.append(i)
    
    return trained_clfs


# %%
def get_jsd(real, synth):
    jsd = 0
    for i in range(len(real)):
        jsd_i = jensenshannon(real[i], synth[i], base=2)
        jsd += jsd_i
    
    return jsd / len(real)
# %%
def evaluate_jsd(real, synth, treatment_col, outcome_col):
    n = len(real)
    n_test = 0.2*n
    test = real[:round(n_test)]
    real = real[round(n_test):]

    pi_real = LogisticRegression()
    pi_synth = LogisticRegression()
    pi_real = train_propensity_function(real, treatment_col, outcome_col, pi_real)
    pi_synth = train_propensity_function(synth, treatment_col, outcome_col, pi_synth)

    probabilities_real = pi_real.predict_proba(np.array(test.drop([treatment_col, outcome_col], axis=1)))
    probabilities_synth = pi_synth.predict_proba(np.array(test.drop([treatment_col, outcome_col], axis=1)))

    return 1 - get_jsd(probabilities_real, probabilities_synth)


# %%
def evaluate_average_u_pehe(real, synth, treatment_col, outcome_col, n_units, binary_y = False):
    n = len(real)
    n_test = 0.2*n
    test = real[:round(n_test)]
    real = real[round(n_test):]

    real_learners = [TLearner(n_unit_in=n_units, binary_y=binary_y, seed=random.randint(0,1000000)), SLearner(n_unit_in=n_units, binary_y=binary_y, seed=random.randint(0,1000000)), 
                DRLearner(n_unit_in=n_units, binary_y=binary_y, seed=random.randint(0,1000000)), RALearner(n_unit_in=n_units, binary_y=binary_y,seed=random.randint(0,1000000))]
    synth_learners = [TLearner(n_unit_in=n_units, binary_y=binary_y, seed=random.randint(0,1000000)), SLearner(n_unit_in=n_units, binary_y=binary_y, seed=random.randint(0,1000000)), 
                DRLearner(n_unit_in=n_units, binary_y=binary_y, seed=random.randint(0,1000000)), RALearner(n_unit_in=n_units, binary_y=binary_y,seed=random.randint(0,1000000))]

    avg_pehe = 0
    for i in range(len(real_learners)):
        l_real = real_learners[i]
        l_synth = synth_learners[i]
        X_real = np.array(real.drop([treatment_col, outcome_col], axis=1))
        y_real = np.array(real[outcome_col])
        w_real = np.array(real[treatment_col])
        l_real.fit(X_real, y_real, w_real)
        X_t = np.array(test.drop([treatment_col, outcome_col], axis=1))
        X_t = torch.tensor(X_t)
        pred_real = l_real.predict(X_t)

        X_synth = np.array(synth.drop([treatment_col, outcome_col], axis=1))
        y_synth = np.array(synth[outcome_col])
        w_synth = np.array(synth[treatment_col])
        l_synth.fit(X_synth, y_synth, w_synth)
        pred_synth = l_synth.predict(X_t)

        pehe = mean_squared_error(pred_real.cpu().detach().numpy(), pred_synth.cpu().detach().numpy(), squared=False)
        avg_pehe += pehe

    return avg_pehe / len(real_learners)