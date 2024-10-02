# %%
from synthcity.plugins import Plugins
import pandas as pd
import numpy as np
import random
from diffprivlib.models import LogisticRegression as LogisticRegressionDP
from sklearn.linear_model import LogisticRegression
from catenets.models.torch import *
from CATENets_dp.catenets_dp.models.torch import TLearner as TLearnerDP
from sklearn.preprocessing import OneHotEncoder

def encode(real):
    encoder = OneHotEncoder()
    categorical = ['x_2', 'x_21', 'x_24']
    encoded = encoder.fit_transform(real[categorical])
    encoded_df = pd.DataFrame(encoded.toarray(), columns=encoder.get_feature_names_out(categorical))
    real_encoded = pd.concat([real, encoded_df], axis=1)
    real_encoded.drop(categorical, axis=1, inplace=True)
    return real_encoded

def generate_sequentially_to_w(real, gen, treatment_col, outcome_col, private=False, epsilon=None, delta=None):
    random.seed()

    #generate covariates
    if private:
        g = Plugins().get(gen, random_state = random.randint(0, 1000000), epsilon=epsilon, delta=delta)
    else:
        g = Plugins().get(gen, random_state = random.randint(0, 1000000))
    real_cov = real.drop([treatment_col, outcome_col], axis=1)
    print(f'Fitting {gen} covariate model')
    g.fit(real_cov)
    print(f'Generating {gen} synthetic covariates')
    synth_cov = g.generate(count = len(real)).dataframe()

    #generate propensities
    X = np.array(real.drop([treatment_col, outcome_col], axis=1))
    y = np.array(real[treatment_col])
    
    if private:
        classifier = LogisticRegressionDP(random_state = random.randint(0, 1000000), epsilon=epsilon)
    else:
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

    return synth_cov_with_prop

# %%
def generate_sequentially(real, gen, treatment_col, outcome_col, private=False, epsilon=None, delta=None, binary_y=False):
    random.seed()
    synth = generate_sequentially_to_w(real, gen, treatment_col, outcome_col, private=private, epsilon=epsilon, delta=delta)

    X = np.array(real.drop([treatment_col, outcome_col], axis=1))
    y = np.array(real[outcome_col])
    w = np.array(real[treatment_col])
    n_units = len(real.drop([treatment_col, outcome_col], axis=1).columns)
    
    if private:
        l = TLearnerDP(n_unit_in=n_units, binary_y=binary_y, seed=random.randint(0,1000000), batch_norm=False)
        print('Fitting private CATE learner')
        l.fit(X, y, w, epsilon=epsilon, delta=delta)
    else:
        l = TLearner(n_unit_in=n_units, binary_y=binary_y, seed=random.randint(0,1000000))
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


# %%
def generate_standard(real, gen, private=False, epsilon=None, delta=None):
    random.seed()

    if private:
        g = Plugins().get(gen, random_state = random.randint(0, 1000000), epsilon=epsilon, delta=delta)
    else:
        g = Plugins().get(gen, random_state = random.randint(0, 1000000))
        
    print(f'Fitting {gen} model')
    g.fit(real)
    print(f'Generating {gen} synthetic dataset')
    synth = g.generate(count = len(real)).dataframe()
    return synth