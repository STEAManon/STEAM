import pandas as pd
from generation import *
from metrics import *
from exps import *



def main():
    print("Loading data...")
    ihdp_full = pd.read_csv('Datasets/ihdp.csv')
    ihdp = ihdp_full.drop(['y_cfactual', 'mu0', 'mu1'], axis=1)
    ihdp['treatment'] = ihdp['treatment'].astype(int)
    # %%
    jobs =pd.read_csv('Datasets/jobs_small.csv')

    # %%
    acic_full = pd.read_csv('Datasets/acic.csv')
    acic_full['y'] = acic_full['y0']
    acic_full.loc[acic_full['z']==1, 'y'] = acic_full.loc[acic_full['z']==1, 'y1']

    acic = acic_full.drop(['y0', 'y1', 'mu0', 'mu1'], axis=1)
    acic_encoded = encode(acic)

    print("Starting experiments...")
    
    print("illustrative failure of current metrics, results will be saved in current_metric_failure.csv")
    illustrative_failure()

    print("show how propensity metric tracks. results saved at propensity_metric_results.csv, plot saved at plots/propensity_metric_plot.pdf")
    propensity_plot()

    print("show how utility metric tracks. results saved at utility_metric_results.csv, plot saved at plots/utility_metric_plot.pdf")
    utility_plot()

    print("model selection using ours vs conventional metrics. results will be saved model_selection.csv")
    selection_exp()

    print("steam vs standard results. results for each dataset and gen model saved at  {dataset}_{gen}_steam_v_standard.csv")
    full_stream_vs_standard_exp(ihdp, jobs, acic_encoded)

    print("confounding insight experiment. results saved at confounding_insight.csv, plot saved at plots/confounding_complexity_plot.pdf")
    confounding_insight_exp()

    print("predictive insight experiment. results saved at predictive_insight.csv, plot saved at plots/CATE_complexity_plot.pdf")
    predictive_insight_exp()

    print("privacy experiment. results saved at privacy_results.csv. plots saved at plots/{metric}_with_epsilon.pdf")
    privacy_exp()

    print("covariate shift experiment. results saved at covariate_shift_results.csv, plot saved at plots/covariate_shift_example.pdf")
    cov_shift_exp()

    print("Experiments completed.")

if __name__ == "__main__":
    main()
