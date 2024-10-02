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

    aids = pd.read_csv('Datasets/aids_preprocessed.csv')

    print("Starting experiments...")
    
    print("extreme illustrative failure of current metrics, in Appendix, results will be saved in results/current_metric_failure.csv")
    illustrative_failure()

    print("model selection experiment varying treatment assignment mechanism, in Apppendix, results saved to results/model_selection_treatment_assignment.csv")
    selection_treatment_assignment_exp()

    print("model selection experiment varying outcome generation mechanism, results saved to results/model_selection_outcome_generation.csv")
    selection_outcome_gen_exp()

    print("steam vs standard results. results for each dataset and gen model saved at  results/{dataset}_{gen}_steam_v_standard.csv")
    full_stream_vs_standard_exp(aids, ihdp, acic_encoded, jobs)

    print("num covariates insight experiment. results saved at results/num_cov_insight.csv, plots saved at plots/covariate_dimension_insight_jsd.pdf, plots/covariate_dimension_insight_U.pdf")
    run_cov_insight_exp()

    print("treatment assignment complexity insight experiment. results saved at results/treatment_assignment_insight.csv, plot saved at plots/treatment_assignment_insight.pdf")
    run_treatment_assignment_insight_exp()

    print("outcome generation complexity insight experiment. results saved at results/outcome_generation_insight.csv, plot saved at plots/predictive_insight_plot.pdf")
    run_outcome_gen_insight_exp()

    print("privacy experiment. results saved at results/privacy_results.csv. plots saved at plots/{metric}_with_epsilon.pdf")
    privacy_exp()

    print("ablation experiment, in Appendix, results in results/{d_name}_{gen}_ablation.csv")
    full_ablation_exp(aids, ihdp, acic_encoded)

    print("hyperparameter stability experiments, in Appendix, results in results/hyperparam_{hyperparam}.csv")
    hyperparam_exp_n_units(ihdp)
    hyperparam_exp_n_layers(ihdp)
    hyperparam_exp_nonlin(ihdp)

    print("covariate shift experiment. results saved at covariate_shift_results.csv, plot saved at plots/covariate_shift_example.pdf")
    cov_shift_exp()

    print("Experiments completed.")

if __name__ == "__main__":
    main()
