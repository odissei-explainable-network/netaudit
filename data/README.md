# Data

## Export

This folder contains aggregated data exported from the remote access environment by Statistics 
Netherlands. The data has been checked approved by Statistics Netherlands for publication. See this [link](https://www.cbs.nl/nl-nl/onze-diensten/maatwerk-en-microdata/microdata-zelf-onderzoek-doen) for details.

- `*_community_edgelist.csv`: Contains average edge utilities between municipalities in the Netherlands. The municipality names can be linked to the municipality IDs through the `*_municipalities_stats.csv` file. Prefix of filenames indicates embeddings for which edge utilities were calculated.
- `*_municipalities_stats.csv`: Contains names, IDs, and aggregated statistics for each municipality.
- `edge_covariates.csv`: Contains edge utility statistics for different relation types.
- `edge_similarity_age.csv`: Contains the average edge utility and the average difference in age (years) for different edge utility deciles.
- `edge_similarity_education.csv`: Contains the average edge utility in different edge utility deciles for differences in achieved highest education.
- `edge_similarity_gender.csv`: Contains the average edge utility in different edge utility deciles for differences in gender.
- `edge_similarity_income.csv`: Contains the average edge utility and the average difference in gross income percentile for different edge utility deciles.
- `edge_similarity_parents_abroad.csv`: Contains the average edge utility in different edge utility deciles for differences in the number of parents born abroad.
- `feature_importance_cv24p013_trust_government.csv`: Contains average SHAP values and average predictor values in different SHAP value deciles for different predictors of trust in the government.
- `feature_importance_cv24p307_voting_behavior_populist.csv`: Contains average SHAP values and average predictor values in different SHAP value deciles for different predictors of right-wing populist voting.
- `liss_preprocessed_categorical_summary.csv`: Summary statistics for categorical variables in imported, linked LISS panel data.
- `liss_preprocessed_correlations.csv`: Bivariate correlations between variables in imported, linked LISS panel data.  For pairs of numerical variables, Pearson correlations were calculated. For pairs
of numerical and categorical variables, point-biserial correlations were calculated. For
pairs of categorical variables, Cramer’s V was calculated.
- `liss_preprocessed_numeric_summary.csv`: Summary statistics for continuous variables in imported, linked LISS panel data.
- `node_covariates_age_income.csv`: Correlations between node-level edge utility statistics and age (years) and gross income percentile.
- `node_covariates_education.csv`: Aggregated node-level edge utility statistics for different highest achieved education levels.
- `node_covariates_education.csv`: Aggregated node-level edge utility statistics for different genders.
- `node_covariates_education.csv`: Aggregated node-level edge utility statistics for different number of parents born abroad.
- `prediction_performance_cv24p013_trust_government_scores.csv`: Prediction scores for trust in the government.
- `prediction_performance_cv24p307_voting_behavior_populist_scores.csv`: Prediction scores for right-wing populist voting.
- `prediction_performance_cv24p307_voting_behavior_scores.csv`: Prediction scores for general voting.

## Models

Model code and checkpoints for the Bayesian regression in the prediction performance analysis.
