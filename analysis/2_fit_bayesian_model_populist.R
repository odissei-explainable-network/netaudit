##########################################################################
# Fit a Bayesian regression model to prediction scores for populist voting
##########################################################################

library(dplyr)
library(tidyr)
library(stringr)
library(forcats)
library(ggplot2)
library(brms)
library(tidybayes)
library(bayesplot)
library(cowplot)
library(marginaleffects)
library(tinytable)

# Read preprocessed prediction scores
df_combined <- read.csv(
    file.path("data", "preprocessed", "prediction_scores.csv"),
    na.strings = "",
    stringsAsFactors = TRUE
) |>
    # marginaleffects does not like variables named 'model'
    rename(feature_set = model) |>
    # Replace Inf values with NA, keeping numeric column types
    mutate(across(c(populist, trust, voting), ~ if_else(is.infinite(.x), NA, .x)))

# Create model formula for populist voting
model_formula <- bf(
    # Account for missing target observations
    populist | mi() ~ estimator * feature_set + year + is_dine +
        num_walks * walk_length + window_size + dimension,
    phi ~ estimator + feature_set,
    family = Beta()
)

# Define priors for intercepts and coefficients of mean and variance
prior_mv <- c(
    set_prior("normal(0, 2.5)", class = "b"),
    set_prior("normal(0, 2.5)", class = "Intercept"),
    set_prior("exponential(1)", class = "b", dpar = "phi", lb = 0),
    set_prior("exponential(1)", class = "Intercept", dpar = "phi", lb = 0)
)

# Constrain priors of NA levels to zero
prior_mv <- c(
    prior_mv,
    set_prior("constant(0)", coef = "num_walks100:walk_lengthNA"),
    set_prior("constant(0)", coef = "num_walksNA:walk_lengthNA"),
    set_prior("constant(0)", coef = "num_walksNA:walk_length5"),
    set_prior("constant(0)", coef = "num_walksNA:walk_length20")
)

for (predictor in c("year", "is_dine", "num_walks", "walk_length", "window_size", "dimension")) {
    prior_mv <- c(prior_mv, set_prior("constant(0)", coef = paste0(predictor, "NA")))
}

# Fit model
model <- brm(
    model_formula,
    data = df_combined,
    family = Beta(),
    prior = prior_mv,
    init = 0,
    chains = 4,
    iter = 2000,
    cores = 4,
    seed = 2024,
    backend = "cmdstanr",
    stan_model_args = list(stanc_options = list("O1")),
    save_model = file.path("data", "models", "model_populist.stan"),
    file =  file.path("data", "models", "model_populist"),
    file_refit = "on_change"
)

summary(model)

# Create figure for posterior predictive distribution
plot_grid(
    pp_check(model, prefix = "ppc", ndraws = 100) +
        labs(x = "Macro AUC score", y = "Density"),
    pp_check(model, type = "ecdf_overlay", ndraws = 100) +
        labs(x = "Macro AUC score", y = "Cumulative density"),
    ncol = 1,
    labels = c("A", "B")
)

ggsave(file.path("figures", "posterior_predictive_check_populist_voting.png"), width = 10, height = 6)

# Define function to get posterior predictive draws
get_posterior_predictive_draws <- function(model, n_draws = 1000) {
    return(
        model |>
            posterior_predict(ndraws = n_draws) |>
            t() |>
            as.data.frame() |>
            cbind(df_combined) |>
            pivot_longer(
                cols = starts_with("V"),
                names_to = ".draw",
                values_to = ".pred",
                names_prefix = "V"
            )
    )
}

# Get posterior predictive draws
df_pred_populist <- model |>
    get_posterior_predictive_draws()

# Compute means and quantile credible intervals for each feature set
rbind(
    # For all model predictions
    df_pred_populist |>
        group_by(feature_set) |>
        mean_qi(populist, .width = c(0.95), na.rm = TRUE) |>
        mutate(subset = "All"),
    # For best subset of model predictions
    df_pred_populist |>
        filter((estimator == "xgb" & walk_length == 20 & num_walks == 100 & dimension == 32) |
                   (estimator == "xgb" & feature_set == "covariates")) |>
        group_by(feature_set) |>
        mean_qi(populist, .width = c(0.95), na.rm = TRUE) |>
        mutate(subset = "Best subset")
) |>
    # Rename feature set levels
    mutate(feature_set = factor(
        feature_set,
        levels = c("embeddings", "covariates", "embeddings_covariates"),
        labels = c("Embeddings", "Covariates", "Embeddings+\ncovariates")
    )) |>
    # Create plot
    ggplot(aes(x = feature_set, y = populist, ymin = .lower, ymax = .upper, color = subset)) +
    geom_errorbar(alpha = 0.5, position = position_dodge(width = 0.2), width = 0.2, size = 1) +
    geom_point(size = 2, color = "black", position = position_dodge2(width = 0.2)) +
    labs(x = "Feature set", y = "Macro AUC score") +
    scale_y_continuous(limits = c(0.5, 0.8), breaks = seq(0.5, 0.8, 0.1)) +
    scale_color_manual(name = "", values = c("blue", "red")) +
    theme_half_open() +
    theme(
        axis.line.x = element_blank(),
        axis.ticks.x = element_blank()
    )

ggsave(path("figures", "prediction_performance_populist.png"), width = 6, height = 4)

# Calc pairwise contrasts of empirical posterior predictions
# (i.e., for observed combinations of varialbe levels)
contr_feature_set <- model |>
    avg_predictions(by = "feature_set", hypothesis = difference ~ pairwise)

print(contr_feature_set, digits = 2)

# Calc contrasts for best subset
contr_feature_set_sub <- model |>
    avg_predictions(
        by = "feature_set",
        hypothesis = difference ~ pairwise,
        newdata = subset((estimator == "xgb" & walk_length == 20 & num_walks == 100 & dimension == 32) | (estimator == "xgb" & feature_set == "covariates"))
    )

print(contr_feature_set_sub, digits = 2)
