# Fit a Bayesian regression model to prediction scores ----------------------------------------

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

# Define function for creating model formulas
create_model_formula <- function(response_var, family, estimate_missing = TRUE) {
  if (estimate_missing) {
    resp <- paste(response_var, "mi()", sep = " | ") # Account for missing target observations
  } else {
    resp <- response_var
  }

  formula_mean <- reformulate(
    c("estimator * feature_set", "year", "is_dine", "num_walks * walk_length",
      "window_size", "dimension"),
    response = resp
  )

  formula_phi <- reformulate(
    c("estimator", "feature_set"),
    response = "phi"
  )

  formula_model <- bf(
    formula_mean,
    formula_phi,
    family = family
  )

  return(formula_model)
}

# Create model formula for populist voting
model_formula_populist <- create_model_formula("populist", Beta())

# Define function for priors
create_model_prior <- function() {
  # Define priors for intercepts and coefficients of mean and variance
  model_prior <- c(
    set_prior("normal(0, 2.5)", class = "b"),
    set_prior("normal(0, 2.5)", class = "Intercept"),
    set_prior("exponential(1)", class = "b", dpar = "phi", lb = 0),
    set_prior("exponential(1)", class = "Intercept", dpar = "phi", lb = 0)
  )

  # Constrain priors of NA levels to zero
  model_prior <- c(
    model_prior,
    set_prior("constant(0)", coef = "num_walks100:walk_lengthNA"),
    set_prior("constant(0)", coef = "num_walksNA:walk_lengthNA"),
    set_prior("constant(0)", coef = "num_walksNA:walk_length5"),
    set_prior("constant(0)", coef = "num_walksNA:walk_length20")
  )

  for (predictor in c("year", "is_dine", "num_walks", "walk_length", "window_size", "dimension")) {
    model_prior <- c(model_prior, set_prior("constant(0)", coef = paste0(predictor, "NA")))
  }

  return(model_prior)
}

model_prior <- create_model_prior()

init <- 0
num_chains <- 4
num_samples <- 2000
seed <- 2024

# Fit model
model_populist <- brm(
  model_formula_populist,
  data = df_combined,
  family = Beta(),
  prior = model_prior,
  init = init,
  chains = num_chains,
  iter = num_samples,
  cores = 4,
  seed = seed,
  backend = "cmdstanr",
  stan_model_args = list(stanc_options = list("O1")),
  save_model = file.path("data", "models", "model_populist.stan"),
  file =  file.path("data", "models", "model_populist"),
  file_refit = "on_change"
)

summary(model_populist)

# Create figure for posterior predictive distribution
plot_grid(
  pp_check(model_populist, prefix = "ppc", ndraws = 100) +
    labs(x = "Macro AUC score", y = "Density"),
  pp_check(model_populist, type = "ecdf_overlay", ndraws = 100) +
    labs(x = "Macro AUC score", y = "Cumulative density"),
  ncol = 1,
  labels = c("A", "B")
)

ggsave(file.path("figures", "posterior_predictive_check_populist_voting.png"), width = 10, height = 6)


# Plot posterior predictions ------------------------------------------------------------------

# Define function to get posterior predictive draws
get_posterior_predictive_draws <- function(model, df, n_draws = 1000) {
  return(
    model |>
      posterior_predict(ndraws = n_draws) |>
      t() |>
      as.data.frame() |>
      cbind(df) |>
      pivot_longer(
        cols = starts_with("V", ignore.case = FALSE),
        names_to = ".draw",
        values_to = ".pred",
        names_prefix = "V"
      )
  )
}

# Get posterior predictive draws
df_pred_populist <- model_populist |>
  get_posterior_predictive_draws(df_combined)

create_posterior_predictive_plot <- function(df_pred, response_var) {
  response_var <- enquo(response_var)
  # Compute means and quantile credible intervals for each feature set
  p <- rbind(
    # For all model predictions
    df_pred |>
      group_by(feature_set) |>
      mean_qi(!! response_var, .width = c(0.95), na.rm = TRUE) |>
      mutate(subset = "All"),
    # For best subset of model predictions
    df_pred |>
      filter((estimator == "xgb" & walk_length == 20 & num_walks == 100 & dimension == 32) |
               (estimator == "xgb" & feature_set == "covariates")) |>
      group_by(feature_set) |>
      mean_qi(!! response_var, .width = c(0.95), na.rm = TRUE) |>
      mutate(subset = "Best subset")
  ) |>
    # Rename feature set levels
    mutate(feature_set = factor(
      feature_set,
      levels = c("embeddings", "covariates", "embeddings_covariates"),
      labels = c("Embeddings", "Covariates", "Embeddings+\ncovariates")
    )) |>
    # Create plot
    ggplot(aes(x = feature_set, y = !! response_var, ymin = .lower, ymax = .upper, color = subset)) +
    geom_errorbar(alpha = 0.5, position = position_dodge(width = 0.2), width = 0.2, size = 1) +
    geom_point(size = 2, color = "black", position = position_dodge2(width = 0.2)) +
    labs(x = "Feature set") +
    scale_y_continuous(limits = c(0.5, 0.8), breaks = seq(0.5, 0.8, 0.1)) +
    scale_color_manual(name = "", values = c("purple", "darkgreen")) +
    theme_half_open() +
    theme(
      axis.line.x = element_blank(),
      axis.ticks.x = element_blank()
    )

  return(p)
}

create_posterior_predictive_plot(df_pred_populist, populist) +
  ylab("Macro AUC score")

ggsave(file.path("figures", "prediction_performance_populist.png"), width = 6, height = 4)


# Calculate contrasts -------------------------------------------------------------------------

options(tinytable_theme_placement_latex_float = "htbp!")

# Calc pairwise contrasts of empirical posterior predictions
# (i.e., for observed combinations of variable levels)
contr_feature_set_populist <- model_populist |>
  avg_predictions(by = "feature_set", hypothesis = difference ~ pairwise)

print(contr_feature_set_populist, digits = 2)

# Calc contrasts for best subset
contr_feature_set_sub_populist <- model_populist |>
  avg_predictions(
    by = "feature_set",
    hypothesis = difference ~ pairwise,
    newdata = subset((estimator == "xgb" & walk_length == 20 & num_walks == 100 & dimension == 32) |
                       (estimator == "xgb" & feature_set == "covariates"))
  )

print(contr_feature_set_sub_populist, digits = 2)


# Create contrast table for appendix ----------------------------------------------------------

# Simple contrasts

calc_simple_contrasts <- function(model) {
  contr_vars_simple <- list("estimator", "feature_set", "year", "is_dine",
                            "walk_length", "num_walks", "window_size")

  df_contr_simple <- Reduce(rbind, lapply(contr_vars_simple, function(v) {
    return(
      model |>
        avg_predictions(
          by = v,
          hypothesis = difference ~ pairwise
        ) |>
        mutate(by = v)
    )
  })) |>
    rbind(
      # Average over dimensions from DeepWalk vs. LINE
      avg_predictions(
        model,
        by = "dimension",
        hypothesis = "(b1 + b4 + b5) / 3 = (b2 + b3 + b6) /  3"
      ) |>
        mutate(hypothesis = "deepwalk - line", by = "method"),
      avg_predictions(
        model,
        by = "dimension",
        hypothesis = difference ~ pairwise,
        newdata = subset(dimension %in% c(8, 16, 24))
      ) |>
        mutate(by = "dimension"),
      avg_predictions(
        model,
        by = "dimension",
        hypothesis = difference ~ pairwise,
        newdata = subset(dimension %in% c(32, 64, 128))
      ) |>
        mutate(by = "dimension")
    )

  df_contr_simple_formatted <- df_contr_simple |>
    as.data.frame() |>
    mutate(var = str_split(hypothesis, " - ", n = 2)) |>
    unnest_wider(var, names_sep = "_level_") |>
    mutate(across(c(var_level_1, var_level_2), ~ str_extract(.x, "([a-z,_]+)|([0-9]+)"))) |>
    filter(!is.na(var_level_1), !is.na(var_level_2)) |>
    mutate(
      across(c(var_level_1, var_level_2), ~ case_match(
        .x,
        "xgb" ~ "XGB",
        "knn" ~ "KNN",
        "lm" ~ "LinReg",
        "embeddings_covariates" ~ "Embed + Cov",
        "embeddings" ~ "Embed",
        "covariates" ~ "Cov",
        "deepwalk" ~ "DeepWalk",
        "line" ~ "LINE",
        .default = .x
      )),
      var_level_1 = if_else(by == "is_dine", "Yes", var_level_1),
      var_level_2 = if_else(by == "is_dine", "No", var_level_2)
    ) |>
    filter(!(by == "dimension" &
               var_level_1 %in% c(8, 16, 24) & var_level_2 %in% c(32, 64, 128) |
               var_level_2 %in% c(8, 16, 24) & var_level_1 %in% c(32, 64, 128)
    )) |>
    select(c(var_level_1, var_level_2, estimate, conf.low, conf.high))

  return(df_contr_simple_formatted)
}

df_contr_simple_formatted_populist <- calc_simple_contrasts(model_populist)

footnote <- "Positive difference values indicate that the marginal predictions for level 1 are higher than those for level 2. Differences for which the 95\\% quantile credible interval (CI) excludes zero are highlighted in bold. KNN = k-nearest neighbor. LinReg = linear regression. XGB = XGBoost. Embed = embeddings. Cov = covariates."

headers <- list(
  "Estimator" = 1,
  "Feature set" = 4,
  "Year" = 7,
  "DINE-transformed" = 10,
  "DeepWalk walk length" = 11,
  "DeepWalk walks per node" = 14,
  "DeepWalk window size" = 15,
  "Method" = 18,
  "Dimension" = 19
)

contr_table <- tt(
  df_contr_simple_formatted_populist,
  caption = "Contrasts for Populist Voting Prediction Peformance Scores",
  notes = paste("Differences are in macro AUC score.", footnote)
) |>
  setNames(c("Level 1", "Level 2", "Difference", "Lower", "Upper")) |>
  style_tt(
    i = c(2, 3, 4, 6, 7, 8, 16, 17, 18, 20, 26, 29, 30, 31, 32, 33),
    j = 3,
    bold = TRUE
  ) |>
  group_tt(
    i = headers,
    j = list(
      "Contrast" = 1:2,
      "95\\% CI" = 4:5
    )
  ) |>
  style_tt(j = 3:5, align = "c") |>
  format_tt(replace = "-", digits = 2, num_fmt = "significant_cell")

clipr::write_clip(print(contr_table, output = "latex"))

# Cross-contrasts

# contr_vars_cross <- list(c("estimator", "feature_set"), c("walk_length", "num_walks"))
#
# df_contr_cross <- Reduce(rbind, lapply(contr_vars_cross, function(v) {
#   return(
#     model |>
#       avg_predictions(
#         by = v,
#         hypothesis = difference ~ pairwise
#       ) |>
#       mutate(by_1 = v[1], by_2 = v[2])
#   )
# }))
#
# df_contr_cross |>
#   as.data.frame() |>
#   mutate(var = str_split(hypothesis, " - ", n = 2)) |>
#   unnest_wider(var, names_sep = "_level_") |>
#   mutate(across(c(var_level_1, var_level_2), ~ str_extract(.x, "([a-z,_, ]+)|([0-9, ]+)"))) |>
#   mutate(across(c(var_level_1, var_level_2), ~ str_split(.x, " "))) |>
#   unnest_wider(c(var_level_1, var_level_2), names_sep = "_cross_") |>
#   filter(!if_any(starts_with("var_level"), ~ .x == ""),
#          var_level_1_cross_1 != var_level_2_cross_1,
#          var_level_1_cross_2 != var_level_2_cross_2) |>
#   mutate(
#     across(starts_with("var_level"), ~ case_match(
#       .x,
#       "xgb" ~ "XGB",
#       "knn" ~ "KNN",
#       "lm" ~ "LinReg",
#       "embeddings_covariates" ~ "Embed + Cov",
#       "embeddings" ~ "Embed",
#       "covariates" ~ "Cov",
#       .default = .x
#     ))
#   ) |>
#   select(c(var_level_1_cross_1, var_level_2_cross_1, var_level_1_cross_2, var_level_2_cross_2, estimate, conf.low, conf.high))


# Prediction scores trust in government -------------------------------------------------------

# Create model formula for trust in government
model_formula_trust <- create_model_formula("trust", zero_inflated_beta(), estimate_missing = FALSE)

# Fit model
model_trust <- brm(
  model_formula_trust,
  data = df_combined,
  family = zero_inflated_beta(),
  prior = model_prior,
  init = init,
  chains = num_chains,
  iter = num_samples,
  cores = 4,
  seed = seed,
  backend = "cmdstanr",
  stan_model_args = list(stanc_options = list("O1")),
  save_model = file.path("data", "models", "model_trust.stan"),
  file =  file.path("data", "models", "model_trust"),
  file_refit = "on_change"
)

summary(model_trust)

# Create figure for posterior predictive distribution
plot_grid(
  pp_check(model_trust, prefix = "ppc", ndraws = 100) +
    labs(x = "R-squared score", y = "Density"),
  pp_check(model_trust, type = "ecdf_overlay", ndraws = 100) +
    labs(x = "R-squared score", y = "Cumulative density"),
  ncol = 1,
  labels = c("A", "B")
)

ggsave(file.path("figures", "posterior_predictive_check_trust.png"), width = 10, height = 6)

# Get posterior predictive draws
df_pred_trust <- model_trust |>
  get_posterior_predictive_draws(df_combined |> filter(!is.na(trust)))

# Plot posterior means across feature sets
create_posterior_predictive_plot(df_pred_trust, trust) +
  ylim(c(0, 0.2)) +
  ylab("R-squared score")

ggsave(file.path("figures", "prediction_performance_trust.png"), width = 6, height = 4)

# Calc contrasts between feature sets
contr_feature_set_trust <- model_trust |>
  avg_predictions(by = "feature_set", hypothesis = difference ~ pairwise)

print(contr_feature_set_trust, digits = 2)

# Calc contrasts for best subset
contr_feature_set_sub_trust <- model_trust |>
  avg_predictions(
    by = "feature_set",
    hypothesis = difference ~ pairwise,
    newdata = subset((estimator == "xgb" & walk_length == 20 & num_walks == 100 & dimension == 32) |
                       (estimator == "xgb" & feature_set == "covariates"))
  )

print(contr_feature_set_sub_trust, digits = 2)

# Calc all contrasts
df_contr_simple_formatted_trust <- calc_simple_contrasts(model_trust)

contr_table <- tt(
  df_contr_simple_formatted_trust,
  caption = "Contrasts for Trust in the Government Prediction Peformance Scores",
  notes = paste("Differences are in R-squared score.", footnote)
) |>
  setNames(c("Level 1", "Level 2", "Difference", "Lower", "Upper")) |>
  style_tt(
    i = c(2, 3, 4, 6, 7, 8, 10, 11, 12, 16, 17, 18, 20, 22, 28, 29, 30, 31, 32, 33),
    j = 3,
    bold = TRUE
  ) |>
  group_tt(
    i = headers,
    j = list(
      "Contrast" = 1:2,
      "95\\% CI" = 4:5
    )
  ) |>
  style_tt(j = 3:5, align = "c") |>
  format_tt(replace = "-", digits = 2, num_fmt = "significant_cell")

clipr::write_clip(print(contr_table, output = "latex"))


# Prediction scores general voting ------------------------------------------------------------

# Create model formula for trust in government
model_formula_voting <- create_model_formula("voting", Beta())

# Fit model
model_voting <- brm(
  model_formula_voting,
  data = df_combined,
  family = Beta(),
  prior = model_prior,
  init = init,
  chains = num_chains,
  iter = num_samples,
  cores = 4,
  seed = seed,
  backend = "cmdstanr",
  stan_model_args = list(stanc_options = list("O1")),
  save_model = file.path("data", "models", "model_voting.stan"),
  file =  file.path("data", "models", "model_voting"),
  file_refit = "on_change"
)

summary(model_voting)

# Create figure for posterior predictive distribution
plot_grid(
  pp_check(model_voting, prefix = "ppc", ndraws = 100) +
    labs(x = "Weighted AUC score", y = "Density"),
  pp_check(model_voting, type = "ecdf_overlay", ndraws = 100) +
    labs(x = "Weighted AUC score", y = "Cumulative density"),
  ncol = 1,
  labels = c("A", "B")
)

ggsave(file.path("figures", "posterior_predictive_check_general_voting.png"), width = 10, height = 6)

# Get posterior predictive draws
df_pred_voting <- model_voting |>
  get_posterior_predictive_draws(df_combined)

# Plot posterior means across feature sets
create_posterior_predictive_plot(df_pred_voting, voting) +
  ylab("Weighted AUC score")

ggsave(file.path("figures", "prediction_performance_general_voting.png"), width = 6, height = 4)

# Calc contrasts between feature sets
contr_feature_set_voting <- model_voting |>
  avg_predictions(by = "feature_set", hypothesis = difference ~ pairwise)

print(contr_feature_set_voting, digits = 2)

# Calc contrasts for best subset
contr_feature_set_sub_voting <- model_voting |>
  avg_predictions(
    by = "feature_set",
    hypothesis = difference ~ pairwise,
    newdata = subset((estimator == "xgb" & walk_length == 20 & num_walks == 100 & dimension == 32) |
                       (estimator == "xgb" & feature_set == "covariates"))
  )

print(contr_feature_set_sub_voting, digits = 2)

# Calc all contrasts
df_contr_simple_formatted_voting <- calc_simple_contrasts(model_voting)

contr_table <- tt(
  df_contr_simple_formatted_voting,
  caption = "Contrasts for General Voting Prediction Peformance Scores",
  notes = paste("Differences are in weighted AUC score.", footnote)
) |>
  setNames(c("Level 1", "Level 2", "Difference", "Lower", "Upper")) |>
  style_tt(
    i = c(2, 3, 4, 6, 8, 10, 12, 14, 16, 17, 18, 20, 22, 23, 24, 26, 28, 29, 30, 31, 32, 33),
    j = 3,
    bold = TRUE
  ) |>
  group_tt(
    i = headers,
    j = list(
      "Contrast" = 1:2,
      "95\\% CI" = 4:5
    )
  ) |>
  style_tt(j = 3:5, align = "c") |>
  format_tt(replace = "-", digits = 2, num_fmt = "significant_cell")

clipr::write_clip(print(contr_table, output = "latex"))
