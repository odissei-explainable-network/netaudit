###############################################################
# Preprocess prediction scores for Bayesian regression modeling
###############################################################

library(dplyr)
library(tidyr)
library(stringr)
library(forcats)

# Set dir with prediction scores exported from CBS Microdata environment
export_dir <- "data/export"

set.seed(2024)

# Load prediction scores
df_trust_government_scores <- read.csv(
  file.path(export_dir, "prediction_performance_cv24p013_trust_government_scores.csv")
)
df_voting_scores <- read.csv(
  file.path(export_dir, "prediction_performance_cv24p307_voting_behavior_scores.csv")
)
df_voting_populist_scores <- read.csv(
  file.path(export_dir, "prediction_performance_cv24p307_voting_behavior_populist_scores.csv")
)

# Create function to extract the network year from the filename
extract_year <- function(df) {
  return(df |>
           mutate(year = regmatches(
             df$filename,
             regexpr("(\\d{4})", df$filename)
           )))
}

# Define columns we want to keep for the analysis
keep_cols <- c("filename", "target", "estimator", "model", "year", "walk_length", "window_size",
               "num_walks", "dimension", "is_dine", "method")

# Define grid with all possible variable level combinations
df_grid <- expand.grid(
  target = c("cv24p307_voting_behavior_populist", "cv24p013_trust_government", "cv24p307_voting_behavior"),
  estimator = c("lm", "knn", "xgb"),
  model = c("embeddings", "covariates", "embeddings_covariates"),
  year = c("2020", "2021", "2022"),
  method = "deepwalk",
  dimension = c(32, 64, 128),
  num_walks = c(10, 100),
  walk_length = c(5, 10, 20),
  window_size = c(2, 5, 8),
  is_dine = c("true", "false")
)

df_combined <- rbind(
  df_trust_government_scores |>
    select(all_of(keep_cols), "R2_test") |> # Target variable R2
    rename(score = R2_test),
  df_voting_scores |>
    select(all_of(keep_cols), "AUC_weighted_test") |> # Target variable AUC weighted
    rename(score = AUC_weighted_test),
  df_voting_populist_scores |>
    select(all_of(keep_cols), "AUC_macro_test") |> # Target variable AUC macro
    rename(score = AUC_macro_test)
) |>
  extract_year() |>
  filter(
    filename != "collapsed_2022_len20_window8_walks10_dim64_scores.json", # Scores from this file are erroneous
    estimator != "dummy" & model != "embeddings_cbs" # Remove dummy and embeddings_cbs scores
  ) |>
  full_join(df_grid) |>
  # Set missing values to Inf to keep numeric column type (needed for saving and loading when adding NA factor levels)
  mutate(score = ifelse(is.na(score), Inf, score)) |>
  # Set undefined variables to NA for covariates
  mutate(across(
    c(walk_length, window_size, num_walks, dimension, is_dine, method, year),
    ~ ifelse(model == "covariates", NA, .x)
  )) |>
  ungroup() |>
  mutate(
    score = ifelse(score <= 0, 0, score), # Set scores below 0 to 0 (for trust in government)
    # Convert independent variables to factors and set levels
    estimator = factor(estimator, levels = c("lm", "knn", "xgb")),
    model = factor(model, levels = c("embeddings", "covariates", "embeddings_covariates")),
    year = addNA(as.factor(year)),
    walk_length = addNA(factor(walk_length, levels = c("5", "10", "20"))),
    window_size = addNA(factor(window_size, levels = c("2", "5", "8"))),
    num_walks = addNA(factor(num_walks, levels = c("10", "100"))),
    dimension = addNA(factor(dimension, levels = c("8", "16", "24", "32", "64", "128"))),
    is_dine = addNA(ifelse(is_dine, 1, 0))
  ) |>
  # Transform to wide format
  group_by(
    target, estimator, model, year, walk_length,
    window_size, num_walks, dimension, is_dine
  ) |>
  mutate(index = row_number()) |>
  pivot_wider(
    id_cols = c(estimator, model, year, walk_length, window_size,
                num_walks, dimension, is_dine, index),
    names_from = target,
    values_from = score
  ) |>
  rename(
    trust = cv24p013_trust_government,
    voting = cv24p307_voting_behavior,
    populist = cv24p307_voting_behavior_populist
  ) |>
  select(-index)

write.csv(
  df_combined,
  file = fs::path("data", "preprocessed", "prediction_scores.csv"),
  row.names = FALSE
)

# Get best performing models by mean test score
df_combined |>
  filter(is.finite(populist)) |>
  group_by(estimator, model, year, walk_length, window_size, num_walks, dimension, is_dine) |>
  summarise(
    mean_test_macro_AUC = mean(populist),
    mean_test_r_squared = mean(trust),
    is_dine = unique(is_dine),
    N = n()
  ) |>
  arrange(desc(mean_test_macro_AUC))

# Check how many unique models there are
df_combined |>
  group_by(is_dine, estimator, model) |>
  summarise(unique_embeddings = n() / 5) # 5 CV folds

# Check missing models
nrow(df_combined |> filter(is.infinite(populist))) / 3 / 3 / 2 # Estimator, model, is_dine
