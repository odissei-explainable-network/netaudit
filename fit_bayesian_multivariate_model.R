library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(gghalves)
library(ggdist)
library(brms)
library(tidybayes)
library(bayesplot)
library(cowplot)
library(marginaleffects)
library(fs)
library(tinytable)

options(tinytable_html_mathjax = TRUE)

export_dir <- path("data", "export")

set.seed(2024)

df_trust_government_scores = read.csv(path(export_dir, "prediction_performance_cv24p013_trust_government_scores.csv"))
df_voting_scores = read.csv(path(export_dir, "prediction_performance_cv24p307_voting_behavior_scores.csv"))
df_voting_populist_scores = read.csv(path(export_dir, "prediction_performance_cv24p307_voting_behavior_populist_scores.csv"))

extract_year <- function(df) {
    return(df |>
               mutate(year = regmatches(
                   df$filename,
                   regexpr("(\\d{4})", df$filename))
               ))
}

keep_cols <- c("filename", "target", "estimator", "model", "year", "walk_length", "window_size", "num_walks", "dimension", "is_dine", "method")

df_combined <- rbind(
    df_trust_government_scores |>
        select(all_of(keep_cols), "R2_test") |>
        rename(score = R2_test),
    df_voting_scores |>
        select(all_of(keep_cols), "AUC_weighted_test") |>
        rename(score = AUC_weighted_test),
    df_voting_populist_scores |>
        select(all_of(keep_cols), "AUC_macro_test") |>
        rename(score = AUC_macro_test)
) |>
    extract_year() |>
    filter(filename != "collapsed_2022_len10_window8_walks10_dim64") |>
    mutate(across(c(walk_length, window_size, num_walks, dimension, is_dine, method, year), ~ ifelse(model == "covariates", NA, .x))) |>
    group_by(target, estimator, method) |>
    mutate(index = row_number()) |>
    filter((model == "covariates" & index %in% 100:104) | model != "covariates") |>
    ungroup() |>
    mutate(score = ifelse(score <= 0, 0, score),
           is_embedding = ifelse(model != "covariates", 1, 0),
           is_deepwalk = if_else(method == "deepwalk", 1, 0, missing = 0),
           method = addNA(factor(method, levels = c("deepwalk", "line"))),
           estimator = factor(estimator, levels = c("lm", "knn", "xgb")),
           model = factor(model, levels = c("embeddings", "covariates", "embeddings_covariates")),
           year = addNA(as.factor(year)),
           walk_length = addNA(factor(walk_length, levels = c("5", "10", "20"))),
           window_size = addNA(factor(window_size, levels = c("2", "5", "8"))),
           num_walks = addNA(factor(num_walks, levels = c("10", "100"))),
           dimension = addNA(factor(dimension, levels = c("8", "16", "24", "32", "64", "128"))),
           is_dine = addNA(ifelse(is_dine, 1, 0))) |>
    select(-c(filename)) |>
    filter(estimator != "dummy" & model != "embeddings_cbs") |>
    group_by(target, estimator, model, year, walk_length, window_size, num_walks, dimension, is_dine) |>
    mutate(index = row_number()) |>
    pivot_wider(id_cols = c(estimator, model, year, walk_length, window_size, num_walks, dimension, is_embedding, is_dine, is_deepwalk, method, index), names_from = target, values_from = score) |>
    rename(trust = cv24p013_trust_government,
           voting = cv24p307_voting_behavior,
           populist = cv24p307_voting_behavior_populist)

formula_mv <- bf(
    mvbind(trust, populist) ~ estimator * model + year + is_dine +
        num_walks * walk_length + window_size + dimension,
    phi ~ model,
    zi ~ model,
    family = zero_inflated_beta()
)

prior_mv <- c(
    set_prior("normal(0, 2.5)", class = "b"),
    set_prior("normal(0, 2.5)", class = "Intercept")
)

for (target in c("trust", "populist")) {
    prior_mv <- c(
        prior_mv,
        # set_prior("exponential(1)", class = "phi", lb = 0, resp = target),
        set_prior("constant(0)", coef = "num_walks100:walk_lengthNA", resp = target),
        set_prior("constant(0)", coef = "num_walksNA:walk_lengthNA", resp = target),
        set_prior("constant(0)", coef = "num_walksNA:walk_length10", resp = target),
        set_prior("constant(0)", coef = "num_walksNA:walk_length20", resp = target)
    )

    for (predictor in c("year", "is_dine", "num_walks", "walk_length", "window_size", "dimension")) {
        prior_mv <- c(prior_mv, set_prior("constant(0)", coef = paste0(predictor, "NA"), resp = target))
    }
}

fit_mv <- brm(
    formula_mv + set_rescor(FALSE),
    data = df_combined,
    family = zero_inflated_beta(),
    prior = prior_mv,
    # sample_prior = "only",
    init = 0,
    chains = 4,
    iter = 500,
    cores = 4,
    seed = 2024,
    backend = "cmdstanr",
    stan_model_args = list(stanc_options = list("O1")),
    save_model = "multivariate_model.stan",
    file = "multivariate_model_zero_inflated",
    file_refit = "on_change"
)

summary(fit_mv)

# mcmc_plot(fit_mv, type = "trace")

# pp_check(fit_mv, prefix = "ppd", resp = "populist")

pp_check(fit_mv, prefix = "ppc", ndraws = 100, resp = "populist")
pp_check(fit_mv, prefix = "ppc", ndraws = 100, resp = "trust")

pp_check(fit_mv, type = "ecdf_overlay", ndraws = 100, resp = "trust")

pp_check(fit_mv, type = "error_hist", ndraws = 100, resp = "trust")

pp_check(fit_mv, type = "xyz", ndraws = 100)

# Plot posterior predictions ------------------------------------------------------------------

get_posterior_predictive_draws <- function(model, resp) {
    return(
        model |>
            posterior_predict(resp = resp, ndraws = 1000) |>
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

df_pred_populist <- fit_mv |>
    get_posterior_predictive_draws(resp = "populist")


plot_posterior_predictive_means <- function(df_pred, df_data, resp, x_var, c_var) {
    resp <- enquo(resp)
    x_var <- enquo(x_var)
    c_var <- enquo(c_var)

    p <- df_pred |>
        filter(.pred > 0) |>
        ggplot() +
        geom_half_boxplot(
            data = df_data,
            aes(x = !! x_var, y = !! resp, fill = !! c_var),
            inherit.aes = FALSE,
            side = "l",
            position = position_dodge(width = 1),
            alpha = 0.25
        ) +
        stat_slabinterval(
            aes(y = .pred, x = !! x_var, fill = !! c_var, fill_ramp = after_stat(level)),
            # normalize = "xy",
            point_interval = "median_qi",
            .width = c(0.5, 0.8, 0.95),
            scale = 0.5, # 2 / length(unique(df_pred |> pull(!! c_var))),
            position = position_dodge(width = 1)
        ) +
        scale_y_continuous(breaks = scales::extended_breaks(7)) +
        scale_color_viridis_d(begin = 0, end = 1, na.value = "black") +
        scale_fill_viridis_d(begin = 0, end = 1.0, na.value = "black") +
        scale_fill_ramp_discrete(na.translate = FALSE, guide = NULL) +
        theme_half_open() +
        theme(
            axis.line.x = element_blank(),
            axis.ticks.x = element_blank(),
            legend.justification = "center",
            legend.direction = "horizontal",
            legend.title.position = "top",
            legend.position = "top"
        )

    if (any(df_pred |> pull(!! resp) <= 0)) {
        p <- p +
            geom_text(aes(x = !! x_var, y = 0.25, group = !! c_var, label = ifelse(prop_zero == 0, "", as.character(prop_zero * 100))), data = df_pred |>
                          group_by(!! x_var, !! c_var) |>
                          summarise(prop_zero = round(mean(!! resp == 0), 2)),
                      position = position_dodge(width = 1), family = "mono", size = 4)

    }

    return(p)
}

populist_y_label <- ylab("Macro AUC score")

relabel_model_levels <- function(df) {
    df |>
        mutate(model = case_match(
            model,
            "embeddings" ~ "Embeddings",
            "covariates" ~ "Covariates",
            "embeddings_covariates" ~ "Embeddings +\ncovariates"
        ))
}

p1 <- plot_posterior_predictive_means(
    relabel_model_levels(df_pred_populist),
    relabel_model_levels(df_combined),
    populist,
    estimator,
    model
) +
    populist_y_label +
    labs(
        x = "Prediction algorithm (estimator)",
        fill = "Feature set",
        color = "Feature set"
    ) +
    scale_x_discrete(labels = c("LinReg", "KNN", "XGB")) +
    ylim(c(0.5, 0.8))

p2 <- plot_posterior_predictive_means(
    relabel_model_levels(df_pred_populist),
    relabel_model_levels(df_combined),
    populist,
    year,
    model
) +
    populist_y_label +
    labs(
        x = "Network year",
        fill = "Feature set",
        color = "Feature set"
    ) +
    ylim(c(0.5, 0.8))


relabel_num_walks_labels <- function(df) {
    df |>
        mutate(num_walks = as.factor(ifelse(num_walks == "<NA>", "Other", num_walks)))
}

p3 <- plot_posterior_predictive_means(
    df_pred_populist,
    df_combined,
    populist,
    walk_length,
    num_walks
) +
    populist_y_label +
    labs(
        x = "DeepWalk walk length",
        fill = "DeepWalk walks per node",
        color = "DeepWalk walks per node"
    ) +
    ylim(c(0.5, 0.8))

relabel_dine_levels <- function(df) {
    df |> mutate(is_dine = as.factor(case_match(
        is_dine,
        "0" ~ "No",
        "1" ~ "Yes"
    )))
}

p4 <- plot_posterior_predictive_means(
    relabel_dine_levels(df_pred_populist),
    relabel_dine_levels(df_combined),
    populist,
    dimension,
    is_dine
) +
    populist_y_label +
    labs(
        x = "Embedding dimensions",
        fill = "DINE-transformed",
        color = "DINE-transformed"
    ) +
    ylim(c(0.5, 0.8))

p_populist <- plot_grid(
    p1,
    p2,
    p3,
    p4,
    ncol = 2,
    align = "vh",
    axis = "tblr"
)


df_pred_trust <- fit_mv |>
    get_posterior_predictive_draws(resp = "trust")

trust_y_label <- ylab("R-squared score")

p5 <- plot_posterior_predictive_means(
    relabel_model_levels(df_pred_trust),
    relabel_model_levels(df_combined),
    trust,
    estimator,
    model
) +
    trust_y_label +
    labs(
        x = "Prediction algorithm (estimator)",
        fill = "Feature set",
        color = "Feature set"
    ) +
    scale_x_discrete(labels = c("LinReg", "KNN", "XGB")) +
    ylim(c(0, 0.3))

p6 <- plot_posterior_predictive_means(
    relabel_model_levels(df_pred_trust),
    relabel_model_levels(df_combined),
    trust,
    year,
    model
) +
    trust_y_label +
    labs(
        x = "Network year",
        fill = "Feature set",
        color = "Feature set"
    ) +
    ylim(c(0, 0.3))

p7 <- plot_posterior_predictive_means(
    df_pred_trust,
    df_combined,
    trust,
    walk_length,
    num_walks
) +
    trust_y_label +
    labs(
        x = "DeepWalk walk length",
        fill = "DeepWalk walks per node",
        color = "DeepWalk walks per node"
    ) +
    ylim(c(0, 0.3))

p8 <- plot_posterior_predictive_means(
    relabel_dine_levels(df_pred_trust),
    relabel_dine_levels(df_combined),
    trust,
    dimension,
    is_dine
) +
    trust_y_label +
    labs(
        x = "Embedding dimensions",
        fill = "DINE-transformed",
        color = "DINE-transformed"
    ) +
    ylim(c(0, 0.3))

p_trust <- plot_grid(
    p5,
    p6,
    p7,
    p8,
    ncol = 2,
    align = "vh",
    axis = "tblr"
)

plot_grid(p_populist, p_trust, nrow = 2, align = "vh", labels = c("A", "B"), label_size = 26)

ggsave(path("figures", "prediction_performance.png"), width = 10, height = 14)



# Contrasts -----------------------------------------------------------------------------------

format_contrasts <- function(df_contrast) {
    df_contrast |>
        mutate(var1 = str_split(contrast, " - "), var2_level1 = NA, var2_level2 = NA) |>
        unnest_wider(c("var1"), names_sep = "_level") |>
        select(var1_level1, var1_level2, var2_level1, var2_level2, estimate, conf.low, conf.high)
}

calc_contrasts <- function(model, resp) {
    contr_estimator <- fit_mv |>
        avg_comparisons(variables = list("estimator" = c("lm", "knn")), resp = resp) |>
        rbind(fit_mv |>
                  avg_comparisons(variables = list("estimator" = c("knn", "xgb")), resp = resp))

    contr_model <- fit_mv |>
        avg_comparisons(variables = list(model = c("covariates", "embeddings")), resp = resp) |>
        rbind(fit_mv |>
                  avg_comparisons(variables = list(model = c("embeddings", "embeddings_covariates")), resp = resp))

    contr_estimator_model <- fit_mv |>
        avg_comparisons(variables = list("estimator" = c("lm", "knn"), model = c("covariates", "embeddings")), cross = TRUE, resp = resp) |>
        rbind(fit_mv |>
                  avg_comparisons(variables = list("estimator" = c("knn", "xgb"), model = c("covariates", "embeddings")), cross = TRUE, resp = resp)) |>
        rbind(fit_mv |>
                  avg_comparisons(variables = list("estimator" = c("lm", "knn"), model = c("embeddings", "embeddings_covariates")), cross = TRUE, resp = resp)) |>
        rbind(fit_mv |>
                  avg_comparisons(variables = list("estimator" = c("knn", "xgb"), model = c("embeddings", "embeddings_covariates")), cross = TRUE, resp = resp))

    contr_year <- fit_mv |>
        avg_comparisons(variables = list("year" = c("2020", "2021")), resp = resp) |>
        rbind(fit_mv |> avg_comparisons(variables = list("year" = c("2021", "2022")), resp = resp))

    contr_is_dine <- fit_mv |>
        avg_comparisons(variables = list("is_dine" = c("0", "1")), resp = resp)

    contr_walk_length <- fit_mv |>
        avg_comparisons(variables = list("walk_length" = c("5", "10")), resp = resp) |>
        rbind(fit_mv |>
                  avg_comparisons(variables = list("walk_length" = c("10", "20")), resp = resp))

    contr_num_walks <- fit_mv |>
        avg_comparisons(variables = list("num_walks" = c("10", "100")), resp = resp)

    contr_walk_length_num_walks <- fit_mv |>
        avg_comparisons(variables = list("walk_length" = c("5", "10"), "num_walks" = c("10", "100")), cross = TRUE, resp = resp) |>
        rbind(fit_mv |>
                  avg_comparisons(variables = list("walk_length" = c("10", "20"), "num_walks" = c("10", "100")), cross = TRUE, resp = resp))

    contr_windw_size <- fit_mv |>
        avg_comparisons(variables = list("window_size" = c("2", "5")), resp = resp) |>
        rbind(fit_mv |> avg_comparisons(variables = list("window_size" = c("5", "8")), resp = resp))

    contr_dimension <- fit_mv |>
        avg_comparisons(variables = list("dimension" = c("8", "16")), resp = resp) |>
        rbind(fit_mv |>
                  avg_comparisons(variables = list("dimension" = c("16", "24")), resp = resp)) |>
        rbind(fit_mv |>
                  avg_comparisons(variables = list("dimension" = c("24", "32")), resp = resp)) |>
        rbind(fit_mv |>
                  avg_comparisons(variables = list("dimension" = c("32", "64")), resp = resp)) |>
        rbind(fit_mv |>
                  avg_comparisons(variables = list("dimension" = c("64", "128")), resp = resp))

    contr_combined <- rbind(
        contr_estimator |> mutate(contrast = case_match(contrast, "xgb - knn" ~ "XGB - KNN", "knn - lm" ~ "KNN - LinReg")) |> format_contrasts(),
        contr_model |> mutate(contrast = case_match(contrast, "embeddings - covariates" ~ "Embed - Cov", "embeddings_covariates - embeddings" ~ "Embed + Cov - Embed")) |> format_contrasts(),
        contr_estimator_model |>
            mutate(
                var1 = str_split(case_match(contrast_estimator, "xgb - knn" ~ "XGB - KNN", "knn - lm" ~ "KNN - LinReg"), " - "),
                var2 = str_split(case_match(contrast_model, "embeddings - covariates" ~ "Embed - Cov", "embeddings_covariates - embeddings" ~ "Embed + Cov - Embed"), " - ")
            ) |>
            unnest_wider(c("var1", "var2"), names_sep = "_level") |>
            select(var1_level1, var1_level2, var2_level1, var2_level2, estimate, conf.low, conf.high),
        contr_year |> format_contrasts(),
        contr_is_dine |> mutate(contrast = "Yes - No") |> format_contrasts(),
        contr_walk_length |> format_contrasts(),
        contr_num_walks |> format_contrasts(),
        contr_walk_length_num_walks |>
            mutate(var1 = str_split(contrast_walk_length, " - "), var2 = str_split(contrast_num_walks, " - ")) |>
            unnest_wider(c("var1", "var2"), names_sep = "_level") |>
            select(var1_level1, var1_level2, var2_level1, var2_level2, estimate, conf.low, conf.high),
        contr_windw_size |> format_contrasts(),
        contr_dimension |> format_contrasts()
    )

    return(contr_combined)
}

contr_populist_combined <- calc_contrasts(fit_mv, "populist")
contr_trust_combined <- calc_contrasts(fit_mv, "trust")

headers <- list(
    "Estimator" = 1,
    "Feature set" = 3,
    # "Estimator $\\times$ feature set" = 5,
    "Year" = 5,
    "DINE-transformed" = 7,
    "DeepWalk walk length" = 8,
    "DeepWalk walks per node" = 10,
    # "DeepWalk walk length $\\times$ walks per node" = 15,
    "DeepWalk window size" = 11,
    "Dimension" = 13
)

options(tinytable_theme_placement_latex_float = "htbp!")

footnote <- "Differences for populist voting are in macro AUC score. Differences for trust in government are in R-squared score. Positive difference values indicate that the marginal predictions for level 1 are higher than those for level 2. Differences for which the 95\\% quantile credible interval (CI) excludes zero are highlighted in bold. KNN = k-nearest neighbor. LinReg = linear regression. XGB = XGBoost. Embed = embeddings. Cov = covariates."

contr_table <- tt(
    contr_populist_combined |>
        rename(diff_populist = estimate, low_populist = conf.low, high_populist = conf.high) |>
        left_join(
            contr_trust_combined |>
                rename(diff_trust = estimate, low_trust = conf.low, high_trust = conf.high)
        ) |>
        filter(is.na(var2_level1)) |>
        select(-c(var2_level1, var2_level2)),
    caption = "Simple contrasts for out-of-sample prediction performance scores for predicting populist voting and trust in the government in the LISS Panel data",
    notes = footnote
) |>
    setNames(c("Level 1", "Level 2", "Difference", "Lower", "Upper", "Difference", "Lower", "Upper")) |>
    style_tt(
        i = c(2, 3, 5, 6, 13, 16, 21, 23, 24, 25, 26),
        j = 3,
        bold = TRUE
    ) |>
    style_tt(
        i = c(2, 3, 5, 6, 8, 13, 14, 16, 21, 23, 24, 25, 26),
        j = 6,
        bold = TRUE
    ) |>
    group_tt(
        i = headers,
        j = list(
            "Contrast" = 1:2,
            "95\\% CI" = 4:5,
            "95\\% CI" = 7:8
        )
    ) |>
    group_tt(
        j = list(
            "Populist voting" = 3:5,
            "Trust government" = 6:8
        )
    ) |>
    style_tt(j = 3:8, align = "c") |>
    format_tt(replace = "-", digits = 2, num_fmt = "significant_cell")

clipr::write_clip(print(contr_table, output = "latex"))

headers_int <- list(
    "Estimator $\\times$ feature set" = 1,
    "DeepWalk walk length $\\times$ walks per node" = 5
)

contr_table_int <- tt(
    contr_populist_combined |>
        rename(diff_populist = estimate, low_populist = conf.low, high_populist = conf.high) |>
        left_join(
            contr_trust_combined |>
                rename(diff_trust = estimate, low_trust = conf.low, high_trust = conf.high)
        ) |>
        filter(!is.na(var2_level1)),
    # width = c(1.2, 1, 1, 1, rep(2/3, 6)),
    caption = "Cross-contrasts for out-of-sample prediction performance scores for predicting populist voting and trust in the government in the LISS Panel data",
    notes = footnote
) |>
    setNames(c("Level 1", "Level 2", "Level 1", "Level 2", "Diff.", "Lower", "Upper", "Diff.", "Lower", "Upper")) |>
    style_tt(
        i = 1:8,
        j = 5,
        bold = TRUE
    ) |>
    style_tt(
        i = 1:8,
        j = 8,
        bold = TRUE
    ) |>
    group_tt(
        i = headers_int,
        j = list(
            "Contrast 1" = 1:2,
            "Contrast 2" = 3:4,
            "95\\% CI" = 6:7,
            "95\\% CI" = 9:11
        )
    ) |>
    group_tt(
        j = list(
            "Populist voting" = 5:7,
            "Trust government" = 8:10
        )
    ) |>
    style_tt(j = 5:10, align = "c") |>
    format_tt(replace = "-", digits = 2, num_fmt = "significant_cell")

clipr::write_clip(print(contr_table_int, output = "latex"))
