# Plot feature importance scores --------------------------------------------------------------

library(dplyr)
library(tidyr)
library(ggplot2)
library(cowplot)
library(stringr)
library(forcats)

export_dir <- file.path("data", "export")

set.seed(2024)

embeddings_filename <- "collapsed_2021_len20_window5_walks100_dim32"
embeddings_filename_sensitivity <- "collapsed_2022_len20_window2_walks100_dim32"

# Create function to rename feature labels
get_predictor_labels <- function(predictor) {
    label <- case_match(
        predictor,
        "cbs_age" ~ "Age",
        "cbs_change_purch_power" ~ "Change purch. power",
        "cbs_gender_2_cat_female" ~ "Female",
        "cbs_gender_2_cat_male" ~ "Male",
        "cbs_gross_income_percentile" ~ "Income percentile",
        "cbs_high_achieved_edu_degree_5_cat_bachelor" ~ "Bachelor's degree",
        "cbs_high_achieved_edu_degree_5_cat_higher_vocational" ~ "Vocational degree",
        "cbs_high_achieved_edu_degree_5_cat_master_doctorate" ~ ">= Master's degree",
        "cbs_high_achieved_edu_degree_5_cat_None" ~ "Degree unknown",
        "cbs_high_achieved_edu_degree_5_cat_primary" ~ "Primary education",
        "cbs_high_achieved_edu_degree_5_cat_vocational" ~ "Secondary education",
        "cbs_num_parents_abroad_3_cat_both" ~ "Both parents abroad",
        "cbs_num_parents_abroad_3_cat_one" ~ "One parent abroad",
        "cbs_num_parents_abroad_3_cat_zero" ~ "No parents abroad",
        "cbs_purch_power_log" ~ "Purch power (log)",
        "cp24p010_happiness" ~ "Happiness",
        "cp24p012_feel_state" ~ "Mood state",
        "cp24p013_feel_trait" ~ "Mood trait",
        "cp24p014_satisfaction" ~ "Satisfaction 1",
        "cp24p015_satisfaction" ~ "Satisfaction 2",
        "cp24p016_satisfaction" ~ "Satisfaction 3",
        "cp24p017_satisfaction" ~ "Satisfaction 4",
        "cp24p018_satisfaction" ~ "Satisfaction 5",
        "cp24p019_interpersonal_trust" ~ "Interpersonal trust",
        "cp24p020+_extraversion" ~ "Extraversion",
        "cp24p021+_agreeableness" ~ "Agreeableness",
        "cp24p022+_conscientiousness" ~ "Conscientiousness",
        "cp24p023+_emotional_stability" ~ "Emotional stability",
        "cp24p024+_imagination" ~ "Openness",
        "cp24p070+_self_esteem" ~ "Self esteem",
        "cp24p135_closeness" ~ "Closeness",
        "cp24p136+_social_desirability" ~ "Social desirability",
        "cp24p198+_optimism" ~ "Optimism",
        "cv24p013_trust_government" ~ "Trust government",
        .default = predictor
    )

    label <- ifelse(str_detect(label, "dim_"), str_c("Dimension ", str_extract(label, "\\d+")), label)

    return(label)
}

# Create function to prepare feature importance scores for plotting
prepare_populist_importance <- function(df, target_level = "populist") {
    return(
        df |>
            rowwise() |>
            # In each row, only one outcome level has a value; we retrieve it with max(na.rm = T)
            mutate(
                mean_importance = max(c_across(starts_with("mean_importance")), na.rm = TRUE)
            ) |>
            # Determine the outcome level by matching it with the mean importance score
            mutate(
                outcome_level = case_when(
                    mean_importance_missing == mean_importance ~ "missing",
                    mean_importance_non_populist == mean_importance ~ "non_populist",
                    mean_importance_populist == mean_importance ~ "populist",
                    mean_importance_not_voted == mean_importance ~ "not_voted"
                ),
                is_dine = ifelse(str_detect(filename, "dine"), "DINE-transformed", "Untransformed"),
                model = case_match(
                    model,
                    "covariates" ~ "Covariates",
                    "embeddings" ~ "Embeddings",
                    "embeddings_covariates" ~ "Embeddings + covariates"
                )
            ) |>
            select(c(predictor, decile, outcome_level, mean_predictor,
                     mean_importance, model, is_dine)) |>
            # Only get importance scores for target outcome level
            filter(outcome_level == target_level) |>
            group_by(predictor, model, is_dine) |>
            # Compute the mean absolute score for each feature
            mutate(sum_importance = mean(abs(mean_importance))) |>
            group_by(model, is_dine) |>
            # Rank the scores for each model
            mutate(rank = dense_rank(sum_importance)) |>
            arrange(desc(rank)) |>
            # Relabel features
            mutate(
                predictor = get_predictor_labels(predictor)
            )
    )
}

# Read and prepare feature importance scores
df_feature_importance_populist <- read.csv(
    file.path(export_dir, "feature_importance_cv24p307_voting_behavior_populist.csv")
) |>
    filter(str_detect(filename, embeddings_filename)) |>
    prepare_populist_importance()

# Create function to plot feature importance scores
create_feature_importance_plot <- function(df) {
    p <- df |>
        # Create character strings for mean absolute importance
        mutate(label = ifelse(sum_importance < 0.01, "<0.01", as.character(round(sum_importance, 3)))) |>
        ggplot(aes(x = mean_importance, y = fct_reorder(predictor, rank))) +
        facet_wrap(vars(model, is_dine), scales = "free_y") +
        geom_jitter(aes(color = mean_predictor), width = 0, height = 0.2, size = 2.0) +
        geom_label(aes(label = str_pad(label, width = max(str_width(label))), x = 0.4), family = "mono", label.size = NA) +
        scale_x_continuous(limits = c(-0.3, 0.45)) +
        scale_color_gradient2(mid = "grey", midpoint = 0.5, limits = c(0, 1), breaks = c(0, 0.5, 1)) +
        labs(x = "SHAP value", y = NULL, color = "Normalized\npredictor\nvalue") +
        guides(color = guide_colorbar(title.position = "top", title.hjust = 0.5)) +
        theme_half_open() +
        theme(
            axis.line.y = element_blank(),
            legend.justification = "center",
            legend.direction = "horizontal"
        ) +
        panel_border()

    return(p)
}

# Create plots
p1 <- create_feature_importance_plot(
    df_feature_importance_populist |>
        filter(model == "Covariates", is_dine == "Untransformed") |>
        # We filter feature ranks after filtering models
        filter(rank > max(rank, na.rm = TRUE) - 10)
) +
    facet_wrap(vars(model), scales = "free_y")

p2 <- create_feature_importance_plot(
    df_feature_importance_populist |>
        filter(model == "Embeddings", is_dine == "Untransformed") |>
        filter(rank > max(rank, na.rm = TRUE) - 10)
)

p3 <- create_feature_importance_plot(
    df_feature_importance_populist |>
        filter(model == "Embeddings", is_dine == "DINE-transformed") |>
        filter(rank > max(rank, na.rm = TRUE) - 10)
)

p4 <- create_feature_importance_plot(
    df_feature_importance_populist |>
        filter(model == "Embeddings + covariates", is_dine == "Untransformed") |>
        filter(rank > max(rank, na.rm = TRUE) - 10)
)

p5 <- create_feature_importance_plot(
    df_feature_importance_populist |>
        filter(model == "Embeddings + covariates", is_dine == "DINE-transformed") |>
        filter(rank > max(rank, na.rm = TRUE) - 10)
)

plot_grid(
    plot_grid(
        NULL,
        p1 + theme(legend.position = "none"),
        get_legend(p1),
        ncol = 3,
        rel_widths = c(0.5, 1, 0.5),
        labels = "A",
        label_size = 26
    ),
    plot_grid(
        p2 + theme(legend.position = "none"),
        p3 + theme(legend.position = "none"),
        p4 + theme(legend.position = "none"),
        p5 + theme(legend.position = "none"),
        nrow = 2,
        align = "vh",
        axis = "tblr",
        labels = c("B", "", "C", ""),
        label_size = 26
    ),
    rel_heights = c(0.5, 1),
    nrow = 2
)

ggsave(file.path("figures", "feature_importance_populist.png"), width = 8, height = 10)


# Sensitivity analysis ------------------------------------------------------------------------

df_feature_importance_populist_sensitivity <- read.csv(
    file.path(export_dir, "feature_importance_cv24p307_voting_behavior_populist.csv")
) |>
    filter(str_detect(filename, embeddings_filename_sensitivity)) |>
    prepare_populist_importance()

p1_s <- create_feature_importance_plot(
    df_feature_importance_populist_sensitivity |>
        filter(model == "Embeddings", is_dine == "Untransformed") |>
        filter(rank > max(rank, na.rm = TRUE) - 10)
)

p2_s <- create_feature_importance_plot(
    df_feature_importance_populist_sensitivity |>
        filter(model == "Embeddings", is_dine == "DINE-transformed") |>
        filter(rank > max(rank, na.rm = TRUE) - 10)
)

p3_s <- create_feature_importance_plot(
    df_feature_importance_populist_sensitivity |>
        filter(model == "Embeddings + covariates", is_dine == "Untransformed") |>
        filter(rank > max(rank, na.rm = TRUE) - 10)
)

p4_s <- create_feature_importance_plot(
    df_feature_importance_populist_sensitivity |>
        filter(model == "Embeddings + covariates", is_dine == "DINE-transformed") |>
        filter(rank > max(rank, na.rm = TRUE) - 10)
)

plot_grid(
    get_legend(p1),
    plot_grid(
        p1_s + theme(legend.position = "none"),
        p2_s + theme(legend.position = "none"),
        p3_s + theme(legend.position = "none"),
        p4_s + theme(legend.position = "none"),
        nrow = 2,
        align = "vh",
        axis = "tblr"
    ),
    nrow = 2,
    rel_heights = c(0.2, 1)
)

ggsave(file.path("figures", "feature_importance_populist_sensitivity.png"), width = 10, height = 10.5)
