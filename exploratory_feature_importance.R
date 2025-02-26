
library(dplyr)
library(tidyr)
library(ggplot2)
library(fs)
library(gghalves)
library(ggh4x)
library(cowplot)
library(stringr)
library(forcats)
library(cetcolor)


export_dir <- path("data", "export")

set.seed(2024)


normalize <- function(x) {
    min_max = range(x, na.rm = TRUE)

    if (min_max[1] == min_max[2]) {
        return(x)
    }
    return((x -  min_max[1]) / (min_max[2] -  min_max[1]))
}

df_feature_importance_populist <- read.csv(
    path(export_dir, "feature_importance_cv24p307_voting_behavior_populist.csv")
) |>
    filter(str_detect(filename, "collapsed_2021_len20_window5_walks100_dim32")) |>
    rowwise() |>
    mutate(
        mean_importance = max(c_across(starts_with("mean_importance")), na.rm = TRUE)
    ) |>
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
    filter(outcome_level %in% c("populist")) |>
    group_by(predictor, model, is_dine) |>
    mutate(sum_importance = abs(sum(mean_importance))) |>
    group_by(model, is_dine) |>
    mutate(rank = dense_rank(sum_importance)) |>
    arrange(desc(rank)) |>
    mutate(
        predictor = case_match(
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
        ), predictor = ifelse(str_detect(predictor, "dim_"), str_c("Dimension ", str_extract(predictor, "\\d+")), predictor))


create_feature_importance_plot <- function(df) {
    p <- df |>
        mutate(label = ifelse(sum_importance < 0.01, "<0.01", as.character(round(sum_importance, 3)))) |>
        ggplot(aes(x = mean_importance, y = fct_reorder(predictor, rank))) +
        facet_wrap(vars(model, is_dine), scales = "free_y") +
        geom_jitter(aes(color = mean_predictor), width = 0, height = 0.2, size = 3.0) +
        geom_label(aes(label = str_pad(label, width = max(str_width(label))), x = 0.4), family = "mono", label.size = NA) +
        scale_x_continuous(limits = c(-0.3, 0.45)) +
        scale_color_viridis_c(limits = c(0, 1), breaks = c(0, 0.5, 1)) +
        labs(x = "SHAP value", y = NULL, color = "Normalized predictor value") +
        guides(color = guide_colorbar(title.position = "top", title.hjust = 0.5)) +
        theme_half_open() +
        theme(
            axis.line.y = element_blank(),
              # legend.title = element_text(size = 10),
            legend.justification = "center",
            legend.direction = "horizontal"
        ) +
        panel_border() +
        background_grid()

    return(p)
}


p1 <- create_feature_importance_plot(
    df_feature_importance_populist |>
        filter(model == "Embeddings", is_dine == "Untransformed") |>
        filter(rank > max(rank, na.rm = TRUE) - 10)
)

p2 <- create_feature_importance_plot(
    df_feature_importance_populist |>
        filter(model == "Embeddings", is_dine == "DINE-transformed") |>
        filter(rank > max(rank, na.rm = TRUE) - 10)
)

p3 <- create_feature_importance_plot(
    df_feature_importance_populist |>
        filter(model == "Embeddings + covariates", is_dine == "Untransformed") |>
        filter(rank > max(rank, na.rm = TRUE) - 10)
)

p4 <- create_feature_importance_plot(
    df_feature_importance_populist |>
        filter(model == "Embeddings + covariates", is_dine == "DINE-transformed") |>
        filter(rank > max(rank, na.rm = TRUE) - 10)
)

p_join_populist <- plot_grid(
    p1 + theme(legend.position = "none"),
    p2 + theme(legend.position = "none"),
    p3 + theme(legend.position = "none"),
    p4 + theme(legend.position = "none"),
    nrow = 2,
    align = "vh",
    axis = "tblr"
)


df_feature_importance_trust <- read.csv(
    path(export_dir, "feature_importance_cv24p013_trust_government.csv")
) |>
    filter(str_detect(filename, "collapsed_2021_len20_window5_walks100_dim32")) |>
    rowwise() |>
    mutate(
        mean_importance = max(c_across(starts_with("mean_importance")), na.rm = TRUE)
    ) |>
    mutate(
        is_dine = ifelse(str_detect(filename, "dine"), "DINE-transformed", "Untransformed"),
        model = case_match(
            model,
            "covariates" ~ "Covariates",
            "embeddings" ~ "Embeddings",
            "embeddings_covariates" ~ "Embeddings + covariates"
        )
    ) |>
    select(c(predictor, decile, mean_predictor,
             mean_importance, model, is_dine)) |>
    group_by(predictor, model, is_dine) |>
    mutate(sum_importance = abs(sum(mean_importance))) |>
    group_by(model, is_dine) |>
    mutate(rank = dense_rank(sum_importance)) |>
    arrange(desc(rank)) |>
    mutate(
        predictor = case_match(
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
        ), predictor = ifelse(str_detect(predictor, "dim_"), str_c("Dimension ", str_extract(predictor, "\\d+")), predictor))


p5 <- create_feature_importance_plot(
    df_feature_importance_trust |>
        filter(model == "Embeddings", is_dine == "Untransformed") |>
        filter(rank > max(rank, na.rm = TRUE) - 10)
)

p6 <- create_feature_importance_plot(
    df_feature_importance_trust |>
        filter(model == "Embeddings", is_dine == "DINE-transformed") |>
        filter(rank > max(rank, na.rm = TRUE) - 10)
)

p7 <- create_feature_importance_plot(
    df_feature_importance_trust |>
        filter(model == "Embeddings + covariates", is_dine == "Untransformed") |>
        filter(rank > max(rank, na.rm = TRUE) - 10)
)

p8 <- create_feature_importance_plot(
    df_feature_importance_trust |>
        filter(model == "Embeddings + covariates", is_dine == "DINE-transformed") |>
        filter(rank > max(rank, na.rm = TRUE) - 10)
)

p_join_trust <- plot_grid(
    p5 + theme(legend.position = "none"),
    p6 + theme(legend.position = "none"),
    p7 + theme(legend.position = "none"),
    p8 + theme(legend.position = "none"),
    ncol = 2,
    align = "vh",
    axis = "tblr"
)

plot_grid(plot_grid(p_join_populist, p_join_trust, nrow = 2, align = "vh", labels = c("A", "B"), label_size = 26), cowplot::get_legend(p5), nrow = 2, rel_heights = c(1, 0.1), rel_widths = c(1, 1))

ggsave(path("figures", "feature_importance.png"), width = 10, height = 14)

