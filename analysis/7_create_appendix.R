# Create Appendix -----------------------------------------------------------------------------

library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(forcats)
library(fs)
library(cowplot)
library(tinytable)


# LISS data summary stats ---------------------------------------------------------------------

get_variable_labels <- function(variable) {
    return(case_match(
        variable,
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
        .default = variable
    ))
}

export_dir <- path("data", "export")

df_numeric <- read.csv(path(export_dir, "liss_preprocessed_numeric_summary.csv"))

summary_table_numeric <- tt(
    df_numeric |>
        select(-n) |>
        mutate(variable = get_variable_labels(variable)),
    caption = "Summary Statistics for Numeric Variables in Linked Dataset",
    notes = "Total sample size was 6063. NA = value not published for privacy reasons."
) |>
    setNames(c("Variable", "Mean", "SD", "Missing")) |>
    format_tt(
        digits = 2
    )

clipr::write_clip(print(summary_table_numeric, output = "latex"))


get_variable_labels_categorical <- function(variable) {
    return(case_match(
        variable,
        "cbs_num_parents_abroad_3_cat" ~ "Parents born abroad",
        "cbs_gender_2_cat" ~ "Gender",
        "cv24p307_voting_behavior" ~ "General voting",
        "cv24p053_voted" ~ "Voted",
        "cv24p307_voting_behavior_populist" ~ "Right-wing populist voting",
        "cbs_high_achieved_edu_degree_5_cat" ~ "Highest achieved education",
        .default = variable
    ))
}


get_level_labels <- function(level) {
    return(case_match(
        level,
        "" ~ "Missing",
        "gl/pvda" ~ "GL/PvdA",
        "pvv" ~ "PVV",
        "bbb" ~ "BBB",
        "nsc" ~ "NSC",
        "cda" ~ "CDA",
        "sp" ~ "SP",
        "pvdd" ~ "PvdD",
        "cu" ~ "CU",
        "fvd" ~ "FvD",
        "ja21" ~ "JA21",
        "denk" ~ "DENK",
        "sgp" ~ "SGP",
        "both" ~ "Both",
        "master_doctorate" ~ "Master/doctorate",
        .default = tools::toTitleCase(level)
    ) |>
        str_replace("_", " ")
    )
}


df_categorical <- read.csv(path(export_dir, "liss_preprocessed_categorical_summary.csv"))

summary_table_categorical <- tt(
    df_categorical |>
        select(-c(n)) |>
        mutate(
            variable = get_variable_labels_categorical(variable),
            level = get_level_labels(level)
        ),
    caption = "Summary Statistics for Categorical Variables in Linked Dataset",
    notes = "Total sample size was 6063. NA = value not published for privacy reasons."
) |>
    setNames(c("Variable", "Level", "n", "Proportion")) |>
    format_tt(
        digits = 2
    ) |>
    style_tt(
        i = 1,
        j = 1,
        rowspan = 4,
        alignv = "t"
    ) |>
    style_tt(
        i = 5,
        j = 1,
        rowspan = 3,
        alignv = "t"
    ) |>
    style_tt(
        i = 8,
        j = 1,
        rowspan = 20,
        alignv = "t"
    ) |>
    style_tt(
        i = 28,
        j = 1,
        rowspan = 4,
        alignv = "t"
    ) |>
    style_tt(
        i = 32,
        j = 1,
        rowspan = 5,
        alignv = "t"
    ) |>
    style_tt(
        i = 37,
        j = 1,
        rowspan = 6,
        alignv = "t"
    )

clipr::write_clip(print(summary_table_categorical, output = "latex"))


df_correlation <- read.csv(path(export_dir, "liss_preprocessed_correlations.csv"))

df_correlation |>
    mutate(
        variable_1 = get_variable_labels_categorical(get_variable_labels(variable_1)),
        variable_2 = get_variable_labels_categorical(get_variable_labels(variable_2))
    ) |>
    ggplot(aes(x = variable_1, y = variable_2, fill = correlation)) +
    geom_tile() +
    scale_fill_gradient2(limits = c(-0.2, 0.85)) +
    labs(x = "", y = "", fill = "Correlation") +
    theme(
        axis.text.x = element_text(angle = 45, hjust = 1)
    )

ggsave("figures/summary_stats_correlations.png")


# Raw LISS data voting summary stats ----------------------------------------------------------

df_liss_raw <- read.csv(fs::path("data", "raw", "liss_data_import_9780.csv"), na.strings = c("NA", "-9", "-8"))

df_liss_raw |>
    filter(cv24p053 != 3 | is.na(cv24p053)) |>
    transmute(populist_voting = case_when(
                  is.na(cv24p053) ~ "Missing",
                  cv24p053 == 2 ~ "Not voted",
                  cv24p307 %in% c(2, 13, 17, 18) ~ "Populist", # 2 = PVV, 3 = FVD, 17 = JA21, 18 = BBB
                  .default = "Non-populist"
              )) |>
    group_by(populist_voting) |>
    summarise(votes = n()) |>
    mutate(share = votes / sum(votes))

df_liss_raw |>
    group_by(cv24p053) |>
    summarise(count = n()) |>
    mutate(share = count /sum(count))
