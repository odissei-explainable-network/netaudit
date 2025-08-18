# Plot edge utility strength ------------------------------------------------------------------

library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(forcats)
library(cowplot)
library(tinytable)

export_dir <- file.path("data", "export")

# Define function to filter for best model and dimension 17
filter_filename <- function(df, year, dimension = 17) {
    df |>
        mutate(
            dim = str_sub(sapply(str_extract_all(filename, "dim\\d+"), function(x) x[2]), 4, 5)
        ) |>
        filter(
            dim == dimension & str_detect(filename, year)
        )
}


# Age and income ------------------------------------------------------------------------------

create_table_age_income <- function(df) {
    tab <- df |>
        filter(
            variable_1 %in% c("age", "income_percentile", "utility_strength"),
            variable_2 %in% c("age", "income_percentile", "utility_strength")
        ) |>
        mutate(
            dim = paste("Dimension", dim),
            across(starts_with("variable_"), ~ case_match(
                .x,
                "utility_strength" ~ "Edge utility strength",
                "income_percentile" ~ "Income percentile",
                "age" ~ "Age"
            ))
        ) |>
        filter(as.numeric(factor(variable_1)) < as.numeric(factor(variable_2))) |>
        select(variable_1, variable_2, correlation, n) |>
        rename("Variable 1" = variable_1, "Variable 2" = variable_2, "Pearson r" = correlation, "Sample size" = n) |>
        tt()

    return(tab)
}

filter_filename(read.csv(file.path(export_dir, "node_covariates_age_income.csv")), "2021") |>
    create_table_age_income()


# Education, parents born abroad, gender ------------------------------------------------------

plot_categorical_utility <- function(df, y_var) {
    # Enquosure for tidy evaluation
    y_var <- enquo(y_var)

    p <- df |>
        ggplot() +
        geom_crossbar(
            aes(
                xmin = utility_strength_mean - 2*utility_strength_sd,
                xmax = utility_strength_mean + 2*utility_strength_sd,
                x = utility_strength_mean,
                y = !! y_var
            ),
            fill = "grey",
            orientation = "y"
        ) +
        geom_point(aes(x = 3.0, y = !! y_var, size = n)) +
        geom_vline(xintercept = 0, linetype = "dashed") +
        labs(x = "Edge utility strength", y = "") +
        scale_x_continuous(limits = c(-2.5, 3.25), breaks = seq(-2, 3, 1)) +
        scale_color_gradient2(mid = "grey") +
        scale_size_continuous(guide = "none", limits = c(0, max(df$n)), range = c(0, 4)) +
        theme_half_open() +
        theme(
            axis.line.y = element_blank()
        ) +
        panel_border()

    return(p)
}

plot_categorical_utility_education <- function(df) {
    p <- df |>
        mutate(
            education = fct_relevel(fct_recode(
                as.factor(ifelse(is.na(education), "Unknown", education)),
                "Primary" = "11",
                "Vocational" = "12",
                "Secondary" = "21",
                "Bachelor's" = "31",
                "Master's\nDoctorate" = "32"
            ), "Unknown", after = 0)
        ) |>
        plot_categorical_utility(education) +
        facet_wrap(vars("Education"))
    return(p)
}

plot_categorical_utility_parents_abroad <- function(df) {
    p <- df |>
        mutate(
            parents_abroad = fct_relevel(fct_recode(
                as.factor(parents_abroad),
                "None" = "geen ouders in het buitenland",
                "One" = "1 ouder in het buitenland",
                "Both" = "beide ouders in het buitenland",
            ), "Both", after = 0L)
        ) |>
        plot_categorical_utility(parents_abroad) +
        facet_wrap(vars("Parents born abroad"))

    return(p)
}

plot_categorical_utility_gender <- function(df) {
    p <- df |>
        mutate(
            gender = fct_recode(
                as.factor(gender),
                "Women" = "Vrouwen",
                "Men" = "Mannen"
            )
        ) |>
        plot_categorical_utility(gender) +
        facet_wrap(vars("Gender"))

    return(p)
}

p2 <- filter_filename(read.csv(file.path(export_dir, "node_covariates_education.csv")), "2021") |>
    plot_categorical_utility_education()

p3 <- filter_filename(read.csv(file.path(export_dir, "node_covariates_parents_abroad.csv")), "2021") |>
    plot_categorical_utility_parents_abroad()

p4 <- filter_filename(read.csv(file.path(export_dir, "node_covariates_gender.csv")), "2021") |>
    plot_categorical_utility_gender()

plot_grid(
    p2,
    p3,
    p4,
    ncol = 3,
    align = "hv",
    axis = "tblr",
    labels = c("A", "B", "C"),
    label_size = 26
)

ggsave(file.path("figures", "edge_utility_strength.png"), width = 10, height = 3)


# Sensitivity analysis ------------------------------------------------------------------------

filter_filename(read.csv(file.path(export_dir, "node_covariates_age_income.csv")), "2022", dimension = 20) |>
    create_table_age_income()

p2_s <- filter_filename(read.csv(file.path(export_dir, "node_covariates_education.csv")), "2022", dimension = 20) |>
    plot_categorical_utility_education()

p3_s <- filter_filename(read.csv(file.path(export_dir, "node_covariates_parents_abroad.csv")), "2022", dimension = 20) |>
    plot_categorical_utility_parents_abroad()

p4_s <- filter_filename(read.csv(file.path(export_dir, "node_covariates_gender.csv")), "2022", dimension = 20) |>
    plot_categorical_utility_gender()

plot_grid(
    p2_s,
    p3_s,
    p4_s,
    ncol = 3,
    align = "hv",
    axis = "tblr",
    labels = c("A", "B", "C"),
    label_size = 26
)

ggsave(file.path("figures", "edge_utility_strength_sensitivity.png"), width = 10, height = 3)
