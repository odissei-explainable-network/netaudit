# Plot edge utilities for different network relations -----------------------------------------

library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(forcats)
library(cowplot)

export_dir <- file.path("data", "export")

# Collapse multiple relation groups
combined_label <- "Neighbor/Colleague/Household"

# Read relation edge utitlies
df_edge_covariates_full <- read.csv(file.path(export_dir, "edge_covariates.csv")) |>
    rowwise() |>
    mutate(
        # Extract dimension
        dim = str_sub(sapply(str_extract_all(filename, "dim\\d+"), function(x) x[2]), 4, 5),
        # Define relation groups
        relation_group = case_match(
            relation_type %/% 100,
            1 ~ combined_label,
            2 ~ combined_label,
            3 ~ "Family",
            4 ~ combined_label,
            5 ~ "Classmate",
            .default = NA
        )
    ) |>
    mutate(relation_type = factor(relation_type))


df_edge_covariates <- df_edge_covariates_full |>
    # Select target filename
    filter(str_detect(filename, "2021") & dim == "17") |>
    pivot_longer(
        cols = starts_with("decile"),
        names_to = "decile",
        values_to = "decile_utility"
    )

# Define funtion to relabel relation lables
convert_relation_labels <- function(x) {
    label <- case_match(
        x,
        "101" ~ "Neighbor 10 closest",
        "102" ~ "Neighbor 200 meters",
        "201" ~ "Colleague",
        "301" ~ "Parent",
        "302" ~ "Co-parent",
        "303" ~ "Grandparent",
        "304" ~ "Child",
        "305" ~ "Grandchild",
        "306" ~ "Sibling (full)",
        "307" ~ "Sibling (half)",
        "308" ~ "Sibling (unknown)",
        "309" ~ "Nephew/niece (full)",
        "310" ~ "Nephew/niece",
        "311" ~ "Uncle/aunt",
        "312" ~ "Partner (married)",
        "313" ~ "Partner (not married)",
        "314" ~ "In-law parent",
        "315" ~ "In-law child",
        "316" ~ "In-law sibling",
        "317" ~ "Step-parent",
        "318" ~ "Step-child",
        "319" ~ "Step-sibling",
        "320" ~ "By-m. nephew/niece (full)",
        "321" ~ "By-m. nephew/niece",
        "322" ~ "By-m. uncle/aunt",
        "401" ~ "HH non-institutional",
        "402" ~ "HH institutional",
        "501" ~ "Primary",
        "502" ~ "Special",
        "503" ~ "Secondary",
        "504" ~ "Vocational",
        "505" ~ "Higher vocational",
        "506" ~ "University",
        .default = NA
    )
    return(label)
}

# Create function to plot relation edge utilities
create_edge_covariate_plot <- function(df) {
    p <- df |>
        mutate(dim = paste("Dimension", dim)) |>
        ggplot() +
        geom_crossbar(
            aes(
                xmin = mean_utility - 2*sd_utility,
                xmax = mean_utility + 2*sd_utility,
                x = mean_utility,
                y = relation_type
            ),
            alpha = 0.5,
            fill = "grey",
            orientation = "y"
        ) +
        geom_point(aes(x = 0.015, y = relation_type, size = edge_count)) +
        geom_vline(xintercept = 0, linetype = "dashed") +
        scale_y_discrete(labels = convert_relation_labels) +
        xlim(c(-0.01, 0.015)) +
        scale_size_continuous(guide = NULL, limits = c(0, max(df$edge_count)), range = c(0, 4)) +
        labs(
            x = "Edge utility",
            y = "Network relation type",
            color = "DINE-transformed embedding dimension"
        ) +
        theme_half_open() +
        theme(
            axis.line.y = element_blank(),
            axis.title.y = element_blank(),
            axis.text = element_text(size = 10)
        ) +
        panel_border()

    return(p)
}

p1 <- df_edge_covariates |>
    filter(relation_group == combined_label, dim == "17") |>
    create_edge_covariate_plot() +
    facet_wrap(vars(relation_group), scales = "free_y")

p2 <- df_edge_covariates |>
    filter(relation_group == "Classmate", dim == "17") |>
    mutate(relation_type = fct_reorder(relation_type, mean_utility)) |>
    create_edge_covariate_plot() +
    facet_wrap(vars(relation_group), scales = "free_y")

p3 <- df_edge_covariates |>
    filter(relation_group == "Family", dim == "17") |>
    mutate(relation_type = fct_reorder(relation_type, mean_utility)) |>
    create_edge_covariate_plot() +
    facet_wrap(vars(relation_group), scales = "free_y")

plot_grid(
    plot_grid(
        p1 + theme(legend.position = "none", strip.text = element_text(size = 10)),
        p2 + theme(legend.position = "none"),
        ncol = 1,
        align = "hv",
        axis = "tblr",
        labels = c("A", "B"),
        label_size = 26
    ),
    p3 + theme(legend.position = "none"),
    ncol = 2,
    labels = c("", "C"),
    label_size = 26
)

ggsave(file.path("figures", "relation_edge_utilities.png"), width = 8, height = 4.5)

# Sensitivity analysis ------------------------------------------------------------------------

df_edge_covariates_sensitivity <- df_edge_covariates_full |>
    filter(str_detect(filename, "2022") & dim == "20") |>
    pivot_longer(
        cols = starts_with("decile"),
        names_to = "decile",
        values_to = "decile_utility"
    )

x_lim <- xlim(c(-0.015, 0.015))

p1_s <- df_edge_covariates_sensitivity |>
    filter(relation_group == combined_label, dim == "20") |>
    create_edge_covariate_plot() +
    facet_wrap(vars(relation_group), scales = "free_y") +
    x_lim

p2_s <- df_edge_covariates_sensitivity |>
    filter(relation_group == "Classmate", dim == "20") |>
    mutate(relation_type = fct_reorder(relation_type, mean_utility)) |>
    create_edge_covariate_plot() +
    facet_wrap(vars(relation_group), scales = "free_y") +
    x_lim

p3_s <- df_edge_covariates_sensitivity |>
    filter(relation_group == "Family", dim == "20") |>
    mutate(relation_type = fct_reorder(relation_type, mean_utility)) |>
    create_edge_covariate_plot() +
    facet_wrap(vars(relation_group), scales = "free_y") +
    x_lim

plot_grid(
    plot_grid(
        p1_s + theme(legend.position = "none", strip.text = element_text(size = 10)),
        p2_s + theme(legend.position = "none"),
        ncol = 1,
        align = "hv",
        axis = "tblr",
        labels = c("A", "B"),
        label_size = 26
    ),
    p3_s + theme(legend.position = "none"),
    ncol = 2,
    labels = c("", "C"),
    label_size = 26
)

ggsave(file.path("figures", "relation_edge_utilities_sensitivity.png"), width = 10, height = 6)
