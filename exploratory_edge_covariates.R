
library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(forcats)
library(fs)
library(cowplot)

export_dir <- path("data", "export")

df_edge_covariates <- read.csv(path(export_dir, "edge_covariates.csv")) |>
    rowwise() |>
    mutate(relation_type = factor(relation_type)) |>
    mutate(
        dim = str_sub(sapply(str_extract_all(filename, "dim\\d+"), function(x) x[2]), 4, 5),
        relation_group = case_match(
            relation_type %/% 100,
            1 ~ "Neighbor",
            2 ~ "Colleague",
            3 ~ "Family",
            4 ~ "Household",
            5 ~ "Classmate",
            .default = NA
        ),
        relation_type_str = case_match(
            relation_type,
            "101" ~ "10 closest",
            "102" ~ "200 meters",
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
            "320" ~ "Marriage nephew/niece (full)",
            "321" ~ "Marriage nephew/niece",
            "322" ~ "Marriage uncle/aunt",
            "401" ~ "Non-institutional",
            "402" ~ "Insitutional",
            "501" ~ "Primary",
            "502" ~ "Special",
            "503" ~ "Secondary",
            "504" ~ "Vocational",
            "505" ~ "Higher vocational",
            "506" ~ "University",
            .default = NA
        )
    ) |>
    filter(str_detect(filename, "2021") & dim %in% c("17", "27")) |>
    pivot_longer(cols = starts_with("decile"), names_to = "decile", values_to = "decile_utility")


convert_relation_labels <- function(x) {
    label <- case_match(
        x,
        "101" ~ "10 closest",
        "102" ~ "200 meters",
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
        "320" ~ "M. nephew/niece (full)",
        "321" ~ "M. nephew/niece",
        "322" ~ "M. uncle/aunt",
        "401" ~ "Non-institutional",
        "402" ~ "Insitutional",
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


create_edge_covariate_plot <- function(df) {
    p <- df |>
        ggplot() +
        facet_wrap(vars(relation_group), scales = "free_y") +
        geom_point(
            aes(x = decile_utility, y = relation_type, color = dim, alpha = decile),
            position = position_dodge(0.5)
        ) +
        geom_pointrange(
            aes(x = mean_utility, y = relation_type, xmax = mean_utility + sd_utility, xmin = mean_utility - sd_utility, color = dim),
            position = position_dodge(0.5),
            size = 0.6,
            linewidth = 1
        ) +
        geom_vline(xintercept = 0) +
        scale_y_discrete(labels = convert_relation_labels) +
        xlim(c(-0.01, 0.025)) +
        scale_color_viridis_d(
            labels = c("17 (- populist voting / + trust government)", "27 (- trust government)"),
            begin = 0.025,
            end = 0.975
        ) +
        scale_alpha_discrete(guide = NULL) +
        labs(
            x = "Edge utility",
            y = "Network relation type",
            color = "DINE-transformed embedding dimension"
        ) +
        theme_half_open() +
        theme(
            axis.line.y = element_blank(),
            axis.ticks.y = element_blank(),
            legend.justification = "center",
            legend.direction = "horizontal",
            legend.title.position = "top",
            legend.title = element_text(hjust = 0.5)
        ) +
        panel_border() +
        background_grid()

    return(p)
}

p1 <- df_edge_covariates |>
    filter(relation_group != "Family") |>
    create_edge_covariate_plot()

p2 <- df_edge_covariates |>
    filter(relation_group == "Family") |>
    create_edge_covariate_plot()

plot_grid(
    get_legend(p1),
    plot_grid(
        p1 + theme(legend.position = "none"),
        p2 + theme(legend.position = "none", axis.title.y = element_blank()),
        ncol = 2,
        align = "hv",
        axis = "tblr",
        rel_widths = c(1, 0.57)
    ),
    nrow = 2,
    rel_heights = c(0.1, 1)
)

ggsave(path("figures", "edge_covariates.png"), width = 12, height = 7)
