# Plot edge utility at municipality level -----------------------------------------------------

library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(ggraph)
library(tidygraph)
library(igraph)
library(sf)
library(sfnetworks)
library(forcats)
library(corrr)
library(cowplot)
library(janitor)

export_dir <- file.path("data", "export")

raw_dir <- file.path("data", "raw")

populist_party_names <- c(
  "PVV (Partij voor de Vrijheid)",
  "BBB",
  "Forum voor Democratie",
  "JA21"
)

# Calculate populist votes at country level
read.csv(file.path(raw_dir, "uitslag_TK20231122_Nederland.csv"), sep = ";") |>
  clean_names() |>
  filter(type == "Partij") |> # Only use votes for parties
  mutate(
    total_votes = sum(aantal_stemmen),
    share_votes = aantal_stemmen / total_votes,
    is_populist = partij %in% populist_party_names
  ) |>
  filter(is_populist) |> # Only votes for populist parties
  group_by(regio) |> # Calculate percentages for entire country
  summarise(populist_votes = sum(share_votes) * 100)

# Calculate populist votes at municipality level
df_election_results <- read.csv(file.path(raw_dir, "uitslag_TK20231122_Gemeente.csv"), sep = ";") |>
  clean_names() |>
  group_by(gemeente) |>
  mutate(
    total_votes = sum(aantal_stemmen),
    share_votes = aantal_stemmen / total_votes,
    is_populist = partij %in% populist_party_names
  ) |>
  filter(is_populist) |>
  summarise(populist_votes = sum(share_votes) * 100)

preprocess_municipality_statistics <- function(df) {
  return(
    df |>
      clean_names() |>
      mutate(
        # Trim whitespace
        soort_regio_2 = trimws(soort_regio_2),
        gemeentenaam_1 = trimws(gemeentenaam_1)
      ) |>
      # Only select municipalities
      filter(soort_regio_2 == "Gemeente") |>
      # Rename municipality column for joins
      rename(gemeente = gemeentenaam_1)
  )
}

# Read in municipality statistics
df_municipality_2021 <- read.csv(file.path(raw_dir, "85039NED_TypedDataSet_03042025_112748_2021.csv"), sep = ";") |>
  preprocess_municipality_statistics()

# Read in map of Dutch provinces
geo_province <- st_read(
  file.path(raw_dir, "cbsgebiedsindelingen2023.gpkg"),
  layer = "provincie_gegeneraliseerd",
  quiet = TRUE
)

# Read in map of Dutch municipalities
geo_municipality <- st_read(
  file.path(raw_dir, "cbsgebiedsindelingen2023.gpkg"),
  layer = "gemeente_gegeneraliseerd",
  quiet = TRUE
) |>
  clean_names() |>
  rename(gemeente = statnaam) |>
  left_join(df_election_results, by = "gemeente") |>
  st_join(geo_province, join = st_within)

# Join map with municipality statistics
geo_municipality_2021 <- geo_municipality |>
  left_join(df_municipality_2021, by = "gemeente") |>
  rename(province = statnaam) |>
  group_by(province)

# Read in edge utilities
df_edge_utilities <- read.csv(file.path(export_dir, "collapsed_2021_len20_window5_walks100_dim32_dine_utilities_dim17_community_edgelist.csv"))

# Read in municipality IDs
df_node_attributes <- read.csv(file.path(export_dir, "collapsed_2021_municipalities_stats.csv"))

# Define function to create graph
create_graph_from_edgelist <- function(df_edgelist, df_nodelist, df_geo) {
  gr_igraph <- graph_from_data_frame(
    df_edgelist |>
      # Make sure to only use municipalities for which results were exported
      filter(source %in% df_nodelist$id & target %in% df_nodelist$id),
    directed = FALSE,
    vertices = df_nodelist |>
      select(id, everything())
  )

  # Join edge utilities with map
  gr <- as_tbl_graph(gr_igraph) |>
    activate(nodes) |>
    right_join(df_geo |> rename(municipality = gemeente), by = "municipality")

  return(gr)
}

# Create graph
gr <- create_graph_from_edgelist(df_edge_utilities, df_node_attributes, geo_municipality_2021)

# Define function to create hierarchical graph (municipalities nested within provinces)
# We create a hierarchical graph to bundle edges within provinces to prevent cluttering
# Edges are bundled according to centroid of each province
create_hierarchy_graph <- function(gr, df_nodelist, df_geo) {
  # Group municipalities by province
  df_hierarchy <- gr |>
    activate(nodes) |>
    data.frame() |>
    group_by(province) |>
    distinct(municipality) |>
    mutate(country = "Netherlands")

  # Create graph with nested structure: Municipality within province within country
  gr_hierarchy <- graph_from_data_frame(rbind(
    data.frame(from = df_hierarchy$country, to = df_hierarchy$province),
    data.frame(from = df_hierarchy$province, to = df_hierarchy$municipality)
  )) |>
    as_tbl_graph() |>
    activate(nodes) |>
    # Join municipality statistics
    left_join(df_nodelist |> mutate(name = municipality), by = "name") |>
    left_join(df_geo |> mutate(name = gemeente), by = "name") |>
    left_join(geo_province |> mutate(name = statnaam) |> select(name, geom), by = "name") |>
    # Calculate degree centrality for each municipality
    left_join(gr |>
                mutate(
                  name = municipality,
                  degree = centrality_degree(weights = weight)
                ) |>
                activate(nodes) |>
                data.frame() |>
                select(name, degree),
              by = "name") |>
    mutate(
      geom = case_when(
        st_is_empty(geom.x) ~ geom.y,
        .default = geom.x
      ),
      # Compute centroid for each province
      center = case_when(
        name == "Netherlands" ~ geo_province$geom |> st_union() |> st_centroid(),
        .default = st_centroid(geom)
      ),
      is_province = name %in% geo_province$statnaam
    )

  return(gr_hierarchy)
}

gr_hierarchy <- create_hierarchy_graph(gr, df_node_attributes, geo_municipality_2021)


# Plot municipality networks ------------------------------------------------------------------

# Define function to plot hierarchical network with populist votes per municipality
plot_sf_network <- function(gr, gr_hierarchy, fill_var) {
  fill_var <- enquo(fill_var)

  df_edges <- gr |>
    activate(edges) |>
    mutate(
      name_from = .N()$municipality[from],
      name_to = .N()$municipality[to]
    ) |>
    select(name_from, name_to, weight) |>
    data.frame()

  df_nodes <- gr_hierarchy |>
    activate(nodes) |>
    data.frame()

  conn <- get_con(
    from = match(df_edges$name_from, df_nodes$name),
    to = match(df_edges$name_to, df_nodes$name),
    value = df_edges$weight
  )

  p <- gr_hierarchy |>
    select(name, center, geom, is_province, !!fill_var) |>
    mutate(geom_bbox = st_boundary(geom)) |>
    as_sfnetwork() |>
    ggraph(layout = "sf") +
    geom_sf(aes(geometry = geom, fill = !!fill_var), color = "darkgrey") +
    geom_conn_bundle(data = conn, aes(alpha = abs(value), width = abs(value)), tension = 0.9, show.legend = FALSE) +
    scale_edge_width_continuous(range = c(0.5, 1.0)) +
    scale_edge_alpha_continuous(range = c(0.05, 0.05)) +
    labs(fill = "Populist votes (%)") +
    scale_fill_steps2(
      low = "Red",
      mid = "grey",
      high = "Blue",
      midpoint = df_election_results$populist_votes |> mean()
    ) +
    coord_sf(ylim = c(3e5, NA)) +
    theme(
      strip.text = element_text(size = 14),
      legend.justification = "center",
      legend.direction = "horizontal",
      legend.title.position = "top",
      legend.title = element_text(hjust = 0.5),
      legend.box = "horizontal",
      legend.box.background = element_blank()
    )

  return(p)
}

# Only plot most positive edges
p1 <- plot_sf_network(
  gr |>
    activate(edges) |>
    filter(weight > quantile(weight, probs = 0.99), num_edges > 100),
  gr_hierarchy,
  populist_votes
) +
  facet_wrap(vars("Most positive"))

# Only plot most negative edges
p2 <- plot_sf_network(
  gr |>
    activate(edges) |>
    filter(weight < quantile(weight, probs = 0.01), num_edges > 100),
  gr_hierarchy,
  populist_votes
) +
  facet_wrap(vars("Most negative"))

plot_grid(
  get_legend(p1),
  plot_grid(
    p1 + theme(legend.position = "none"),
    p2 + theme(legend.position = "none"),
    ncol = 2,
    align = "hv",
    axis = "tblr"
  ),
  ncol = 1,
  rel_heights = c(0.2, 1.0),
  labels = "C",
  label_size = 26
)

ggsave(file.path("figures", "municipality_networks.png"), width = 10, height = 5)


# Plot municipality correlations --------------------------------------------------------------

# Define function to convert labels
convert_variable_labels <- function(x) {
  label <- case_match(
    x,
    "niet_westers_totaal_18" ~ "Non-western",
    "opleidingsniveau_laag_64" ~ "Lower education",
    "opleidingsniveau_middelbaar_65" ~ "Intermediate education",
    "opleidingsniveau_hoog_66" ~ "Higher education",
    "populist_votes" ~ "Populist votes",
    "k_0tot15jaar_8" ~ "0-15 years",
    "k_15tot25jaar_9" ~ "15-25 years",
    "k_65jaar_of_ouder_12" ~ "> 64 years",
    "gemiddeld_inkomen_per_inwoner_72" ~ "Mean income per capita",
    "mediaan_vermogen_van_particuliere_huish_82" ~ "Median household assets",
    "a_landbouw_bosbouw_en_visserij_92" ~ "Agriculture/forestry/\nfishery businesses",
    "omgevingsadressendichtheid_117" ~ "Address density",
    "degree" ~ "Municipality degree"
  )

  return(label)
}

# Define function to calc correlations between municipality variables
calc_degree_correlations <- function(gr) {
  return(gr |>
    activate(nodes) |>
    mutate(
      degree = centrality_degree(weights = weight),
      across(c(
        k_0tot15jaar_8, k_15tot25jaar_9, k_65jaar_of_ouder_12,
        niet_westers_totaal_18,
        opleidingsniveau_laag_64, opleidingsniveau_middelbaar_65,
        opleidingsniveau_hoog_66
      ), ~ .x / aantal_inwoners_5)
    ) |>
    data.frame() |>
    select(
      degree,
      populist_votes,
      k_0tot15jaar_8,
      k_15tot25jaar_9,
      k_65jaar_of_ouder_12,
      niet_westers_totaal_18,
      opleidingsniveau_laag_64,
      opleidingsniveau_middelbaar_65,
      opleidingsniveau_hoog_66,
      gemiddeld_inkomen_per_inwoner_72,
      mediaan_vermogen_van_particuliere_huish_82,
      omgevingsadressendichtheid_117
    ) |>
    correlate())
}

# Define function to plot municipality correlations
plot_degree_correlations <- function(df_cor, variable) {
  variable <- enquo(variable)

  p <- df_cor |>
    filter(!is.na(!!variable)) |>
    mutate(term = fct_reorder(term, !!variable)) |>
    ggplot(aes(x = !!variable, y = term)) +
    geom_point() +
    geom_segment(aes(x = 0, xend = !!variable)) +
    scale_x_continuous(limits = c(-0.8, 0.8), breaks = c(seq(-0.8, 0.8, 0.4))) +
    scale_y_discrete(name = NULL, labels = convert_variable_labels) +
    theme_half_open() +
    theme(
      axis.line.y = element_blank(),
      plot.margin = unit(c(1, 0.2, 0, 0), units = "cm")
    )

  return(p)
}

df_cor <- calc_degree_correlations(gr)

p3 <- plot_degree_correlations(df_cor, degree) +
  labs(x = "Correlation average edge utility strength")

p4 <- plot_degree_correlations(df_cor, populist_votes) +
  labs(x = "Correlation populist votes")

plot_grid(
  p3,
  p4,
  ncol = 2,
  align = "hv",
  axis = "tblr",
  labels = c("A", "B"),
  label_size = 26
)

ggsave(file.path("figures", "municipality_correlations.png"), width = 10, height = 5)


# Sensitivity analysis ------------------------------------------------------------------------

df_covariates_2022 <- read.csv(file.path(raw_dir, "85318NED_TypedDataSet_03042025_112801_2022.csv"), sep = ";") |>
  clean_names() |>
  mutate(soort_regio_2 = trimws(soort_regio_2), gemeentenaam_1 = trimws(gemeentenaam_1)) |>
  filter(soort_regio_2 == "Gemeente") |>
  rename(gemeente = gemeentenaam_1)

geo_municipality_2022 <- geo_municipality |>
  left_join(df_covariates_2022, by = "gemeente") |>
  rename(province = statnaam) |>
  group_by(province)

df_edgelist_sensitivity <- read.csv(file.path(export_dir, "collapsed_2022_len20_window2_walks100_dim32_dine_utilities_dim20_community_edgelist.csv"))

df_node_attributes_sensitivity <- read.csv(file.path(export_dir, "collapsed_2022_municipalities_stats.csv"))

gr_sensitivity <- create_graph_from_edgelist(df_edgelist_sensitivity, df_node_attributes_sensitivity, geo_municipality_2022)

gr_hierarchy_sensitivity <- create_hierarchy_graph(gr_sensitivity, df_node_attributes_sensitivity, geo_municipality_2022)

p1_s <- plot_sf_network(
  gr_sensitivity |>
    activate(edges) |>
    filter(weight > quantile(weight, probs = 0.99), num_edges > 100),
  gr_hierarchy_sensitivity,
  populist_votes
) +
  facet_wrap(vars("Most positive"))

p2_s <- plot_sf_network(
  gr_sensitivity |>
    activate(edges) |>
    filter(weight < quantile(weight, probs = 0.01), num_edges > 100),
  gr_hierarchy_sensitivity,
  populist_votes
) +
  facet_wrap(vars("Most negative"))

plot_grid(
  get_legend(p1_s),
  plot_grid(
    p1_s + theme(legend.position = "none"),
    p2_s + theme(legend.position = "none"),
    ncol = 2,
    align = "hv",
    axis = "tblr"
  ),
  ncol = 1,
  rel_heights = c(0.2, 1.0),
  labels = "C",
  label_size = 26
)

ggsave(file.path("figures", "municipality_networks_sensitivity.png"), width = 10, height = 5)

df_cor_sensititity <- calc_degree_correlations(gr_sensitivity)

p3_s <- plot_degree_correlations(df_cor_sensititity, degree) +
  labs(x = "Correlation with municipality degree")

p4_s <- plot_degree_correlations(df_cor_sensititity, populist_votes) +
  labs(x = "Correlation with populist votes")

plot_grid(
  p3_s,
  p4_s,
  ncol = 2,
  align = "hv",
  axis = "tblr",
  labels = c("A", "B"),
  label_size = 26
)

ggsave(file.path("figures", "municipality_correlations_sensitivity.png"), width = 10, height = 5)
