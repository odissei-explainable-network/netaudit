# Analysis Scripts

This folder contains analysis scripts for the paper "Using Interpretable Network Embeddings to 
Understand Right-Wing Populist Voting Behavior in Population-Scale Registry Networks" (link).

Running the scripts requires aggregated data in the `data/raw/` folder (available on OSF).

The original analysis was run in R with the following setup (returned by `sessionInfo`):
```
R version 4.4.2 (2024-10-31 ucrt)
Platform: x86_64-w64-mingw32/x64
Running under: Windows 10 x64 (build 19045)

Matrix products: default


locale:
[1] LC_COLLATE=English_Netherlands.utf8  LC_CTYPE=English_Netherlands.utf8    LC_MONETARY=English_Netherlands.utf8
[4] LC_NUMERIC=C                         LC_TIME=English_Netherlands.utf8    

time zone: Europe/Amsterdam
tzcode source: internal

attached base packages:
[1] stats     graphics  grDevices utils     datasets  methods   base     

loaded via a namespace (and not attached):
 [1] utf8_1.2.4        R6_2.5.1          tidyselect_1.2.1  magrittr_2.0.3    gtable_0.3.6      glue_1.8.0        tibble_3.2.1     
 [8] pkgconfig_2.0.3   generics_0.1.3    dplyr_1.1.4       lifecycle_1.0.4   ggplot2_3.5.1     cli_3.6.3         fansi_1.0.6      
[15] scales_1.3.0      grid_4.4.2        vctrs_0.6.5       compiler_4.4.2    rstudioapi_0.17.1 tools_4.4.2       munsell_0.5.1    
[22] pillar_1.9.0      colorspace_2.1-1  rlang_1.1.4
```