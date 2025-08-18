# Prepare LISS data for import ----------------------------------------------------------------

library(dplyr)

df_values <- read.csv2("cv24p_EN_1.0p.csv")

df_values_sub <- df_values %>%
    select(nomem_encr,
           cv24p001, # general satisfaction with government
           num_range(prefix = "cv24p0", range = c(
               13:29, # confidence in institutions
               47:52, # statements
               53 # vote: yes/no 2017
           )),
           cv24p307 # vote for which party
    )


df_pers <- read.csv2("cp24p_EN_1.0p.csv")

df_pers_sub <- df_pers %>%
    select(nomem_encr, num_range(prefix = "cp24p0", range = c(
        14:18, # life satisfaction
        19 # trust
    )))


df_join <- df_values %>%
    full_join(df_pers, by = "nomem_encr")


write.csv(df_join, "liss_data_import_cbs.csv", row.names = FALSE)
