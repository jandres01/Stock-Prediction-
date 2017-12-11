
#### Header ####

# Program created to pull data from CRSP/Compustat to use for building an neural
# network that can identify the stocks that will have the highest decile of alpha
# next period and the lowest decile of alpha to construct a long-short portfolio
#
# Created:
#   Date - Mar. 31, 2017
#   R Version - 3.3.3 (RStudio Version 1.0.136)
# Last Modified:
#   Date - Mar. 31, 2017
#   R Version - 3.3.3 (RStudio Version 1.0.136)


#### Sources ####
  # Increase Java max heap size to 12 GB
  options(java.parameters = "-Xmx64g")
  # Increase warning length to max allowed so if SQL statements produce error it will all print
  options(warning.length = 8170)
  library("wrdsr")
  library("dplyr") 
  library("readr")
  library("tibble")
  library("indclass")
  library("lubridate")
  library("zoo")

#### Establish connection to WRDS ####

  wrds_db <- wrdsConnect("jestover", "mkQ2NiG86D$UuUw@U$0l", librefs = "wrdssec '/wrds/sec/sasdata'", conn = "postgresql")
  local_db <- src_sqlite(path.expand("/home/jandres/data/revised-monthly-momentum-financial-data.sqlite"), create = T)

##### Pull data from CRSP Monthly Securities and CRSP/Compustat Merged #####
  
  # Merging Compustat values with CRSP 3 months after the end of the fiscal year to make sure it is after the data is available to the market
  
  sql <- "SELECT permno AS permno,
                 date AS date,
                 ret AS return,
                 prc AS price
          FROM crspa.msf AS msf
          ORDER BY permno, date"
  # Fetch the desired data
  results <- dbSendQuery(wrds_db, sql)
  crsp <- as_data_frame(dbFetch(results))
 
  crsp <- crsp %>%
    mutate(date = as.Date(date),
           month = month(date),
           year = year(date)) %>%
    arrange(permno, date) %>%
    group_by(permno) %>%
    mutate(next_month_return = lead(return),
           return_1 = lag(return),
           return_2 = lag(return, n = 2),
           return_3 = lag(return, n = 3),
           return_4 = lag(return, n = 4),
           return_5 = lag(return, n = 5),
           return_6 = lag(return, n = 6),
           return_7 = lag(return, n = 7),
           return_8 = lag(return, n = 8),
           return_9 = lag(return, n = 9),
           return_10 = lag(return, n = 10),
           return_11 = lag(return, n = 11),
           total_return = ((1 + return_1) * (1 + return_2) * (1 + return_3) * (1 + return_4) * (1 + return_5) * (1 + return_6) * (1 + return_7) * (1 + return_8) * (1 + return_9) * (1 + return_10) * (1 + return_11)) - 1,
           price = abs(price)) %>%
    ungroup() %>% 
    group_by(date) %>% 
    mutate(next_month_return_perc = percent_rank(next_month_return),
           total_return_perc = percent_rank(total_return),
           perc_1 = percent_rank(return_1),
           perc_2 = percent_rank(return_2),
           perc_3 = percent_rank(return_3),
           perc_4 = percent_rank(return_4),
           perc_5 = percent_rank(return_5),
           perc_6 = percent_rank(return_6),
           perc_7 = percent_rank(return_7),
           perc_8 = percent_rank(return_8),
           perc_9 = percent_rank(return_9),
           perc_10 = percent_rank(return_10),
           perc_11 = percent_rank(return_11)) %>%
    ungroup() %>%
    mutate(prod_1 = return_1 * perc_1,
           prod_2 = return_2 * perc_2,
           prod_3 = return_3 * perc_3,
           prod_4 = return_4 * perc_4,
           prod_5 = return_5 * perc_5,
           prod_6 = return_6 * perc_6,
           prod_7 = return_7 * perc_7,
           prod_8 = return_8 * perc_8,
           prod_9 = return_9 * perc_9,
           prod_10 = return_10 * perc_10,
           prod_11 = return_11 * perc_11,
           total_return_prod = total_return * total_return_perc)
  
  # Fremove observations with missing return data and with share prices below $5 
  # (Following Zhang 2006 - Information Uncertainty and Stock Returns)
  crsp <- crsp %>%
    filter(price >= 5, !is.na(return), !is.na(next_month_return), !is.na(next_month_return_perc), !is.na(return_1), 
           !is.na(return_2), !is.na(return_3), !is.na(return_4), !is.na(return_5), !is.na(return_6), 
           !is.na(return_7), !is.na(return_8), !is.na(return_9), !is.na(return_10), !is.na(return_11))
  
#  db_drop_table(local_db$con, "crsp") 
  copy_to(local_db, crsp, temporary = FALSE,
          indexes = list(c("permno", "date"), "date"))
  rm(crsp)
  gc()
  
