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
  options(java.parameters = "-Xmx12g")
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
  local_db <- src_sqlite(path.expand("/home/jandres/data/momentum-financial-data.sqlite"), create = T)

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
           return_lag_1 = lag(return),
           return_lag_2 = lag(return, n = 2),
           return_lag_3 = lag(return, n = 3),
           return_lag_4 = lag(return, n = 4),
           return_lag_5 = lag(return, n = 5),
           return_lag_6 = lag(return, n = 6),
           return_lag_7 = lag(return, n = 7),
           return_lag_8 = lag(return, n = 8),
           return_lag_9 = lag(return, n = 9),
           return_lag_10 = lag(return, n = 10),
           return_lag_11 = lag(return, n = 11),
           total_past_return = (((1 + return_lag_1) * (1 + return_lag_2) * (1 + return_lag_3) * (1 + return_lag_4) * (1 + return_lag_5) * (1 + return_lag_6) * (1 + return_lag_7) * (1 + return_lag_8) * (1 + return_lag_9) * (1 + return_lag_10) * (1 + return_lag_11)) - 1),
           price = abs(price)) %>%
    ungroup() %>% 
    group_by(date) %>% 
    mutate(next_month_return_percentile = percent_rank(next_month_return),
           total_past_return_percentile = percent_rank(total_past_return)) %>%
    ungroup()
  
  # Fremove observations with missing return data and with share prices below $5 
  # (Following Zhang 2006 - Information Uncertainty and Stock Returns)
  crsp <- crsp %>%
    filter(price >= 5, !is.na(return), !is.na(next_month_return), !is.na(next_month_return_percentile), !is.na(total_past_return),
           !is.na(total_past_return_percentile), !is.na(return_lag_1), !is.na(return_lag_2), !is.na(return_lag_3),
           !is.na(return_lag_4), !is.na(return_lag_5), !is.na(return_lag_6), !is.na(return_lag_7), !is.na(return_lag_8),
           !is.na(return_lag_9), !is.na(return_lag_10), !is.na(return_lag_11))
  
  #db_drop_table(local_db$con, "crsp") 
  copy_to(local_db, crsp, temporary = FALSE,
          indexes = list(c("permno", "date"), "date"))
  rm(crsp)
  gc()
  
