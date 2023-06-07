
# Activation of necessary packages ----

pacman::p_load(tidyverse, lubridate, janitor,
               tidymodels, bestNormalize,
               Hmisc, ggstatsplot, GGally, skimr, # for EDA
               tidymodels, bestNormalize, # Tidy ML
               DT, plotly, ggthemr, ggrepel, # Interactive Data Display
               vip, # Feature Importance
               broom, jtools, interactions, # report models
               themis, foreach,
               xgboost, ranger, kernlab, finetune) # setting your dependencies

install.packages("devtools")
devtools::install_github('Mikata-Project/ggthemr')
sessionInfo()

setwd("I:/R")

drive_download("~/R_data_portfolio/car_prediction/car_finalized.RData",
               overwrite = TRUE)

load("car_finalized.RData")

# IMPORT ----

install.packages("ggrepel")

library(googledrive)

drive_download("~/R_data_portfolio/car_prediction/Car_price_prediction.csv",
  overwrite = TRUE)

car <- read_csv("Car_price_prediction.csv")

# TIDY ----

car %>% 
  skim()

# check for missing using base R
car %>% 
  sapply(function(x) sum(is.na (x)
                         )
         )

# see the na data
car %>% 
  filter(if_any(everything(), is.na)
         ) %>% 
  print(n = nrow(.)
        )

unique(car$`Engine Fuel Type`)
unique(car$Model)

(colMeans(is.na(car)))*100

car %>% 
  filter(Make == "Suzuki") %>% 
  print(n = nrow(.))

# getting all the variables name in tbl
car %>% 
  names(.) %>% 
  as_tibble()

# TRANSFORMATION ----

car_cleaned <- 
  car %>% 
  clean_names() %>% 
  dplyr::distinct() %>% 
  mutate(log10_MSRP = log10(msrp)
         ) %>% 
  mutate_if(is.character, factor) %>% 
  select(-msrp)

dup_car_cleaned <- 
  car_cleaned %>%
  group_by_all() %>%
  filter(n()>1) %>%
  ungroup()

car_cleaned  %>% 
  skim()

# EDA ----

car_eda <- 
  recipe( ~ .,
          data = car_cleaned) %>% 
  step_rm(model) %>% 
  step_unknown(engine_fuel_type, 
               new_level = "unknown") %>% 
  step_impute_mean(engine_cylinders, engine_hp,
                   number_of_doors) %>% 
  step_YeoJohnson(all_numeric_predictors()
              ) %>%
  step_normalize(all_numeric_predictors()
                 ) %>%
  step_dummy(all_nominal_predictors()
             ) %>%
  prep(verbose = T) %>% 
  bake(new_data = NULL)

car_eda %>% 
  skim()

car_eda %>% 
  sapply(function(x) sum(is.na (x)
                         )
         )

unique(car_eda$`Engine Fuel Type`)

car_corr_mat <-
  car_eda %>% 
  as.matrix(.) %>% 
  rcorr(.) %>% 
  tidy(.) %>% 
  rename(var1 = column1,
         var2 = column2,
         corr = estimate) %>% 
  mutate(absCORR = abs(corr)
         ) %>% 
  # filter(var1 == "log10_MSRP" |
  #          var2 == "log10_MSRP") %>%
  .[,c(-4,-5)] %>% 
  arrange(desc(absCORR)
          ) %>% 
  datatable()

car_cleaned %>% 
  ggplot(aes(x = 	highway_mpg,
             y = log10_MSRP)
         ) + 
  geom_point(color = "dodgerblue",
             alpha = 0.25) +
  geom_smooth(method = "lm",
              formula = y ~ x,
              se = F,
              color = "red") + 
  geom_smooth(method = "lm",
              formula = y ~ poly(x,
                                 degree = 3),
              se = F,
              color = "green") + 
  geom_smooth(method = "loess", # LOESS for EDA
              formula = y ~ x,
              se = F,
              color = "purple") + 
  theme_bw()

# PREDICTIVE MODEL ----

## Step 1: Data Split ----

set.seed(05062301)

### Planning stage of Split ----
car_split <- # this is a random sample scheme
  car_cleaned %>% 
  initial_split(prop = .80) # by default, 75% for training set

car_split

### Execution stage of Split ----

car_training <- 
  car_split %>% 
  training()

car_testing <- 
  car_split %>% 
  testing()

car_training

set.seed(05062302)

car_validate <- 
  car_training %>% 
  validation_split(prop = 0.9)

## Step 2: Pre-Process (Feature Engineering) ----

### Pre-processing ----

car_training %>% 
  names(.) %>% 
  as_tibble()

car_training %>% 
  skim()

recipe_imputemean_poly <- 
  recipe(formula = log10_MSRP ~ ., 
         data = car_training) %>%
  update_role(model, new_role = "id variable") %>% 
  step_zv(all_numeric_predictors()
          ) %>% 
  step_unknown(engine_fuel_type, 
               new_level = "unknown") %>% 
  step_impute_mean(engine_cylinders, engine_hp,
                  number_of_doors) %>% 
  step_YeoJohnson(all_numeric_predictors()
                  ) %>%
  step_normalize(all_numeric_predictors()
                 ) %>% 
  step_dummy(all_nominal_predictors()
             ) %>% 
  step_poly(engine_hp, 
            degree = 2)

recipe_imputemean_poly %>% 
  prep(verbose = T)

recipe_imputemean_interact <- 
  recipe(formula = log10_MSRP ~ ., 
         data = car_training) %>%
  update_role(model, new_role = "id variable") %>% 
  step_zv(all_numeric_predictors()
          ) %>% 
  step_unknown(engine_fuel_type, 
               new_level = "unknown") %>% 
  step_impute_mean(engine_cylinders, engine_hp,
                   number_of_doors) %>% 
  step_YeoJohnson(all_numeric_predictors()
                  ) %>%
  step_normalize(all_numeric_predictors()
                 ) %>% 
  step_dummy(all_nominal_predictors()
             ) %>% 
  step_interact(terms = ~ engine_hp:engine_cylinders:year
                )

recipe_imputemean_interact %>% 
  prep(verbose = T)

recipe_imputemean_interactpoly <- 
  recipe(formula = log10_MSRP ~ ., 
         data = car_training) %>%
  update_role(model, new_role = "id variable") %>% 
  step_zv(all_numeric_predictors()
          ) %>% 
  step_unknown(engine_fuel_type, 
               new_level = "unknown") %>% 
  step_impute_mean(engine_cylinders, engine_hp,
                   number_of_doors) %>% 
  step_YeoJohnson(all_numeric_predictors()
                  ) %>%
  step_normalize(all_numeric_predictors()
                 ) %>% 
  step_dummy(all_nominal_predictors()
             ) %>% 
  step_interact(terms = ~ ends_with("mpg"):engine_cylinders
                ) %>% 
  step_poly(engine_hp, 
            degree = 3) %>% 
  step_poly(year, 
            degree = 3)

recipe_imputemean_interactpoly %>% 
  prep(verbose = T) %>% 
  bake(new_data = NULL) %>% 
  skim()

## Step 3: FIT ----

RF <- 
  rand_forest() %>% 
  set_args(trees = tune(), 
           mtry = tune(),
           min_n = tune()
           ) %>% 
  set_engine("ranger",
             importance = "impurity") %>% 
  set_mode("regression")


XG_BOOST <- 
  boost_tree(trees = tune(),
             mtry = tune(),
             min_n = tune(),
             tree_depth = tune(),
             sample_size = tune(),
             learn_rate = tune(),
             loss_reduction = tune()
             ) %>% 
  set_engine("xgboost") %>% 
  set_mode("regression")

set.seed(05062303)

cv10 <- 
  car_training %>% 
  vfold_cv(v = 10)

## Step 4: TUNE ----

xg_param <-
  XG_BOOST %>% 
  extract_parameter_set_dials() %>% 
  update(mtry = finalize(mtry(),  car_training)
         )

xg_grid <- xg_param %>% 
  grid_max_entropy(size = 10)

xgb_grid <- grid_max_entropy(
  trees(),
  finalize(mtry(), car_training),
  min_n(),
  tree_depth(),
  sample_size = sample_prop(),
  learn_rate(),
  loss_reduction(),
  size = 20
  )

rf_param <- 
  RF %>% 
  extract_parameter_set_dials() %>% 
  update(mtry = finalize(mtry(),  car_training)
  )

rf_grid <- 
  rf_param %>% 
  grid_latin_hypercube(size = 10) 

rf_grid <- grid_latin_hypercube(
  trees(),
  finalize(mtry(), car_training),
  min_n(),
  size = 20
  )

model_metrics <- metric_set(rmse, rsq, mae)


all_workflows <-
  workflow_set(preproc = list(imputemean_interactpoly =
                                recipe_imputemean_interactpoly
                              # imputemean_poly =
                              #   recipe_imputemean_poly,
                              # # imputeknn_yeo_pca =
                              # #   recipe_imputeknn_pca,
                              # imputemean_interact =
                              #   recipe_imputemean_interact
                              ),
  models = list(#random_forest = RF,
                XGB = XG_BOOST)
  ) %>% 
  # option_add(grid = rf_grid
  #            ) %>%
  # option_add(grid = xgb_grid, id = "imputemean_interact_XGB"
  #            ) %>%
  # # # option_add(grid = xgb_grid, id = "imputeknn_yeo_XGB"
  # # #            ) %>% 
  # option_add(grid = xgb_grid, id = "imputemean_poly_XGB"
  #            ) %>%
  option_add(grid = xgb_grid, id = "imputemean_interactpoly_XGB"
             )

#cl <- makeCluster(no_cores, type = "SOCK")#, setup_strategy = "sequential")
# registerDoParallel(cl)
#getDoParWorkers()
# registerDoMC(cores = no_cores)

library(doParallel)
cores <- detectCores()
cl <- makeCluster(cores[1]-1)
doParallel::registerDoParallel(cl)

# doParallel::registerDoParallel()

CONTROL_TOWER_ML <- 
  all_workflows %>% 
  workflow_map(verbose = T,
               seed = 05062304, 
               resamples = cv10, 
               metrics = model_metrics,
               control = control_grid(parallel_over = "everything",
                                      save_workflow = T,
                                      verbose = T,
                                      save_pred = T) 
               ) 

RANK <-
  CONTROL_TOWER_ML %>% 
  rank_results(select_best = T)



RANKINGS_cleaned <- 
  RANK %>% 
  mutate(method = map_chr(wflow_id,
                          ~ str_split(.x,
                                      "_",
                                      simplify = T)[1]
                          )
         ) %>% 
  # filter(rank <= 3 & .metric == "rmse") %>% 
  dplyr::select(wflow_id, model, .config, rmse = mean, rank) %>% 
  group_by(wflow_id) %>% 
  slice_min(rank,
            with_ties = F) %>% 
  ungroup() %>% 
  arrange(rank)

RANKINGS_cleaned

best_run <- RANKINGS_cleaned

workflow_ID_best <-
  RANKINGS_cleaned %>% 
  slice_min(rank,
            with_ties = F) %>% 
  pull(wflow_id)

workflow_best <-
  CONTROL_TOWER_ML %>% 
  extract_workflow_set_result(workflow_ID_best) %>% 
  select_best(metric = "rmse")

FINALIZED_workflow <- 
  CONTROL_TOWER_ML %>% 
  extract_workflow(workflow_ID_best) %>% # There were multiple models
  finalize_workflow(workflow_best)

best_fit <- 
  FINALIZED_workflow %>% # The optimal parameters for the chosen one
  last_fit(car_split)

## Assess ----

### Model Level Metrics ----

best_fit_metrics <-
  best_fit %>% 
  collect_metrics()

best_fit_metrics

### Individual Level Metrics ----

individual_countries_predictions <- 
  best_fit %>% 
  collect_predictions()

individual_countries_predictions

car_names <- 
  car %>% 
  janitor::clean_names()

### Plotting Metrics ----

for_naming <- 
  car_names %>% 
  rowid_to_column(".row")

data_for_plot <- 
  for_naming %>% 
  inner_join(individual_countries_predictions,
             by = ".row") %>% 
  dplyr::select(.row, make, model, .pred, msrp) %>% 
  rename(actual_msrp = msrp) %>% 
  mutate(predicted_msrp = 10^.pred)

ggthemr("dust")

set.seed(05062305)

data_for_plot %>% 
  ggplot(aes(x = actual_msrp,
             y = predicted_msrp,
             text = make,
             label = model)
  ) +
  geom_abline(color = "red",
              lty = 2) + 
  geom_label_repel() + 
  # geom_point() + 
  coord_obs_pred() + # note that this comes from tune::
  scale_x_continuous(labels = scales::dollar) + 
  scale_y_continuous(labels = scales::dollar) +
  labs(x = "Actual MSRP of Cars",
       y = "Predicted MSRP of Cars",
       title = "Predicting Car MSRP",
       subtitle = "Currency = USD") 

for_plotly <- 
  data_for_plot %>% 
  ggplot(aes(x = actual_msrp,
             y = predicted_msrp,
             text = make,
             label = model)
  ) +
  geom_abline(color = "red",
              lty = 2) + 
  geom_point(color = "deepskyblue2",
             alpha = 0.50) + 
  # coord_obs_pred() + # note that this comes from tune::
  scale_x_continuous(labels = scales::dollar) + 
  scale_y_continuous(labels = scales::dollar) +
  labs(x = "Actual MSRP of Cars",
       y = "Predicted MSRP of Cars",
       title = "Predicting Car MSRP",
       subtitle = "Currency = USD") 

for_plotly %>% 
  ggplotly()

# Variable (Feature) Importance ----

library(vip)

importance <- 
  XG_BOOST %>% 
  finalize_model(workflow_best
  ) %>% 
  set_engine("xgboost")

model_summary_for_importance <- 
  workflow() %>% 
  add_recipe(recipe_imputemean_interactpoly) %>% 
  add_model(importance) %>% # UPDATED here
  fit(analysis(car_validate$splits[[1]]
               )
      ) %>% 
  extract_fit_parsnip() %>% 
  vip::vip(aesthetics = list(fill = "deepskyblue3",
                             alpha = 0.75)
           )

model_summary_for_importance 

# Finalized Model ----

finalized_xgb <- 
  boost_tree(trees = 1346,
             mtry = 15,
             min_n = 35,
             tree_depth = 7,
             sample_size = 0.827604670915753,
             learn_rate = 0.0728621988626886
             ) %>% 
  set_engine("xgboost") %>% 
  set_mode("regression")

finalized_model <- 
  workflow() %>% 
  add_recipe(recipe_imputemean_interactpoly) %>%
  add_model(finalized_xgb) %>% 
  fit(car_cleaned)

# Store Your Algorithm ----
finalized_model %>% 
  saveRDS("car_finalized_model.rds")

# Store Your RAM Space ----
save.image("car_finalized.RData")
# Thank you for working with the script :)
