
setwd("E:/R")

# Activation of necessary packages ----

pacman::p_load(tidyverse, lubridate, janitor,
               tidymodels, bestNormalize,
               Hmisc, ggstatsplot, GGally, skimr, # for EDA
               tidymodels, bestNormalize, # Tidy ML
               DT, plotly, # Interactive Data Display
               vip, # Feature Importance
               broom, jtools, interactions, # report models
               themis, foreach,
               xgboost, ranger, kernlab, finetune) # setting your dependencies

sessionInfo()

load("car_finalized.RData")

# IMPORT ----

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
  mutate(log10_MSRP = log10(msrp)
         ) %>% 
  mutate_if(is.character, factor) %>% 
  select(-msrp)

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
  step_orderNorm(all_numeric_predictors()
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

recipe_imputemean_yeo <- 
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
             )

recipe_imputemean_yeo %>% 
  prep(verbose = T)

recipe_imputeknn_yeo <- 
  recipe(formula = log10_MSRP ~ ., 
         data = car_training) %>%
  update_role(model, new_role = "id variable") %>% 
  step_zv(all_numeric_predictors()
          ) %>%
  step_unknown(engine_fuel_type, 
               new_level = "unknown") %>% 
  step_impute_knn(engine_cylinders, engine_hp,
                   number_of_doors) %>% 
  step_YeoJohnson(all_numeric_predictors()
                  ) %>%
  step_normalize(all_numeric_predictors()
                 ) %>%
  step_dummy(all_nominal_predictors()
             )

recipe_imputeknn_yeo %>% 
  prep(verbose = T)

recipe_imputeknn_pca <- 
  recipe(formula = log10_MSRP ~ ., 
         data = car_training) %>%
  update_role(model, new_role = "id variable") %>% 
  step_zv(all_numeric_predictors()
          ) %>% 
  step_unknown(engine_fuel_type, 
               new_level = "unknown") %>% 
  step_impute_knn(engine_cylinders, engine_hp,
                  number_of_doors) %>% 
  step_YeoJohnson(all_numeric_predictors()
                  ) %>%
  step_normalize(all_numeric_predictors()
                 ) %>% 
  step_pca(all_numeric_predictors(),
           num_comp = 2
           ) %>% 
  step_dummy(all_nominal_predictors()
             ) 

recipe_imputeknn_pca %>% 
  prep(verbose = T)

recipe_imputemean_interact1 <- 
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
  step_interact(terms = ~ city_mpg:engine_cylinders
                )
  # step_interact(terms = ~ city_mpg:highway_mpg) %>% 
  # step_interact(terms = ~ city_mpg:engine_cylinders)
# step_interact(terms = ~ transmission_type:engine_fuel_type
# )

recipe_imputemean_interact1 %>% 
  prep(verbose = T)

recipe_imputemean_interact2 <- 
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
  step_interact(terms = ~ city_mpg:engine_cylinders
                ) %>% 
  step_interact(terms = ~ city_mpg:highway_mpg
                ) %>% 
  step_interact(terms = ~ highway_mpg:engine_cylinders
                )

recipe_imputemean_interact2 %>% 
  prep(verbose = T)

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
  size = 10
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
  size = 10
  )

model_metrics <- metric_set(rmse, rsq, mae)


all_workflows <-
  workflow_set(preproc = list(imputemean_interact1 =
                                recipe_imputemean_interact1,
                              # imputeknn_yeo = 
                              #   recipe_imputeknn_yeo,
                              # imputeknn_yeo_pca =
                              #   recipe_imputeknn_pca,
                              imputemean_interact2 =
                                recipe_imputemean_interact2
                              ),
  models = list(#random_forest = RF,
                XGB = XG_BOOST)
  ) %>% 
  # option_add(grid = rf_grid
  #            ) %>%
  option_add(grid = xgb_grid, id = "imputemean_interact1_XGB"
             ) %>%
  # option_add(grid = xgb_grid, id = "imputeknn_yeo_XGB"
  #            ) %>% 
  # option_add(grid = xgb_grid, id = "imputeknn_yeo_pca_XGB"
  #            ) %>% 
  option_add(grid = xgb_grid, id = "imputemean_interact2_XGB"
             )

#cl <- makeCluster(no_cores, type = "SOCK")#, setup_strategy = "sequential")
# registerDoParallel(cl)
#getDoParWorkers()
# registerDoMC(cores = no_cores)

library(doParallel)
cores <- detectCores()
cl <- makeCluster(cores[1]-1)
doParallel::registerDoParallel(cl)

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

previous_run

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

furniture_names <- 
  furniture %>% 
  janitor::clean_names()

### Plotting Metrics ----

for_naming <- 
  furniture_names %>% 
  rowid_to_column(".row")

data_for_plot <- 
  for_naming %>% 
  inner_join(individual_countries_predictions,
             by = ".row") %>% 
  dplyr::select(.row, item_id, name, category, .pred, price) %>% 
  rename(actual_price = price) %>% 
  mutate(predicted_price = 10^.pred)

ggthemr("dust")

set.seed(220070261)

data_for_plot %>% 
  ggplot(aes(x = actual_price,
             y = predicted_price,
             text = name,
             label = item_id)
  ) +
  geom_abline(color = "red",
              lty = 2) + 
  geom_label_repel() + 
  # geom_point() + 
  coord_obs_pred() + # note that this comes from tune::
  scale_x_continuous(labels = scales::dollar) + 
  scale_y_continuous(labels = scales::dollar) +
  labs(x = "Actual Price of Furniture",
       y = "Predicted Price of Furniture",
       title = "Predicting IKEA Furniture Price",
       subtitle = "Currency = USD") 

for_plotly <- 
  data_for_plot %>% 
  ggplot(aes(x = actual_price,
             y = predicted_price,
             text = name,
             label = category)
  ) +
  geom_abline(color = "red",
              lty = 2) + 
  geom_point(color = "deepskyblue2",
             alpha = 0.50) + 
  # coord_obs_pred() + # note that this comes from tune::
  scale_x_continuous(labels = scales::dollar) + 
  scale_y_continuous(labels = scales::dollar) +
  labs(x = "Actual Price of Furniture",
       y = "Predicted Price of Furniture",
       title = "Predicting IKEA Furniture Price",
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
  add_recipe(recipe_knn_impute_interact_3) %>% 
  add_model(importance) %>% # UPDATED here
  fit(analysis(furniture_validate$splits[[1]]
  )
  ) %>% 
  extract_fit_parsnip() %>% 
  vip::vip(aesthetics = list(fill = "deepskyblue3",
                             alpha = 0.75)
  )

model_summary_for_importance 

# Finalized Model ----

finalized_xgb <- 
  boost_tree(trees = 1000,
             mtry = 28,
             min_n = 14,
             tree_depth = 11,
             sample_size = 0.564700039247982,
             learn_rate = 0.0462100793562759
  ) %>% 
  set_engine("xgboost") %>% 
  set_mode("regression")

finalized_model <- 
  workflow() %>% 
  add_recipe(recipe_knn_impute_interact_3) %>%
  add_model(finalized_xgb) %>% 
  fit(furniture_cleaned)

# Store Your Algorithm ----
finalized_model %>% 
  saveRDS("furniture_finalized_model.rds")

# Store Your RAM Space ----
save.image("car_finalized.RData")
# Thank you for working with the script :)
