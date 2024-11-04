library(tidyverse)
library(tidymodels)
library(vroom)
library(glmnet)
library(recipes)
library(embed)
library(lme4)
library(kknn)


test <- vroom("test.csv")
train <- vroom("train.csv")
train$ACTION <- as.factor(train$ACTION)

recipeTargetEncoded <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_nominal_predictors())


# define model
tree_mod <- rand_forest(mtry = tune(),
                        min_n=tune(),
                        trees=500) %>%
  set_engine("ranger") %>% 
  set_mode("classification")

# create workflow
tree_wf <- workflow() %>%
  add_recipe(recipeTargetEncoded) %>%
  add_model(tree_mod)

# set up grid of tuning values
tuning_params <- grid_regular(mtry(range = c(1,10)),
                              min_n(),
                              levels = 5)

# set up k-fold CV
folds <- vfold_cv(train, v = 5, repeats=1)

CV_results <- tree_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_params,
            metrics=metric_set(roc_auc))

# find best tuning params
bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

# finalize workflow and make predictions
tree_model <- rand_forest(mtry = 10, 
                          min_n = 2,
                          trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

tree_wf <- workflow() %>%
  add_recipe(recipeTargetEncoded) %>%
  add_model(tree_model) %>%
  fit(data=train)

RAND_predictions <- tree_wf %>%
  predict(new_data = test, type = "prob")


submission <- RAND_predictions %>%
  bind_cols(., test) %>%
  select(id, .pred_1) %>%
  rename(ACTION = .pred_1)

vroom_write(x=submission, file="./RANDPreds.csv", delim=",")
