library(tidyverse)
library(tidymodels)
library(vroom)
library(glmnet)
library(recipes)
library(embed)
library(lme4)

test <- vroom("test.csv")
train <- vroom("train.csv")
train$ACTION <- as.factor(train$ACTION)

recipeTargetEncoded <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_predictors(), fn = factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_nominal_predictors())

pen_model <- logistic_reg(mixture=tune(), 
                          penalty=tune()) %>%
  set_engine("glmnet")

penalized_workflow <- workflow() %>%
  add_recipe(recipeTargetEncoded) %>%
  add_model(pen_model)

tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5)

folds <- vfold_cv(train, v = 5, repeats=1)

CV_results <- penalized_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid, 
            metrics = metric_set(roc_auc)) 

bestTune <- CV_results %>%
  select_best(metric="roc_auc")

print(bestTune) # to get best hyperparameters

final_wf <- penalized_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = train)

penalized_predictions <- final_wf %>%
  predict(new_data = test, type = "prob")


submission <- penalized_predictions %>%
  bind_cols(., test) %>%
  select(id, .pred_1) %>%
  rename(ACTION = .pred_1)

vroom_write(x=submission, file="./PenLogisticPreds.csv", delim=",")


