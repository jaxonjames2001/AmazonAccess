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

knn_model <- nearest_neighbor(neighbors=tune()) %>%
  set_mode("classification") %>%
  set_engine("kknn") 

knn_workflow <- workflow() %>%
  add_recipe(recipeTargetEncoded) %>%
  add_model(knn_model)

knn_tuning_grid <- grid_regular(neighbors(),
                            levels = 5)

folds <- vfold_cv(train, v = 5, repeats=1)

CV_results <- knn_workflow %>%
  tune_grid(resamples=folds,
            grid=knn_tuning_grid, 
            metrics = metric_set(roc_auc)) 

bestTune <- CV_results %>%
  select_best(metric="roc_auc")

final_knn_wf <- knn_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = train)

knn_predictions <- final_knn_wf %>%
  predict(new_data = test, type = "prob")


submission <- knn_predictions %>%
  bind_cols(., test) %>%
  select(id, .pred_1) %>%
  rename(ACTION = .pred_1)

vroom_write(x=submission, file="./KNNPreds.csv", delim=",")

