library(tidyverse)
library(tidymodels)
library(vroom)
library(skimr)
library(GGally)
library(ggplot2)
library(glmnet)
library(stacks)
library(recipes)
library(embed)
library(kknn)
library(discrim)
library(naivebayes)
library(themis)


test <- vroom("test.csv")
train <- vroom("train.csv")
train$ACTION <- as.factor(train$ACTION)

recipeTargetEncoded <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_predictors()) %>%
  step_smote(all_outcomes(), neighbors=4) 

nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
set_mode("classification") %>%
set_engine("naivebayes") 

nb_wf <- workflow() %>%
add_recipe(recipeTargetEncoded) %>%
add_model(nb_model)

knn_tuning_grid <- grid_regular(Laplace(), smoothness(),levels = 5)

folds <- vfold_cv(train, v = 5, repeats=1)

CV_results <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=knn_tuning_grid, 
            metrics = metric_set(roc_auc)) 

bestTune <- CV_results %>%
  select_best(metric="roc_auc")

final_knn_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train)

knn_predictions <- final_knn_wf %>%
  predict(new_data = test, type = "prob")


submission <- knn_predictions %>%
  bind_cols(., test) %>%
  select(id, .pred_1) %>%
  rename(ACTION = .pred_1)

vroom_write(x=submission, file="./SMOTENaivePreds.csv", delim=",")
