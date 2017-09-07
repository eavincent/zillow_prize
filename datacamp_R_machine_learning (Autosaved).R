---
title: "Machine Learning in R"
author: "Liz Vincent"
date: "9/4/2017"
output: html_document
---
# DataCamp: Machine Learning with R Skill Track

## Course 1: Supervised learning in R: Classification

* Train a machine to learn from prior examples
* Classification: concept to be learned is a set of _categories_

### Classification tasks - K nearest neighbors
* Coordinates in a feature-space - e.g. red signs close to other red signs
* R function `knn(train, test, cl)` (k nearest neighbors) in `class` package equires training data, test data, and classification labels for training data
* k determines the size of the neighborhood
* if k=1, only the single nearest neighbor is used to classify the test data
* if k>1, the majority category for the neighbors is chosen
* In the case of a tie, winner typically chosen at random
* Larger k can be better if the nearest neighbor(s) happen to be a different classification, but bigger is not always better
* Smaller neighborhoods mean the classifier can detect subtler patterns
* Optimal k depends on complexity of the pattern to be learned as well as the influence of noisy data - larger k filters out noise better but may ignore subtle structures of the data
* Rule of thumb: set `k = sqrt(# obs in training group)`
* May be useful to determine the certainty of a prediction - can set `prob=T` in the `knn()` function and inspect the probabilities using attr(x, "prob")
* `aggregate(x~y,data,function)`
* Confusion matrix using `table(predicted,actual)`
* Computes overall accuracy using `mean(predicted == actual)`
* `knn()` and most nearest neighbors functions assume numeric data bc distance between categorical variables cannot be calculated
* In instances where a categorical variable cannot be converted to numberic, a dummy binary variable is created for each possibility except one (e.g. if the options are rectangle, diamond, and octagon there would be 2 dummy variables rectangle and diamond. Rectangles are given 1 0, diamonds are 0 1, and octagons are 0 0.)
* Dummy-coded variables can be used directly in distance functions
* For all categories to contribute equally, rescale categories such that they are all on the same scale (e.g. a 0 to 1 scale)
    
### Bayesian methods
* Joint probabilities (P(A and B)) and independent events
* Conditional probabilities (P(A | B)) and dependent events
* P(A | B) = P(A and B) / P(B)
* Naive Bayes: estiamte conditional probability of an outcome
* `naivebayes` package function `naive_bayes(prediction ~ predictor, data)` builds a model
* `predict(m, future_conditions)` predicts outcomes based on future predictions using the Naive Bayes model `m`
* Use `type="prob"` argument to get probabilities for each outcome.
* More conditionals make the computation more 
* The naive assumption is that the events are independent
* Joint probability can be calculated by multiplying individual probabilities - can calculate multiple overlaps in events by multiplying simpler overlaps
* Laplace correction: add a constant (usually 1) to each probability event so that things that have not happened yet but still _could_ happen in the future (i.e. infrequent events) are not always predicted as 0 due to the fact that they haven't happened _yet_
* Numeric properties and uncategorized text are difficult for naive bayes - use bins to get around this problem

### Binary predictions with regression
* Logistic regression produces an S-shaped curve where for any input value of x, every value of y is between 0 and 1
* `glm(y~ x1 + x2 + x3, data = data, family="binomial")` tells `R` to perform logistic regression
* Once model is built, can be used to estimate proabilities
* `type="response"` produces predicted probabilities, as opposed to the default log of odds
* To make predictions, probabilities must be converted into a response using `ifelse()` statement for where the probability falls, e.g. `pred<-ifelse(prob > 0.5, 1, 0)`
* Rare events create challenges for classification models: when one outcome is very rare, predicting the opposite can result in a very high accuracy overall, but a very low accuracy (0) for the outcome we are actually interested in
* May be necessary to sacrifice some overall accuracy in order to better target the outcome of interest
* Visualization called ROC curve provides way to better understand model's ability to distinguish between positive and negative outcomes
* ROC curve depicts relationship of % positive outcomes vs. % of other outcomes
* Diagonal line is the baseline performance for a very poor model
* To quantify the performance of a model use AUC: Area Under the ROC Curve
* Baseline model that is no better than random chance has AUC = 0.50
* Perfect model has AUC = 1.00
* Curves of varying shapes can have the same AUC value, so it is important to not just look at the AUC value but also at the shape of the ROC curve

### Automatic feature recognition
#### Stepwise regression
* Backward deletion begins with a model containing all of the predictors and removes each predictor one at a time
* If removing a predictor has minimal impact on the model, it is removed
* Predictors with the least impact on the model are removed first, and step by step predictors that have minimal effects on the model are removed
* Forward selection is the same idea but works the other way: starts with 0 predictors and adds predictors step-by-step
* It is possible that backward and forward selection will result in different models

### Decision trees
* Divide-and-conquer: algorithm looks for an initial split that creates the most homogenous group
* `rpart()` makes a tree, `predict()` can again be used to predict a result using the `type="class"` parameter
* Divide and conquer cannot consider combinations of two or more features - decision tree will always create "axis-parallel splits"
* Weakness: decision tree may then be over-complex for data
* Trees can become overly complex very easily
* Overly large or overly complex trees may over-fit the data and therefore model the noise - it focuses on subtle patterns that may not apply more generally
* When machine learning overfits the data we must be careful not to overestimate the accuracy of the model - it may perfectly model the training data but not do as well on future predictions
* Because of over-fitting, it is important to simulate future data with a test data set that the algorithm cannot use when constructing the tree
* If the tree performs much better on the training set than on the test set, this suggests the tree has been over-fitted
* Pre- and post-pruning can be done with `rpart` package
#### Pre-pruning
* Stop divide and conquer once the tree reaches a predefined size
* Stop the tree from growing a branch with less than n observations
* A tree stopped too early may miss sublte patterns
* Pre-pruning with `rpart` can be done with `rpart.control(maxdepth, minsplit)` as the argument to `control` when running the `rpart()` function
#### Post-pruning
* Nodes and branches with only minor impact on the tree's classification accuracy can be pruned
* Post-pruning with `rpart` is done after the call to `rpart()` with `prune(model, cp=cutpoint)`

### Decision tree forest
* A number of classification trees can be combined into a collection known as a decision tree forest
* Among the most powerful machine learning classifiers, but easy to use
* Not a single, overly-complex tree, but many simpler trees that together reflect the complexity of the data
* Growing diverse trees requires growing conditions to be varied from tree-to-tree, done by allocating each tree a random subset of data for training
* "Random forest" refers to a specific growing algorithm in which features and examples may differ from tree to tree
* Ensemble methods of machine learning, such as random forest, are based on the principle that weaker learners become strong with teamwork
* In random forest, each tree is asked to make a prediction and the group's overall prediction is determined by a majority vote
* R package `randomForest` applies the random forest algorithm
* `m <- randomForest(y ~ x1 + x2 + x3, data, ntree, mtry)` where `ntree` is the number of trees in the forest and `mtry` is the number of predictors (features) per tree - default is `sqrt(p)` where `p` is the total number of predictors (usually okay to leave as-is)
* Use `predict()` again
* Even with a large number of trees the model usually runs fairly quickly because each tree only uses a small portion of the dataset
* Due to the random nature of the forest, results may very each time the forest is created

## Course 2: Supervised Learning in R: Regression

* Regression: predicts a _numeric_ outcome (expected value) from a set of inputs
* Linear regression model is a type of supervised learning