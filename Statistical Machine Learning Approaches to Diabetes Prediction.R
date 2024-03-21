# Data import
diabetes_data <- read.table("C:\\Users\\54088\\OneDrive\\桌面\\SW\\S5 2023\\STAT 4052\\HW\\Diabetes4.txt")

summary(diabetes_data)
# NA only occurs in hypertension, HbA1c_level, and Diabetes.
# For continuous values impute with mean of that variable. 
# For categorical values exclude the observations with missingness from the analysis.


# Stage1 - dealing with missing data
# stage 1 
stage1_diabetes = diabetes_data

# replace the missing value in continuous variables with the mean of that variable
stage1_diabetes$HbA1c_level[is.na(stage1_diabetes$HbA1c_level)] = mean(diabetes_data$HbA1c_level, na.rm = TRUE)

# remove the observation with missing value in categorical variables 
stage1_diabetes <- stage1_diabetes[!is.na(stage1_diabetes$hypertension), ]
stage1_diabetes <- stage1_diabetes[!is.na(stage1_diabetes$diabetes), ]

stage1_diabetes$gender <- as.factor(stage1_diabetes$gender)
stage1_diabetes$smoking_history <- as.factor(stage1_diabetes$smoking_history)

# one-hot code the training and validation set
library(mltools)
library(data.table)
stage1_data <- one_hot(as.data.table(stage1_diabetes), dropUnusedLevels = TRUE)

# factor some variables 
stage1_data$hypertension <- as.factor(stage1_data$hypertension)
stage1_data$heart_disease <- as.factor(stage1_data$heart_disease)
stage1_data$diabetes <-as.factor(stage1_data$diabetes)
colnames(stage1_data)[11] <- "smoking_history_no_info"
colnames(stage1_data)[12] <- "smoking_history_not_current"

#since the gender:Other has only one observation left in the dataset after omit the NAs which is in test set in this case. Therefore, we have to exclude this column.   
stage1_data <- subset(stage1_data, select = -c(3))

summary(stage1_data)


# Stage1 - logistic regression
library(leaps)

set.seed(1234)
train_indices = sample(1:nrow(stage1_data), nrow(stage1_data)*0.8)
train_stage <- stage1_data[train_indices, ]
test_stage <- stage1_data[-train_indices, ]

# Lab5
# Select which model is the best model 
# Use stepwise and AIC to determine which model is the optimal model
stepwise_model <- step(glm(diabetes ~ ., data = stage1_data, family = "binomial"), direction = "both", trace = FALSE)
stepwise_model

# Fit the optimal model and predict
model_logistic <- glm(diabetes ~ age + hypertension + heart_disease + smoking_history_former + smoking_history_no_info + bmi + HbA1c_level + blood_glucose_level, family = "binomial", data = train_stage)
summary(model_logistic)

pre_logistic = predict(model_logistic, newdata = test_stage, type = "response")
pre_logistic_class = ifelse(pre_logistic > 0.5, 1, 0)

table(test_stage$diabetes, pre_logistic_class)
err_logistic_stage1 = mean(test_stage$diabetes!=pre_logistic_class)
err_logistic_stage1

# Diagnostic plot
plot(model_logistic, which = 5)
# There is no influencial points

# Deviance & Pearson chi-square test
pchisq(model_logistic$deviance, 3343, lower.tail = FALSE)

Pearson = sum(residuals(model_logistic, type = "pearson")^2)
pchisq(Pearson, 3343, lower.tail = FALSE)


# Stage1 - LDA, QDA, KNN
# Sensibly choose either KNN, LDA or QDA to predict diabetes
# Has to explain why LDA is best among these three methods 
# Lab9
library(caret)
library(MASS)
# try LDA
model_lda = lda(diabetes ~ age + hypertension + heart_disease + smoking_history_former + smoking_history_no_info + bmi + HbA1c_level + blood_glucose_level, train_stage)

pre_lda = predict(model_lda, newdata = test_stage)

# Confusion matrix
table(test_stage$diabetes, pre_lda$class)

# Misclassificantion error
err_lda_stage1 <- mean(test_stage$diabetes!=pre_lda$class)
err_lda_stage1

# try QDA
model_qda = qda(diabetes ~ age + hypertension + heart_disease + smoking_history_former + smoking_history_no_info + bmi + HbA1c_level + blood_glucose_level, train_stage)

pre_qda = predict(model_qda, newdata = test_stage)

# Confusion matrix
table(test_stage$diabetes, pre_qda$class)

# Misclassification error
err_qda_stage1 <- mean(test_stage$diabetes!=pre_qda$class)
err_qda_stage1
# LDA had lower validation error and hence is preferred over QDA


# KNN unless we find something useful for this one, otherwise we said since LDA has perform better than QDA, meaning that in this case we can say that it has better fit for linear model like LDA. 
library(class)
# try k = 3
train_predictors <- cbind(train_stage$age, train_stage$hypertension, train_stage$heart_disease, 
                          train_stage$smoking_history_former, train_stage$bmi, 
                          train_stage$HbA1c_level, train_stage$blood_glucose_level)

test_predictors <- cbind(test_stage$age, test_stage$hypertension, test_stage$heart_disease, 
                         test_stage$smoking_history_former, test_stage$bmi, 
                         test_stage$HbA1c_level, test_stage$blood_glucose_level)

predictors_train <- as.matrix(train_stage[, c("age", "hypertension", "heart_disease", "smoking_history_former", "smoking_history_no_info", "bmi", "HbA1c_level", "blood_glucose_level")])
predictors_test <- as.matrix(test_stage[, c("age", "hypertension", "heart_disease", "smoking_history_former", "smoking_history_no_info", "bmi", "HbA1c_level", "blood_glucose_level")])


# try k = 3
pre3 = knn(train = train_predictors, test = test_predictors, cl = train_stage$diabetes, k = 3)
table(test_stage$diabetes, pre3)
val_err4 = mean(test_stage$diabetes != pre3)
val_err4

# try k = 5
pre5 = knn(train = predictors_train, test = predictors_test, cl = train_stage$diabetes, k = 5)
table(test_stage$diabetes, pre5)
val_err5 = mean(test_stage$diabetes != pre5)
val_err5

# try k = 11
pre11 = knn(train = predictors_train, test = predictors_test, cl = train_stage$diabetes, k = 11)
table(test_stage$diabetes, pre11)
val_err6 = mean(test_stage$diabetes != pre11)
val_err6

# Do ROC curve for LDA, QDA, and KNN. Compare the error rate. and we can see that lda has the minimum error rate. 
data.frame(val_err4, val_err5, val_err6)


# Stage1 - Random Forest and Bagging and Boosting
# Use either random forest, bagging or boosting to predict diabetes
# Lab10
# try bagging - classification
# Random forest -> m = sqrt(p) = 2 
set.seed(4052)
library(randomForest)
model_rf = randomForest(diabetes ~ age + hypertension + heart_disease + smoking_history_former + smoking_history_no_info + bmi + HbA1c_level + blood_glucose_level, data = stage1_data, mtry = 2, importance = TRUE)
model_rf

# Two importance variables are HbA1c_level and blood_glucose_level
# error rate = 0.0316, which is less than LDA 

pre_rf <- predict(model_rf, test_stage)
table(pre_rf, test_stage$diabetes)
mean(pre_rf!=test_stage$diabetes)


model_bagging = randomForest(diabetes ~ age + hypertension + heart_disease + smoking_history_former + smoking_history_no_info + bmi + HbA1c_level + blood_glucose_level, data = stage1_data, mtry = 7, importance = TRUE)
model_bagging


# Random forest for only training set
model_rf_train = randomForest(diabetes ~ age + hypertension + heart_disease + smoking_history_former + smoking_history_no_info + bmi + HbA1c_level + blood_glucose_level, data = train_stage, mtry = 2, importance = TRUE)
model_rf_train

# Two importance variables are HbA1c_level and blood_glucose_level
# error rate = 0.0316, which is less than LDA 

pre_rf_train <- predict(model_rf_train, test_stage)
table(pre_rf_train, test_stage$diabetes)
err_rf_stage1 <- mean(pre_rf_train!=test_stage$diabetes)
err_rf_stage1


# Add RF ROC curve and AUC(LDA and QDA)
library(ROCR)


pre_rf_prob <- predict(model_rf_train, newdata = test_stage, type="prob")

rf_pre = pre_rf_prob[,2]
lda_pre = pre_lda$posterior[,2]
qda_pre = pre_qda$posterior[,2]


pred_log = prediction(pre_logistic, test_stage$diabetes)
perf_log = performance(pred_log, "tpr", "fpr")
pred_rf = prediction(rf_pre, test_stage$diabetes)
perf_rf = performance(pred_rf, "tpr", "fpr")
pred_lda = prediction(lda_pre, test_stage$diabetes)
perf_lda = performance(pred_lda, "tpr", "fpr")
pred_qda = prediction(qda_pre, test_stage$diabetes)
perf_qda = performance(pred_qda, "tpr", "fpr")

par(mfrow = c(2, 2))
plot(perf_log, colorize = TRUE, main = "Logistic test")
plot(perf_lda, colorize = TRUE, main = "LDA test")
plot(perf_qda, colorize = TRUE, main = "QDA test")
plot(perf_rf, colorize = TRUE, main = "RF test")

AUC_log = performance(pred_log, "auc")@y.values[[1]]
AUC_rf = performance(pred_rf, "auc")@y.values[[1]]
AUC_lda = performance(pred_lda, "auc")@y.values[[1]]
AUC_qda = performance(pred_qda, "auc")@y.values[[1]]

data.frame(model = c("Logistic", "LDA", "QDA", "RF"), AUC = c(AUC_log, AUC_lda, AUC_qda, AUC_rf))

# Stage2 - Iterative Regression
# iterative regression
stage2_diabetes = diabetes_data

# HbA1c
stage2_diabetes$HbA1c_level[is.na(stage2_diabetes$HbA1c_level)] = mean(diabetes_data$HbA1c_level, na.rm = TRUE)

# hypertension 
stage2_diabetes$hypertension[is.na(stage2_diabetes$hypertension)] = 0

# diabetes
stage2_diabetes$diabetes[is.na(stage2_diabetes$diabetes)] = 0 

# Iterative regression
n_iter = 10
for(i in 1:n_iter){
  # impute HbA1c(cont)
  m_HbA1c = lm(HbA1c_level ~ ., stage2_diabetes, subset=!is.na(diabetes_data$HbA1c_level))
  pre_HbA1c = predict(m_HbA1c, stage2_diabetes[is.na(diabetes_data$HbA1c_level),])
  stage2_diabetes$HbA1c_level[is.na(diabetes_data$HbA1c_level)] = pre_HbA1c
  
  # impute hypertension
  library(nnet)
# impute hypertension (binary)
  m_hypertension = glm(hypertension ~ ., family = binomial, data = stage2_diabetes, subset = !is.na(diabetes_data$hypertension))
  pre_hypertension = predict(m_hypertension, stage2_diabetes[is.na(diabetes_data$hypertension), ], type = "response")
  stage2_diabetes$hypertension[is.na(diabetes_data$hypertension)] = ifelse(pre_hypertension > 0.5, 1, 0)

  # impute diabetes
  m_diabetes = glm(diabetes ~., family = binomial, data = stage2_diabetes, subset = !is.na(diabetes_data$hypertension))
  pre_diabetes = predict(m_diabetes, stage2_diabetes[is.na(diabetes_data$diabetes),], type = "response")
  stage2_diabetes$diabetes[is.na(diabetes_data$diabetes)] = ifelse(pre_diabetes > 0.5, 1, 0)
}


# Stage2 - Logistic Regression 
# stage2
stage2_diabetes$gender <- as.factor(stage2_diabetes$gender)
stage2_diabetes$smoking_history <- as.factor(stage2_diabetes$smoking_history)

# one-hot-encoding
stage2_data <- one_hot(as.data.table(stage2_diabetes), dropUnusedLevels = TRUE)

# factor some variables 
stage2_data$hypertension <- as.factor(stage2_data$hypertension)
stage2_data$heart_disease <- as.factor(stage2_data$heart_disease)
stage2_data$diabetes <-as.factor(stage2_data$diabetes)
colnames(stage2_data)[11] <- "smoking_history_no_info"
colnames(stage2_data)[12] <- "smoking_history_not_current"

#since the gender:Other has only one observation left in the dataset after omit the NAs which is in test set in this case. Therefore, we have to exclude this column.   
stage2_data <- subset(stage2_data, select = -c(3))

set.seed(1234)
train_indices = sample(1:nrow(stage2_data), nrow(stage2_data)*0.8)
train_stage2 <- stage2_data[train_indices, ]
test_stage2 <- stage2_data[-train_indices, ]


# Lab5
# Select which model is the best model 
# Use stepwise and AIC to determine which model is the optimal model
stepwise_model2 <- step(glm(diabetes ~ ., data = stage2_data, family = "binomial"), direction = "both", trace = FALSE)
stepwise_model2

# Fit the optimal model and predict
model_logistic2 <- glm(diabetes ~ age + hypertension + heart_disease + smoking_history_former + smoking_history_no_info + bmi + HbA1c_level + blood_glucose_level, family = "binomial", data = train_stage2)
summary(model_logistic2)

pre_logistic2 = predict(model_logistic2, newdata = test_stage2)
pre_logistic_class2 <- ifelse(pre_logistic2 > 0.5, 1, 0)
table(test_stage2$diabetes, pre_logistic_class2)
err_logistic_stage2 = mean(test_stage2$diabetes!=pre_logistic_class2)
err_logistic_stage2

# Diagnostic plot
plot(model_logistic2, which = 5) 

# Deviance & Pearson chi-square test
pchisq(model_logistic2$deviance, 3991, lower.tail = FALSE)

Pearson2 = sum(residuals(model_logistic2, type = "pearson")^2)
pchisq(Pearson2, 3991, lower.tail = FALSE)



# Stage2 - LDA, QDA, KNN
# Sensibly choose either KNN, LDA or QDA to predict diabetes
# Has to explain why LDA is best among these three methods 
# Lab9
str(train_stage2)
library(caret)
library(MASS)
# try LDA
model_lda2 = lda(diabetes ~ age + hypertension + heart_disease + smoking_history_former + smoking_history_no_info + bmi + HbA1c_level + blood_glucose_level, train_stage2)

pre_lda2 = predict(model_lda2, newdata = test_stage2)

# Confusion matrix
table(test_stage2$diabetes, pre_lda2$class)

# Misclassificantion error
err_lda_stage2 <- mean(test_stage2$diabetes!=pre_lda2$class)

# try QDA
model_qda2 = qda(diabetes ~ age + hypertension + heart_disease + smoking_history_former + smoking_history_no_info + bmi + HbA1c_level + blood_glucose_level, train_stage2)

pre_qda2 = predict(model_qda2, newdata = test_stage2)

# Confusion matrix
table(test_stage2$diabetes, pre_qda2$class)

# Misclassification error
err_qda2 <- mean(test_stage2$diabetes!=pre_qda2$class)
# LDA had lower validation error and hence is preferred over QDA

# KNN unless we find something useful for this one, otherwise we said since LDA has perform better than QDA, meaning that in this case we can say that it has better fit for linear model like LDA. 
library(class)
# Correcting the KNN syntax to use cbind for combining predictors
train_predictors <- cbind(train_stage2$age, train_stage2$hypertension, train_stage2$heart_disease, 
                          train_stage2$smoking_history_former, train_stage2$bmi, 
                          train_stage2$HbA1c_level, train_stage2$blood_glucose_level)

test_predictors <- cbind(test_stage2$age, test_stage2$hypertension, test_stage2$heart_disease, 
                         test_stage2$smoking_history_former, test_stage2$bmi, 
                         test_stage2$HbA1c_level, test_stage2$blood_glucose_level)

# Using the corrected matrix in the KNN function
pre9 = knn(train = train_predictors, test = test_predictors, cl = train_stage2$diabetes, k = 3)
table(test_stage2$diabetes, pre9)
val_err7 = mean(test_stage2$diabetes != pre9)
val_err7

pre10 = knn(train = train_predictors, test = test_predictors, cl = train_stage2$diabetes, k = 5)
table(test_stage2$diabetes, pre10)
val_err8 = mean(test_stage2$diabetes != pre10)
val_err8

pre11 = knn(train = train_predictors, test = test_predictors, cl = train_stage2$diabetes, k = 10)
table(test_stage2$diabetes, pre11)
err_knn_stage2 = mean(test_stage2$diabetes != pre11)
err_knn_stage2


# Do ROC curve for LDA, QDA, and KNN. Compare the error rate. and we can see that lda has the minimum error rate.


# Stage2 - RF and Bagging and Boosting

# Use either random forest, bagging or boosting to predict diabetes
# Lab10
# try bagging - classification
set.seed(4052)
library(randomForest)
model_rf2 = randomForest(diabetes ~ age + hypertension + heart_disease + smoking_history_former + smoking_history_no_info + bmi + HbA1c_level + blood_glucose_level, data = stage2_data, mtry = 2, importance = TRUE)
model_rf2

pre_rf2 <- predict(model_rf2, test_stage2)
table(pre_rf2, test_stage2$diabetes)
mean(pre_rf2!=test_stage2$diabetes)


# Two importance variables are HbA1c_level and blood_glucose_level
# error rate = 0.0316, which is less than LDA 
model_bagging2 = randomForest(diabetes ~ age + hypertension + heart_disease + smoking_history_former + smoking_history_no_info + bmi + HbA1c_level + blood_glucose_level, data = stage2_data, mtry = 7, importance = TRUE)
model_bagging2

pre_bagging2 <- predict(model_bagging2, test_stage2)
table(pre_bagging2, test_stage2$diabetes)
mean(pre_bagging2!=test_stage2$diabetes)


# Random forest for only training set
model_rf_train2 = randomForest(diabetes ~ age + hypertension + heart_disease + smoking_history_former + smoking_history_no_info + bmi + HbA1c_level + blood_glucose_level, data = train_stage2, mtry = 2, importance = TRUE)
model_rf_train2

# Two importance variables are HbA1c_level and blood_glucose_level
# error rate = 0.0316, which is less than LDA 

pre_rf_train2 <- predict(model_rf_train2, test_stage2)
table(pre_rf_train2, test_stage2$diabetes)
err_rf_stage2 <- mean(pre_rf_train2!=test_stage2$diabetes)

varImpPlot(model_rf2)

# Add RF ROC curve and AUC(LDA and QDA)
library(ROCR)

pre_rf_prob2 <- predict(model_rf_train2, newdata = test_stage2, type="prob")

rf_pre2 = pre_rf_prob2[,2]
lda_pre2 = pre_lda2$posterior[,2]
qda_pre2 = pre_qda2$posterior[,2]

pred_log2 = prediction(pre_logistic2, test_stage2$diabetes)
perf_log2 = performance(pred_log2, "tpr", "fpr")
pred_lda2 = prediction(lda_pre2, test_stage2$diabetes)
perf_lda2 = performance(pred_lda2, "tpr", "fpr")
pred_qda2 = prediction(qda_pre2, test_stage2$diabetes)
perf_qda2 = performance(pred_qda2, "tpr", "fpr")
pred_rf2 = prediction(rf_pre2, test_stage2$diabetes)
perf_rf2 = performance(pred_rf2, "tpr", "fpr")

par(mfrow = c(2, 2))
plot(perf_log2, colorize = TRUE, main = "Logistic test")
plot(perf_lda2, colorize = TRUE, main = "LDA test")
plot(perf_qda2, colorize = TRUE, main = "QDA test")
plot(perf_rf2, colorize = TRUE, main = "RF test")

AUC_log2 = performance(pred_log2, "auc")@y.values[[1]]
AUC_rf2 = performance(pred_rf2, "auc")@y.values[[1]]
AUC_lda2 = performance(pred_lda2, "auc")@y.values[[1]]
AUC_qda2 = performance(pred_qda2, "auc")@y.values[[1]]

data.frame(model = c("Logistic", "LDA", "QDA", "RF"), AUC = c(AUC_log2, AUC_lda2, AUC_qda2, AUC_rf2))

# Comparison between stage1 and stage2
data.frame(
    Names = c("Stage1: Test Error rate for logistic regression model", "Stage1: Test Error rate for LDA", "Stage1: Test Error rate for RF", "Stage2: Test Error rate for logistic regression model", "Stage2: Test Error rate for LDA", "Stage2: Test Error rate for RF"),
    Values = c(err_logistic_stage1, err_lda_stage1, err_rf_stage1, err_logistic_stage2, err_lda_stage2, err_rf_stage2))
