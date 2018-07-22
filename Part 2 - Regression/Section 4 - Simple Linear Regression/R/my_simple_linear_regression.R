# 1. Data Preprocessing

# A| Importing Data
dataset = read.csv('Salary_Data.csv')

# B| Splitting Data
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# 2. Simple Linear Regression

# A| Fitting Model into Training Set
regressor = lm(formula = Salary ~ YearsExperience, data = training_set)

# B| Predicting Results
y_pred = predict(regressor, newdata = test_set)

# C| Visualizing Models
# install.packages('ggplot2')
library(ggplot2)

# Training Set
ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             color='red') +
  geom_line(aes(x= training_set$YearsExperience, y= predict(regressor, newdata = training_set)),
            color='blue') +
  ggtitle('Salary vs Experience (Training Set)') +
  xlab('Years of Experience') +
  ylab('Salary')


# Test Set
ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             color='green') +
  geom_line(aes(x= training_set$YearsExperience, y= predict(regressor, newdata = training_set)),
            color='blue') +
  ggtitle('Salary vs Experience (Training Set)') +
  xlab('Years of Experience') +
  ylab('Salary')