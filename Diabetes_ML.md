Machine Learning in R
================
Wachira
9/30/2021

### About our dataset

This dataset is originally from the National Institute of Diabetes and
Digestive and Kidney Diseases. The objective of the dataset is to
diagnostically predict whether or not a patient has diabetes, based on
certain diagnostic measurements included in the dataset.Several
constraints were placed on the selection of these instances from a
larger database. In particular, all patients here are females at least
21 years old of Pima Indian heritage.

The datasets consists of several medical predictor variables and one
target variable, Outcome. Predictor variables includes the number of
pregnancies the patient has had, their BMI, insulin level, age, and so
on. We will load this data as we proceed

### Note

This is a continuation of the Diabetes work from Part 1(EDA). In part 2,
we will be trying to generate machine learning models using the same
data to try predict the accuracy of the models.

### Install & Import Libraries

``` r
#install.packages('caret')  #takes quite some time to install
#install.packages('kernlab')
library('tidyr')
library('dplyr')
```

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
library('ggplot2')
library('caret')
```

    ## Loading required package: lattice

``` r
library('kernlab')
```

    ## 
    ## Attaching package: 'kernlab'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     alpha

### Load our dataset

``` r
setwd('/home/wachirah/Downloads/Diabetes_R/') #set your own directory
df <- read.csv('diabetes.csv', sep = ',', header = TRUE)
```

### Processing of the data

``` r
#summary(df)
#Convert the outcome variable data type to factor type
df$Outcome <- as.factor(df$Outcome)
summary(df)
```

    ##   Pregnancies        Glucose      BloodPressure    SkinThickness  
    ##  Min.   : 0.000   Min.   :  0.0   Min.   :  0.00   Min.   : 0.00  
    ##  1st Qu.: 1.000   1st Qu.: 99.0   1st Qu.: 62.00   1st Qu.: 0.00  
    ##  Median : 3.000   Median :117.0   Median : 72.00   Median :23.00  
    ##  Mean   : 3.845   Mean   :120.9   Mean   : 69.11   Mean   :20.54  
    ##  3rd Qu.: 6.000   3rd Qu.:140.2   3rd Qu.: 80.00   3rd Qu.:32.00  
    ##  Max.   :17.000   Max.   :199.0   Max.   :122.00   Max.   :99.00  
    ##     Insulin           BMI        DiabetesPedigreeFunction      Age       
    ##  Min.   :  0.0   Min.   : 0.00   Min.   :0.0780           Min.   :21.00  
    ##  1st Qu.:  0.0   1st Qu.:27.30   1st Qu.:0.2437           1st Qu.:24.00  
    ##  Median : 30.5   Median :32.00   Median :0.3725           Median :29.00  
    ##  Mean   : 79.8   Mean   :31.99   Mean   :0.4719           Mean   :33.24  
    ##  3rd Qu.:127.2   3rd Qu.:36.60   3rd Qu.:0.6262           3rd Qu.:41.00  
    ##  Max.   :846.0   Max.   :67.10   Max.   :2.4200           Max.   :81.00  
    ##  Outcome
    ##  0:500  
    ##  1:268  
    ##         
    ##         
    ##         
    ## 

## Machine Learning

For this part 2, we will be using CARET(Classification And Regression
Training) package for our machine learning

But 1st we normalize our data.

``` r
#library(caret) #library is already loaded
preprocess_range_modeltr <- preProcess(df, method = 'range')
trainData <- predict(preprocess_range_modeltr, newdata=df)
levels(trainData$Outcome) <- c("class0","class1") #convert your binary into character for caret package

head(trainData)
```

    ##   Pregnancies   Glucose BloodPressure SkinThickness   Insulin       BMI
    ## 1  0.35294118 0.7437186     0.5901639     0.3535354 0.0000000 0.5007452
    ## 2  0.05882353 0.4271357     0.5409836     0.2929293 0.0000000 0.3964232
    ## 3  0.47058824 0.9195980     0.5245902     0.0000000 0.0000000 0.3472429
    ## 4  0.05882353 0.4472362     0.5409836     0.2323232 0.1111111 0.4187779
    ## 5  0.00000000 0.6884422     0.3278689     0.3535354 0.1985816 0.6423249
    ## 6  0.29411765 0.5829146     0.6065574     0.0000000 0.0000000 0.3815201
    ##   DiabetesPedigreeFunction       Age Outcome
    ## 1               0.23441503 0.4833333  class1
    ## 2               0.11656704 0.1666667  class0
    ## 3               0.25362938 0.1833333  class1
    ## 4               0.03800171 0.0000000  class0
    ## 5               0.94363792 0.2000000  class1
    ## 6               0.05251921 0.1500000  class0

``` r
summary(trainData)
```

    ##   Pregnancies         Glucose       BloodPressure    SkinThickness   
    ##  Min.   :0.00000   Min.   :0.0000   Min.   :0.0000   Min.   :0.0000  
    ##  1st Qu.:0.05882   1st Qu.:0.4975   1st Qu.:0.5082   1st Qu.:0.0000  
    ##  Median :0.17647   Median :0.5879   Median :0.5902   Median :0.2323  
    ##  Mean   :0.22618   Mean   :0.6075   Mean   :0.5664   Mean   :0.2074  
    ##  3rd Qu.:0.35294   3rd Qu.:0.7048   3rd Qu.:0.6557   3rd Qu.:0.3232  
    ##  Max.   :1.00000   Max.   :1.0000   Max.   :1.0000   Max.   :1.0000  
    ##     Insulin             BMI         DiabetesPedigreeFunction      Age        
    ##  Min.   :0.00000   Min.   :0.0000   Min.   :0.00000          Min.   :0.0000  
    ##  1st Qu.:0.00000   1st Qu.:0.4069   1st Qu.:0.07077          1st Qu.:0.0500  
    ##  Median :0.03605   Median :0.4769   Median :0.12575          Median :0.1333  
    ##  Mean   :0.09433   Mean   :0.4768   Mean   :0.16818          Mean   :0.2040  
    ##  3rd Qu.:0.15041   3rd Qu.:0.5455   3rd Qu.:0.23409          3rd Qu.:0.3333  
    ##  Max.   :1.00000   Max.   :1.0000   Max.   :1.00000          Max.   :1.0000  
    ##    Outcome   
    ##  class0:500  
    ##  class1:268  
    ##              
    ##              
    ##              
    ## 

### Initialize a fit control for our models

The trainControl function allows us to subject multiple models to the
same parameters

``` r
fitControl <- trainControl(method = 'cv', #k-cross fold validation
                           number = 5, #the number of folds
                           savePredictions = 'final', #saves the optimal tuning parameter
                           classProbs = T, #returns class probabilities
                           summaryFunction = twoClassSummary) #generates results as a summary
```

### Training the model

``` r
model1 <- train(Outcome ~ ., data=trainData, method='knn', tuneLength=4, trControl=fitControl) #KNN model
```

    ## Warning in train.default(x, y, weights = w, ...): The metric "Accuracy" was not
    ## in the result set. ROC will be used instead.

``` r
model2 <- train(Outcome ~ ., data=trainData, method='svmRadial', tuneLength=4, trControl=fitControl) #SVM
```

    ## Warning in train.default(x, y, weights = w, ...): The metric "Accuracy" was not
    ## in the result set. ROC will be used instead.

``` r
model3 <- train(Outcome ~ ., data=trainData, method='rpart', tuneLength=4, trControl=fitControl) #Random forest
```

    ## Warning in train.default(x, y, weights = w, ...): The metric "Accuracy" was not
    ## in the result set. ROC will be used instead.

### Compare model performances

For model comparison, we use the resamples function

``` r
models_compare <- resamples(list(KNN=model1, SVM=model2, RandomForest=model3))

##summary of the models
summary(models_compare)
```

    ## 
    ## Call:
    ## summary.resamples(object = models_compare)
    ## 
    ## Models: KNN, SVM, RandomForest 
    ## Number of resamples: 5 
    ## 
    ## ROC 
    ##                   Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
    ## KNN          0.7539815 0.7689815 0.7935849 0.7832058 0.7969811 0.8025000    0
    ## SVM          0.7762963 0.7903774 0.8448148 0.8247023 0.8527778 0.8592453    0
    ## RandomForest 0.7216038 0.7498148 0.7627358 0.7628309 0.7816667 0.7983333    0
    ## 
    ## Sens 
    ##              Min. 1st Qu. Median  Mean 3rd Qu. Max. NA's
    ## KNN          0.82    0.83   0.83 0.852    0.89 0.89    0
    ## SVM          0.84    0.86   0.87 0.868    0.87 0.90    0
    ## RandomForest 0.76    0.77   0.86 0.848    0.92 0.93    0
    ## 
    ## Spec 
    ##                   Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
    ## KNN          0.4444444 0.4905660 0.5000000 0.5188679 0.5555556 0.6037736    0
    ## SVM          0.4528302 0.5185185 0.5740741 0.5484277 0.5740741 0.6226415    0
    ## RandomForest 0.4444444 0.5555556 0.5740741 0.5676450 0.5849057 0.6792453    0

### Plotting the metrics scores for the model performances

``` r
scales <- list(x=list(relation='free'), y=list(relation='free'))
bwplot(models_compare, scales=scales)
```

![](Diabetes_ML_files/figure-gfm/plot-1.png)<!-- -->

## Discussion

ROC(Receiver Operating Characteristic) is a curve of probability and AUC
represents the degree of separability.AUC is the area under the
curve.When AUC is used under an ROC , it is one of the most important
evaluation metrics for checking any classification model’s performance
at various threshold settings. The AUROC(Area Under Receiver Operating
Characteristic) curve tells us how much the model is capable of
distinguishing between classes

An excellent model has AUC near to 1. A poor model has an AUC near 0
which in actual sense means it is reciprocating the result i.e. the
model is predicting a negative class as a positive class and vice-versa.
When the AUC is 0.5, it means the model has no class separation capacity
whatsoever

ROC curves are appropriate when observations are balanced between each
class while precision-recall curves are appropriate for imbalanced
datasets,both in binary classification predictive modelling problems

*Obtained from
<https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/>
&
<https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5>*
