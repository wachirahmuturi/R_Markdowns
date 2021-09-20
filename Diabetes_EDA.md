Exploratory Data Analysis using R
================
Wachira
9/17/2021

### Background Info on our Dataset

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

### Import Libraries

``` r
library(tidyr)
library(dplyr)
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
library(ggplot2)
```

### Load your dataset

We will be importing our dataset. I downloaded the dataset from Kaggle
at <https://www.kaggle.com/uciml/pima-indians-diabetes-database>

``` r
setwd('/home/wachirah/Downloads/Diabetes_R/')
df <- read.csv('diabetes.csv', header = TRUE, sep = ',')
str(df)
```

    ## 'data.frame':    768 obs. of  9 variables:
    ##  $ Pregnancies             : int  6 1 8 1 0 5 3 10 2 8 ...
    ##  $ Glucose                 : int  148 85 183 89 137 116 78 115 197 125 ...
    ##  $ BloodPressure           : int  72 66 64 66 40 74 50 0 70 96 ...
    ##  $ SkinThickness           : int  35 29 0 23 35 0 32 0 45 0 ...
    ##  $ Insulin                 : int  0 0 0 94 168 0 88 0 543 0 ...
    ##  $ BMI                     : num  33.6 26.6 23.3 28.1 43.1 25.6 31 35.3 30.5 0 ...
    ##  $ DiabetesPedigreeFunction: num  0.627 0.351 0.672 0.167 2.288 ...
    ##  $ Age                     : int  50 31 32 21 33 30 26 29 53 54 ...
    ##  $ Outcome                 : int  1 0 1 0 1 0 1 0 1 1 ...

``` r
head(df)
```

    ##   Pregnancies Glucose BloodPressure SkinThickness Insulin  BMI
    ## 1           6     148            72            35       0 33.6
    ## 2           1      85            66            29       0 26.6
    ## 3           8     183            64             0       0 23.3
    ## 4           1      89            66            23      94 28.1
    ## 5           0     137            40            35     168 43.1
    ## 6           5     116            74             0       0 25.6
    ##   DiabetesPedigreeFunction Age Outcome
    ## 1                    0.627  50       1
    ## 2                    0.351  31       0
    ## 3                    0.672  32       1
    ## 4                    0.167  21       0
    ## 5                    2.288  33       1
    ## 6                    0.201  30       0

`Note` The Outcome column is only made of 1s and Os. 1 interprets a
person as Diabetic while 0 interprets them as Non-diabetic

### Check for Missing Data

``` r
colSums(is.na(df))
```

    ##              Pregnancies                  Glucose            BloodPressure 
    ##                        0                        0                        0 
    ##            SkinThickness                  Insulin                      BMI 
    ##                        0                        0                        0 
    ## DiabetesPedigreeFunction                      Age                  Outcome 
    ##                        0                        0                        0

Our dataset has no ‘NA’ values. The data has already been cleaned up;
this is quite a normal characteristic with Kaggle datasets. In the case
of NA values,one approach for cleaning data is to drop rows with NA
values. Another approach is to use median approximation i.e
df*c**o**l*\[*i**s*.*n**a*(*d**f*col)\] = median(df$col, na.rm=TRUE)

### Descriptive analytics

``` r
df$Outcome <- as.factor(df$Outcome)
#head(df)
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

## Plot the Data

### Boxplots

![](/boxplot-1.png)<!-- -->![](/boxplot-2.png)<!-- -->

`Discussion`: For the boxplot of BMI against Outcome, we see the
distribution of BMI being higher in diabetics as compared to that of
non-diabetics. This finding supports the fact that obese individuals are
more prone to diabetes

For the boxplot of Glucose against Outcome, it is quite clear that the
range of glucose levels are different between diabetic and non-diabetic
persons. As is known, the level of blood glucose is primarily the main
indicator

### Scatterplots

``` r
plt3 <- ggplot(df, aes(x=Age, y=Glucose, col=Outcome)) + geom_point() + geom_smooth(method="loess")
#loess method is for local regression fitting
plt3
```

    ## `geom_smooth()` using formula 'y ~ x'

![](/scatterplot-1.png)<!-- -->

``` r
plt4 <- ggplot(df, aes(x=Age, y=BMI, col=Outcome)) + geom_point() + geom_smooth(method = 'loess')
#loess method is for local regression fitting
plt4
```

    ## `geom_smooth()` using formula 'y ~ x'

![](/scatterplot-2.png)<!-- -->

`Discussion`: Looking at the scatterplot of Glucose against Age, it is
evident that diabetic persons have higher glucose levels(as seen by the
blue dots distribution in the 1st and 2nd quadrants of the graph) and as
people grow older, the risk for diabetes increases.

Viewing the scatterplot of BMI against age, from the regression line we
see diabetic patients have a higher BMI although it appears the
regression lines are almost coming into contact. To know the viability
of the BMI variable as a diagnostic tool, it would be useful to perform
a statistical analysis

### Statistical Analysis(Hypothesis Testing)

``` r
df1 <- df %>%
  select(BMI, Outcome) %>%
  filter(Outcome == 0)
df2 <- df %>%
  select(BMI, Outcome) %>%
  filter(Outcome == 1)

t.test(df1$BMI, df2$BMI, mu=0)
```

    ## 
    ##  Welch Two Sample t-test
    ## 
    ## data:  df1$BMI and df2$BMI
    ## t = -8.6193, df = 573.47, p-value < 2.2e-16
    ## alternative hypothesis: true difference in means is not equal to 0
    ## 95 percent confidence interval:
    ##  -5.940864 -3.735811
    ## sample estimates:
    ## mean of x mean of y 
    ##  30.30420  35.14254

`Discussion`: From the t.test results, the p-value is below .05 meaning
we reject the null hypothesis(difference in means = 0).There is a
***statistically significant difference*** in means of BMI persons with
and without diabetes. This places BMI as a good indicator of diabetes
