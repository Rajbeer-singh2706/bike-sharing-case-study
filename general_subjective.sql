-- ###General Subjective Questions 

-- #1. Explain the linear regression algorithm in detail.     (4 marks) 
-- ANSWER 

Linear Regression is a simple yet powerful algorithm used to model the relationship between a dependent variable (target) and one or 
more independent variables (predictors). The goal of linear regression is to find a linear relationship between the input variables 
and the output. Lets break down the concept and details of Linear Regression:

Linear regression performs the task to predict a dependent variable value (y) based on a given independent variable (x). 
So, this regression technique finds out a linear relationship between x (input) and y (output). Hence, the name is Linear Regression.
Linear Regression may further divided into
   1.  Simple Linear Regression/ Univariate Linear regression
   2.   Multivariate Linear Regression

-- #2. Explain the Anscombe’s quartet in detail.     (3 marks) 
-- ANSWER 
Anscombes Quartet is a group of four datasets that have nearly identical summary statistics but exhibit strikingly different 
distributions and visual patterns when plotted. It was created by statistician Francis Anscombe in 1973 to demonstrate the 
importance of visualizing data before analyzing it and relying solely on summary statistics.

### Key Concepts Illustrated by Anscombes Quartet:

1. Importance of Data Visualization: Anscombe’s Quartet shows that similar summary statistics (like mean, variance, correlation) 
can hide significant differences in data patterns. This emphasizes that relying solely on statistics can be misleading, and 
visualizations like scatter plots reveal important insights.
2. Limitations of Summary Statistics: The quartet highlights how datasets can have the same statistical measures 
(e.g., mean, standard deviation, correlation) but differ in structure. This demonstrates that statistics alone may not 
capture the underlying patterns in the data.

### Lessons from Anscombes Quartet:

1. Visual Analysis is Essential: Even though summary statistics are the same, visualizations like scatter plots reveal the 
true nature of the relationships between variables.
   
2. Impact of Outliers: Outliers can heavily influence regression models and summary statistics. Detecting and handling 
them properly is crucial.

3. Non-Linearity: Regression lines and correlation measures assume linearity, but real-world data often follow non-linear 
patterns, which can only be identified through plotting.

--3. What is Pearson’s R?(3 marks) 
-- ANSWER 
Pearsons R, also known as the Pearson correlation coefficient (denoted as \( r \)), is a statistical measure that quantifies the
strength and direction of the linear relationship between two continuous variables. It was developed by Karl Pearson in the early 
20th century.
The Pearson correlation coefficient (r) is the most widely used correlation coefficient and is known by many names:

 * Pearson’s r
 * Bivariate correlation
 * Pearson product-moment correlation coefficient (PPMCC)
 * The correlation coefficient

The Pearson correlation coefficient is a descriptive statistic, meaning that it summarizes the characteristics of a dataset. 
Specifically, it describes the strength and direction of the linear relationship between two quantitative variables.

Although interpretations of the relationship strength (also known as effect size) vary between disciplines, the table below 
gives general rules of thumb:

-- 4. What is scaling? Why is scaling performed? What is the difference between normalized scaling and standardized scaling? (3 marks) 
-- ANSWER 
Scaling is the process of transforming the features of your data to a similar range or standard so that they can be compared on 
the same scale. In machine learning, many algorithms are sensitive to the magnitude of the features. Therefore, scaling ensures 
that the data is treated uniformly and that no feature dominates due to its range of values.

### Why is Scaling Performed?

1. Improves Model Performance:
   - Some machine learning algorithms like k-nearest neighbors (KNN), support vector machines (SVM), and gradient descent-based algorithms (e.g., linear regression, logistic regression) are sensitive to the scale of features.
   - When features are on different scales, models that use distance-based metrics may give undue importance to variables with larger ranges.

2. Ensures Faster Convergence:
   - Algorithms that involve gradient descent optimization converge faster when features are scaled since large differences in feature magnitudes can cause the model to take longer to optimize.

3. Prevents Dominance of Certain Features:
   - Without scaling, features with a large range may dominate the model, leading to biased predictions.

---

### Difference Between Normalized Scaling and Standardized Scaling

#### 1. Normalized Scaling (Min-Max Scaling):

- Definition: Normalization transforms the data into a fixed range, usually between 0 and 1. It is calculated as:
- Range: The resulting values lie between 0 and 1 (or any predefined range like [-1, 1]).

- When to Use:
   - Normalization is preferred when the distribution of the data is not Gaussian (non-normal) or when the algorithm does not make any assumption about the distribution of data (e.g., KNN, neural networks).
   - It is useful when you need to preserve the relative relationships between data points (e.g., the ratio of values).

#### 2. Standardized Scaling (Z-score Standardization):

- Definition: Standardization scales data to have a mean of 0 and a standard deviation of 1. It is calculated as:
- Range: The values are not bounded within any specific range. Standardized values typically range between -3 and +3, but they can 
exceed this based on the distribution.

- When to Use:
   - Standardization is preferred when the data follows a normal (Gaussian) distribution and when algorithms assume a standard distribution of the input data (e.g., linear regression, logistic regression, SVM, K-means).
   - It is often used in cases where the algorithms assumption includes a normal distribution, or when the spread of data across different features is important.


### Key Differences

| Aspect              | Normalized Scaling (Min-Max Scaling)                     | Standardized Scaling (Z-score Standardization)       |
|---------------------|----------------------------------------------------------|------------------------------------------------------|
| Range           | Scales data to a fixed range, usually [0, 1]              | Scales data to have mean 0 and standard deviation 1   |
| Effect on Outliers | Sensitive to outliers (can distort range)                | Less sensitive to outliers but still impacted         |
| Use Case        | Useful for distance-based algorithms (e.g., KNN, NN)      | Useful when the data follows a Gaussian distribution  |
| Examples        | Useful in deep learning and KNN models                    | Common in linear models like linear/logistic regression|


--5. You might have observed that sometimes the value of VIF is infinite. Why does this happen? (3 marks)
-- ANSWER 
The Variance Inflation Factor (VIF) measures how much the variance of a regression coefficient is inflated due to multicollinearity 
between the independent variables. In cases where VIF becomes infinite, it indicates a perfect multicollinearity between one or 
more independent variables. This means that one predictor variable is a perfect linear combination of the other predictor(s), 
making it impossible to compute a unique solution for the regression coefficients.

### Why VIF Becomes Infinite:

1. Perfect Multicollinearity:
   - If one independent variable is a perfect linear function of another (or a combination of others), the VIF for that variable 
   will be infinite. For example, if:
     Then the regression model cannot estimate unique coefficients because X^3 does not provide any new information beyond X^1 
     and \( X_2 \). The VIF for \( X_3 \) will be infinite.
   
2. Singular Matrix in Regression:
   - When perfect multicollinearity exists, the design matrix (matrix of predictor variables) in a regression model becomes 
   singular (non-invertible). This leads to an inability to compute the inverse of the matrix, which is required to estimate regression coefficients. As a result, the VIF calculation breaks down, and the value is effectively infinite.


### How to Handle Infinite VIF:
1. Remove Redundant Variables: Identify and remove variables that are perfect linear combinations of other variables.
2. Principal Component Analysis (PCA): Use dimensionality reduction techniques like PCA to transform correlated variables into a 
set of uncorrelated components.
3. Regularization: Techniques like Ridge Regression can help mitigate the effects of multicollinearity by adding a penalty for 
large coefficients.

-- 6. What is a Q-Q plot? Explain the use and importance of a Q-Q plot in linear regression. (3 marks)
-- ANSWER 

A Q-Q (Quantile-Quantile) Plot is a graphical tool used to compare the distribution of a dataset to a theoretical distribution, 
typically the normal distribution. It helps assess whether a dataset follows a particular distribution by plotting the quantiles of 
the dataset against the quantiles of the theoretical distribution.

- X-axis: Theoretical quantiles (from the theoretical distribution, usually normal).
- Y-axis: Sample quantiles (from the observed data).

If the data follows the theoretical distribution, the points in the Q-Q plot will roughly lie along a 45-degree line. 
Deviations from this line indicate departures from the expected distribution.

### How to Interpret a Q-Q Plot:

1. Straight Line: If the points follow a straight, diagonal line, it suggests that the data follows the theoretical 
distribution (e.g., normal distribution).
   
2. Upward/Downward Curvature:
   - A concave or convex shape indicates that the data is skewed:
     - Upward curve (concave): Data is right-skewed (positively skewed).
     - Downward curve (convex): Data is left-skewed (negatively skewed).

3. S-Shaped: If the points form an "S" shape, it indicates that the data has heavy tails (kurtosis), meaning there are more 
extreme values than expected in a normal distribution.


### Use and Importance of Q-Q Plot in Linear Regression

In linear regression, several assumptions must be satisfied for the model to be valid. One of these assumptions is that the 
residuals (errors) should be normally distributed. A Q-Q plot is commonly used to check this normality assumption.

#### Importance of Q-Q Plot in Linear Regression:

1. Assess Normality of Residuals:
   - The most crucial assumption in linear regression is that the residuals (differences between observed and predicted values) 
   are normally distributed. By plotting the residuals in a Q-Q plot, you can visually assess whether they deviate from normality.
   - Why is this important?: If residuals are not normally distributed, the confidence intervals, hypothesis tests (like t-tests),
    and p-values for the regression coefficients may become invalid.

2. Detect Skewness and Kurtosis:
   - A Q-Q plot helps you detect skewness (asymmetry) or heavy/light tails in the residuals. Non-normally distributed residuals 
   might suggest that a different model is more appropriate or that transformations (e.g., logarithmic or square root) should be 
   applied to the data.

3. Diagnose Model Fit:
   - If the residuals are not normally distributed, this could indicate that the linear model does not fit the data well, and 
   alternative models (such as polynomial regression or non-linear models) may need to be considered.

4. Identifying Outliers:
   - A Q-Q plot can reveal outliers that do not follow the distribution of the rest of the data. If a few points deviate 
   significantly from the line, it may indicate influential outliers that could skew the regression model.

