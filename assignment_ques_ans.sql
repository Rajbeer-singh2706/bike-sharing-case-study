-- Q1)From your analysis of the categorical variables from the dataset, what could you infer about their effect on the dependent 
-- variable?       
-- ANSWER 
To analyze the effect of categorical variables on the dependent variable, we can break it down into a few key steps. Typically, 
this is done using exploratory data analysis (EDA) and statistical techniques like cross-tabulation, chi-square tests, and 
visualizations (such as bar charts or box plots) to determine relationships between the categorical variables and the dependent 
variable.

Heres a general approach to analyzing the effect of categorical variables:

1. Cross-Tabulation
   - Method: For each categorical variable, create a cross-tabulation or contingency table to observe how the values of the categorical 
    variables are distributed across the dependent variable categories.
   - Inference: This helps in identifying patterns or imbalances. For example, if you are looking at default status (dependent variable), you can see if certain categories (e.g., "employment type") have higher rates of default.

### 2. Chi-Square Test of Independence
   - Method: Perform a chi-square test to assess if there is a statistically significant relationship between a categorical variable and the dependent variable.
   - Inference: If the test shows a significant result (p-value < 0.05), this suggests that the categorical variable might influence the dependent variable. For example, loan type could significantly affect the likelihood of default.

### 3. Bar Charts and Count Plots
   - Method: Use bar charts or count plots to visualize the distribution of the dependent variable across different categories.
   - Inference: Visualizing the distribution can give a clear idea of the relationship. For instance, if a particular category (e.g., "marital status") leads to a significantly higher proportion of defaults, it indicates that the category may be a key driver of the dependent variable.

### 4. Categorical Encoding (One-Hot or Label Encoding)
   - Method: Encode categorical variables and then use logistic regression or decision trees to assess the effect of these encoded variables on the dependent variable.
   - Inference: The importance of each categorical feature can be gauged from the model’s coefficients (in regression) or feature importance scores (in decision trees or random forests).

### 5. Box Plots (for ordinal categories)
   - Method: For ordinal categorical variables, box plots can show how the dependent variable (if continuous) varies across different categories.
   - Inference: If there is a clear trend or difference in the distribution across categories, it suggests an influence on the dependent variable. For example, income brackets might show a clear difference in loan default likelihood.

### Sample Inference:
- If you have education level as a categorical variable and loan default as the dependent variable, you might find that people with higher education levels tend to default less. This would indicate that education level has a negative correlation with loan default.

The exact inference depends on your dataset and the specific variables in question, but these are general approaches used to infer the effect of categorical variables on a dependent variable.


-- Q2) Why is it important to use drop_first=True during dummy variable creation?
-- ANSWER 

Using `drop_first=True` in `get_dummies()` is important in dummy variable creation to prevent multicollinearity when performing 
regression analysis or machine learning algorithms. Heres why:

1. Multicollinearity:
   - When you have multiple categories for a categorical variable (e.g., "red," "blue," "green" for color), `get_dummies()` will create a 
   separate dummy variable for each category.
   - If all categories are represented by dummy variables, one of the dummies can always be perfectly predicted by the others. This results 
   in perfect multicollinearity, where one variable is a linear combination of others.
   - Multicollinearity can distort statistical tests and make it difficult to interpret the coefficients in regression models.

2. Redundant Information:
   - When you include all dummy variables, one is redundant. For example, if a categorical variable has three levels (A, B, C), 
   creating three dummies means if you know two of the dummy values, you can infer the third.
   - Example: If a variable takes the values A, B, or C, and you create three dummy variables:
     - A: [1, 0, 0]
     - B: [0, 1, 0]
     - C: [0, 0, 1]
   - The third column can be inferred if you have the first two, leading to redundancy.

3. drop_first=True:
   - When `drop_first=True`, pandas drops the first dummy variable and only creates (k-1) dummy variables for k categories.
   - This removes the redundancy and solves the multicollinearity issue.
   - The dropped category is treated as a reference category, and the remaining dummies represent how the other categories differ 
   from that reference.

import pandas as pd

# Example DataFrame
df = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'blue', 'green']
})

# Without drop_first
print(pd.get_dummies(df))

# With drop_first
print(pd.get_dummies(df, drop_first=True))

#Output without `drop_first=True`:
|   | color_blue | color_green | color_red |
|---|------------|-------------|-----------|
| 0 | 0          | 0           | 1         |
| 1 | 1          | 0           | 0         |
| 2 | 0          | 1           | 0         |
| 3 | 1          | 0           | 0         |
| 4 | 0          | 1           | 0         |

#Output with `drop_first=True`:

|   | color_blue | color_green |
|---|------------|-------------|
| 0 | 0          | 0           |
| 1 | 1          | 0           |
| 2 | 0          | 1           |
| 3 | 1          | 0           |
| 4 | 0          | 1           |

 In this case, "red" is the reference category, and the remaining dummies represent how "blue" and "green" differ from "red."

 ### Conclusion:
 Using `drop_first=True` simplifies your model by avoiding redundant information and prevents multicollinearity, improving model 
 interpretability and efficiency.


-- 3. Looking at the pair-plot among the numerical variables, which one has the highest correlation with the target variable? 
-- ANswer
Based on the pair-plot you provided, the relationship between the variables can be observed visually. The target variable seems to be 
 "cnt" (likely representing a count of some event). From the scatter plots:

- The variable "temp" (temperature) shows the strongest positive linear relationship with the target "cnt." This can be seen from the 
diagonal pattern in the scatter plot between "cnt" and "temp." 
- Similarly, the variable "atemp" (which may represent apparent temperature) also shows a strong correlation with "cnt."

Among these two, "temp" appears to have the strongest correlation with the target, based on visual inspection.


--Q4) Based on the final model, which are the top 3 features contributing significantly towards explaining the demand of the 
-- shared bikes?
-- ANSWER 
'''
variables year , season/ weather situation and month are significant in predicting the demand for shared bikes .
'''


-- Q5) How did you validate the assumptions of Linear Regression after building the model on the training set?

After building a Linear Regression model, its crucial to validate the assumptions of the model to ensure its accuracy and 
generalizability. The following are the key assumptions of Linear Regression and common techniques to validate them

### 1. Linearity of the relationship between features and target:
   - Assumption: The dependent variable (target) should have a linear relationship with each independent variable (features).
   - How to validate:
     - Residual Plot: Plot residuals (difference between observed and predicted values) versus the predicted values. The residuals 
     should be randomly scattered around zero, without any distinct patterns (e.g., curved or funnel-shaped).
     - Scatter Plot: Plot the features against the target variable to visually check for linear relationships.
     - Partial Regression Plots: These help to visualize the effect of each predictor on the target while keeping other variables
      constant.
   
   Corrective Action: Apply transformations (e.g., log, polynomial features) if the relationships are non-linear.

### 2. Homoscedasticity (constant variance of errors):
   - Assumption: The variance of the residuals should remain constant across all levels of predicted values.
   - How to validate:
     - Residual Plot: Look for patterns in the residual plot. If the residuals show a "funnel" shape (i.e., the variance increases or decreases as the predicted values increase), it indicates heteroscedasticity.
   
   Corrective Action: Apply transformations to the dependent variable (e.g., log transformation) or use models that can handle heteroscedasticity (e.g., Generalized Least Squares).

### 3. Independence of errors:
   - Assumption: The residuals (errors) should be independent of each other, meaning there is no autocorrelation.
   - How to validate:
     - Durbin-Watson Test: This statistical test detects the presence of autocorrelation in the residuals. A value close to 2 indicates no autocorrelation, while values close to 0 or 4 suggest positive or negative autocorrelation, respectively.
     - Plot Residuals over Time: If your data is time-based (e.g., time series), plot the residuals over time to detect any patterns (which indicate dependence).
   
   Corrective Action: If autocorrelation is present, you might need to use time-series-specific techniques such as ARIMA models.

### 4. Normality of residuals:
   - Assumption: The residuals should be normally distributed.
   - How to validate:
     - Histogram or Q-Q Plot: Plot a histogram of the residuals or use a Q-Q plot (Quantile-Quantile plot) to check if the residuals follow a normal distribution. In a Q-Q plot, the points should fall along the 45-degree reference line if the residuals are normally distributed.
     - Shapiro-Wilk Test: A formal statistical test for normality. However, this test can be overly sensitive in large datasets, so visual inspections are often more practical.
   
   Corrective Action: If the residuals are not normally distributed, you may need to apply transformations to the target variable (e.g., log transformation).

