# 2025_ia651_raavi_govindagari

Code for 2025 IA651 Final Project. Teammates Poorna Raavi and Shiva Prasad

Course: Machine Learning

# Project Title: Calorie Burn Prediction using Fitbit Data

## Project Overview

The project focuses on leveraging Machine Learning (ML) techniques to predict the number of calories burned by an individual based on their physical activity data. The objective is to build regression models that can estimate calorie expenditure by analyzing a variety of input features related to daily exercise and movement.

## Project Goals

We want to experiment with a wide range of regression models to predict calorie values accurately. We aim to apply advanced techniques such as grid search for hyperparameter tuning and ensemble learning to optimize model performance. Through this approach, we intend to identify the most effective model for calorie prediction.

## Dataset Details

### Motivation

We used the Fitbit Fitness Tracker dataset from Kaggle, which captures physical activity and health data for multiple users over a period of time. The dataset includes:

- Total steps in day
- Total Distance
- Very/Fairly/Lightly Active Minutes
- Sedentary Minutes
- Calories

**Source**: [[kaggle-Fitbit Fitness Tracker Data](https://www.google.com/url?q=https%3A%2F%2Fwww.kaggle.com%2Fdatasets%2Farashnic%2Ffitbit%2Fdata)]
This dataset is useful in understanding user behavior and estimating energy expenditure patterns, which can support fitness goals and health recommendations.

### Problem Statement

The goal of this project is to predict how many calories a person will burn based on their activity levels and other physical metrics.

### Why is this useful?

Accurately predicting calorie expenditure can:

- Help users better manage their fitness goals
- Enable health apps to provide more personalized recommendations
- Assist in early detection of irregular activity patterns

### Process Overview

1. Data Preprocessing

- Handling missing values
- Scaled/normalizing numeric features

2. Exploring Data Analysis(EDA)

- Visualized distributions and correlations
- Indentifying key features impacting calories burn

3. Models

- Tried regression models like
  - Linear Regression
  - L1 & L2
  - Support Vector Regressor
  - Decision Tree Regressor
  - Voting Regressor
  - Randomforest Regressor
  - Ada Boosting Regressor
  - Gradient Boosting Regressor
  - XGBoosting Regressor

4. Evaluation

- Tested all models on test dataset
- Evaluated the predictions by each model on test set
- Interpreted model behavior using feature importance

## Exploratory Data Analysis (EDA)

- X is all the features:
  features = [
  'TotalSteps',
  'TotalDistance',
  'VeryActiveMinutes',
  'FairlyActiveMinutes',
  'LightlyActiveMinutes',
  'SedentaryMinutes'
  ]
  & y is calories column
- The problem is **regression problem** beacuse we are predicting calories burned which is a continous values.
- The number of observations we have: 1373 entries

- **Distribution Plot** of each feature is as follows:
  ![Feature-Total Steps](images/image.png)
  ![Feature- Total Distance](<images/image(1).png>)
  ![Feature- Very Active Minutes](<images/image(2).png>)
  ![Feature- Fairly Active Minutes](<images/images(3).png>)
  ![Feature- Lightly Active Minutes](<images/image(4).png>)

**Correlation Matrix:** The matrix shows strong positive correlations between TotalSteps, TotalDistance, VeryActiveMinutes, LightlyActiveMinutes, and Calories, indicating that increased activity leads to more steps, distance, and calorie burn. SedentaryMinutes has a weak negative correlation with active metrics and Calories, implying less activity and lower calorie expenditure. FairlyActiveMinutes shows moderate positive correlations with other active metrics and Calories, though slightly weaker than the "very" and "lightly" active categories.

![Correlation matrix](<images/image(6).png>)

**Box plots:** The box plots highlight outliers in steps, distance, active minutes, and calories, indicating varied activity levels and potential data anomalies. 'TotalSteps' and 'Calories' show higher variability, as seen in their wider interquartile ranges. Skewness in some features suggests differences between median and mean, pointing to a need for possible data transformations. These insights emphasize the importance of handling outliers and preparing data carefully for accurate modeling.

![Box plot](<images/image(7).png>)
![Box plot](<images/image(8).png>)
![Box plot](<images/image(9).png>)
![Box plot](<images/image(10).png>)
![Box plot](<images/image(11).png>)
![Box plot](<images/image(12).png>)
![Box plot](<images/image(13).png>)

**Scatter plots:**

_Total Steps vs. Calories_: The scatter plot shows a positive correlation, more steps generally lead to higher calorie burn, though the relationship isn't perfectly linear. A few outliers suggest that other factors may influence calorie expenditure and warrant further investigation.

![Scatter plot](<images/image(14).png>)

_Total Distance vs. Calories_:
A similar positive trend is seen between TotalDistance and Calories, with a strong cluster of points indicating a consistent relationship. Outliers may reflect anomalies where distance doesn’t align with expected calorie burn.

![Scatter plot](<images/image(15).png>)

## Model Fitting

### Train/Test Splitting

**Splitting Process**: The dataset was split into features (X) and target ('Calories'), followed by a train-test split using train_test_split with 80% data for training and 20% for testing, ensuring consistent results with a fixed random_state=42.

**Decision on Train/Test Sizes**: An 80-20 split was chosen to provide sufficient data for training while preserving a meaningful subset for evaluating model performance. This helps ensure reliable generalization to unseen data.

### Model Selection

**Selected Models:** A comprehensive suite of regression models was implemented and evaluated to predict the target variable:

- **Linear Regression** – Used as a baseline model for initial performance benchmarking.

- **Ridge and Lasso Regression** – Regularized linear models used to mitigate multicollinearity. Hyperparameter alpha was optimized using GridSearchCV with 5-fold cross-validation on standardized features.

- **Support Vector Regressor (SVR)** – Applied after log transformation and standardization to manage feature skewness. Hyperparameters C and gamma were tuned using GridSearchCV with 5-fold cross-validation to minimize mean squared error.

- **Decision Tree Regressor** – Tuned using GridSearchCV for the best max_depth with 5-fold cross-validation, optimizing for root mean squared error.

- **Voting Regressor** – An ensemble model combining predictions from Linear, Ridge, SVR, and Decision Tree regressors to capture strengths of multiple algorithms.

- **Random Forest Regressor** – Implemented both with default settings and via a pipeline that included scaling. Hyperparameter tuning was performed using GridSearchCV, optimizing parameters such as n_estimators and max_leaf_nodes.

- **AdaBoost Regressor** – Utilized Decision Trees as weak learners. Hyperparameters such as n_estimators, learning_rate, and base estimator max_depth were tuned using GridSearchCV with 5-fold cross-validation.

- **Gradient Boosting Regressor** – Trained with a range of hyperparameters including learning_rate, n_estimators, max_depth, and subsample, with optimization via GridSearchCV.

- **XGBoost Regressor** – Tuned using GridSearchCV with 5-fold cross-validation for parameters including n_estimators, max_depth, and learning_rate.

<br>

**Reason for Model Choice:**

A diverse set of models was chosen to compare simple linear models, regularized approaches, and more powerful ensemble methods. This allowed exploration of linear vs. non-linear relationships, robustness to overfitting, and the benefits of model ensembling. GridSearchCV was applied to ensure optimal hyperparameter selection across models.

**Hyperparameter Tuning and Search Process:**

All applicable models were systematically optimized using GridSearchCV with 5-fold cross-validation to ensure robust and generalizable performance. This involved searching across defined parameter grids to identify the combination that minimized validation error (typically Root Mean Squared Error or Mean Squared Error, depending on model).

- **Ensemble Approaches**: Two ensemble strategies were incorporated:

  - Voting Regressor: Aggregated predictions from diverse base models to improve generalization and stability.

- **Boosting Methods**: Both AdaBoost and Gradient Boosting approaches were used to iteratively reduce prediction errors by combining weak learners.

This methodical selection and tuning ensured the models were well-calibrated and capable of capturing both linear and nonlinear relationships in the data.

## Validation/Metrics

**MAE (Mean Absolute Error)**: Measures the average of the absolute errors between predicted and actual values. A lower MAE indicates better model performance.

**MSE (Mean Squared Error)**: Measures the average squared difference between predicted and actual values. Like MAE, a lower MSE indicates better model performance, but it penalizes larger errors more heavily.

**RMSE (Root Mean Squared Error)**: The square root of MSE, providing error in the same units as the target variable. Lower RMSE indicates better model performance.

**R² Score (Coefficient of Determination)**: Represents the proportion of the variance in the dependent variable that is predictable from the independent variables. An R² value closer to 1.0 indicates a better fit of the model.

**Metrics Comparsion Table of Each Model:**

| Model                                  | MAE    | MSE        | RMSE   | R² Score |
| -------------------------------------- | ------ | ---------- | ------ | -------- |
| **Linear Regression**                  | 388.66 | 266,882.00 | 516.61 | 0.528    |
| **Ridge Regression**                   | 391.13 | 266,478.34 | 516.22 | 0.528    |
| **Lasso Regression**                   | 390.68 | 266,589.79 | 516.32 | 0.528    |
| **SVR (higher gamma and C)**           | 329.50 | 211,811.97 | 460.23 | 0.625    |
| **Decision Tree Regressor**            | 371.36 | 246,235.02 | 496.22 | 0.564    |
| **Voting Regressor**                   | 371.52 | 233,374.86 | 483.09 | 0.587    |
| **RandomForest (Grid Search)**         | 311.98 | 181,677.43 | 426.24 | 0.678    |
| **AdaBoosting (no GridSearch)**        | 473.04 | 370,839.04 | 608.97 | 0.343    |
| **AdaBoosting (with GridSearch)**      | 340.76 | 201,522.33 | 448.91 | 0.643    |
| **GradientBoosting (no GridSearch)**   | 335.27 | 200,564.23 | 447.84 | 0.645    |
| **GradientBoosting (with GridSearch)** | 313.56 | 180,074.25 | 424.35 | 0.681    |
| **XGBoost Regression**                 | 295.32 | 173,964.52 | 417.09 | 0.692    |

<br>

**Model Performance Summary:**

**Best Performing Models**:

- **XGBoost Regression** demonstrated the lowest MAE (295.32) and MSE (173,964.52), coupled with the highest R² score (0.692), making it the top performer.

- RandomForest Regression also performed well, with a good R² score (0.678) and relatively low MAE (311.98).

- GradientBoosting with Grid Search (R²: 0.681) and AdaBoosting with Grid Search (R²: 0.643) showed improvements over their non-tuned versions, but they still couldn't match the performance of XGBoost or RandomForest.

**Underperforming Model**:

- **AdaBoosting (without Grid Search)** had the worst performance with a very high MAE (473.04), a high RMSE (608.97), and the lowest R² score (0.343).

**Moderately Performing Models:**

- Decision Tree also lagged behind, with a relatively low R² score of 0.564 and a higher MAE (371.36).

- SVR with higher C and gamma values showed solid results with an R² of 0.625, indicating it’s better at capturing non-linear relationships than some of the simpler models.

- Linear, Ridge, and Lasso models produced nearly identical performance metrics with low R² values around 0.528, indicating they did not capture the underlying data patterns as effectively as other more complex models like XGBoost.

- The Voting Regressor, which combines multiple models, slightly outperformed individual models like Decision Trees and AdaBoosting, achieving an R² score of 0.587.

## Predictions using Best Model(XGBoost) on test set

| Index | Actual Calories | Predicted Calories |
| ----- | --------------- | ------------------ |
| 0     | 2354            | 1845.14            |
| 1     | 2324            | 1889.69            |
| 2     | 2044            | 2101.89            |
| 3     | 3439            | 2788.95            |
| 4     | 1878            | 1636.48            |

The predictions on the first 5 test samples show reasonably close estimates to the actual calorie values, capturing the overall trend effectively. While there's some underestimation on higher calorie values (e.g., 3439 predicted as 2788.95), the model demonstrates strong generalization and accurate performance on most typical activity ranges.

## Conclusion

XGBoost regressor emerged as the most effective model for predicting calories burned based on daily activity metrics, outperforming other models across all evaluation metrics. It successfully captured the complex, non-linear relationships in the dataset, making it a robust choice for this regression task. Simpler models like Linear Regression underperformed due to their inability to model these interactions, while ensemble methods with proper tuning showed competitive results.

## Going Further

- Including more user records over a longer period would help the model generalize better across diverse activity patterns.

- Further fine-tuning or stacking models like XGBoost, GradientBoost, and RandomForest could marginally improve results.
