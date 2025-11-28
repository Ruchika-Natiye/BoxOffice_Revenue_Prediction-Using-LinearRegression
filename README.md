# BoxOffice_Revenue_Prediction-Using-LinearRegression
## 🎬 Box Office Revenue Prediction Using Machine Learning
Predicting domestic box-office revenue using regression and NLP-based genre features.

## 📌 Project Overview
This project builds a machine learning model to predict the domestic revenue of movies based on:
* Release metadata

* MPAA rating

* Distributor

* Number of opening theatres

* Movie genres (encoded using Bag-of-Words)

* Release duration
The model uses XGBoost Regressor as the final predictor and evaluates it using Mean Absolute Error (MAE).

## 📂 Dataset
The dataset contains multiple features including:

| Feature          | Description                               |
| ---------------- | ----------------------------------------- |
| title            | Movie title                               |
| distributor      | Company releasing the movie               |
| MPAA             | Movie rating                              |
| genres           | List of genre tags                        |
| opening_theaters | Opening weekend theatre count             |
| release_days     | Days since release                        |
| domestic_revenue | U.S. box office revenue (target variable) |

## ⭐ Key Highlights
✔ Genre encoding using NLP CountVectorizer

✔ XGBoost for accurate predictions

✔ Extensive EDA: histograms, boxplots, heatmaps

✔ Log-transformation for skewed revenue distributions

✔ Automated sparse-column removal

## 🧹 Step-by-Step Code Working
### Loading the dataset into a pandas DataFrame

![](https://github.com/Ruchika-Natiye/BoxOffice_Revenue_Prediction-Using-LinearRegression/blob/dd340c2e3d36ce4be18bbee1fa5b7cd66e10b2a2/i1.png)

### Checking Dataset Size & Data Types

![](https://github.com/Ruchika-Natiye/BoxOffice_Revenue_Prediction-Using-LinearRegression/blob/3a048ca01ede976cc520efd1269abdf6faf864b3/i2.png)

### Exploring the Dataset

![](https://github.com/Ruchika-Natiye/BoxOffice_Revenue_Prediction-Using-LinearRegression/blob/a00ff29dd15b57a3c4f8ea4ba347be779a1f3feb/i3.png)

### Checking & Handling Missing Values 

![]()

### Visualizing MPAA Rating Distribution

![]()

### Average Domestic Revenue by MPAA Rating

![]()

### Visualizing Distributions of Key Numeric Features

![]()

### Detecting Outliers Using Boxplots

![]()

### Checking Distributions After Log Transformation

![]()

### Removing Rare Genre Columns with Mostly Zero Values

![]()

### Visualizing Strong Correlations Between Numeric Features

![]()

### Preparing Data for Model Training and Validation

![]()

### Evaluating Model Performance on Training and Validation Sets

![]()









## ✅ Final Output
* Error
  
* Validation Error

* Model ready for deployment on unseen movie data

## 
