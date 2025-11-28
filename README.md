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

![](https://github.com/Ruchika-Natiye/BoxOffice_Revenue_Prediction-Using-LinearRegression/blob/7060d3ef8ea9f451516ee849a2c1fbcc80237de4/i4.png)

### Average Domestic Revenue by MPAA Rating

![](https://github.com/Ruchika-Natiye/BoxOffice_Revenue_Prediction-Using-LinearRegression/blob/a3f696d5c34523cb7f14e037f8f4a17d2549b0c2/i5.png)

### Visualizing Distributions of Key Numeric Features

![](https://github.com/Ruchika-Natiye/BoxOffice_Revenue_Prediction-Using-LinearRegression/blob/2a055aaaaca2124f803aeff8f3275f9f15d70b58/i6.png)

### Detecting Outliers Using Boxplots

![](https://github.com/Ruchika-Natiye/BoxOffice_Revenue_Prediction-Using-LinearRegression/blob/24154ae1916db5688a26644822258e7863ff779a/i7.png)

### Checking Distributions After Log Transformation

![](https://github.com/Ruchika-Natiye/BoxOffice_Revenue_Prediction-Using-LinearRegression/blob/a609df5a26a7ea6236b59ced27a2a3ca95bcdf11/i8.png)

### Removing Rare Genre Columns with Mostly Zero Values

![](https://github.com/Ruchika-Natiye/BoxOffice_Revenue_Prediction-Using-LinearRegression/blob/cd508c40fab7be5d1d68b81a47580dbcafd3095a/i9.png)

### Visualizing Strong Correlations Between Numeric Features

![](https://github.com/Ruchika-Natiye/BoxOffice_Revenue_Prediction-Using-LinearRegression/blob/a6d4c129cd8279045ad96c99b11f31996b1fc8ff/i10.png)

### Preparing Data for Model Training and Validation

![](https://github.com/Ruchika-Natiye/BoxOffice_Revenue_Prediction-Using-LinearRegression/blob/fc3a855fae9dd1d3eea94afeaf8e5b6cc993b603/i11.png)

### Evaluating Model Performance on Training and Validation Sets

![]()









## ✅ Final Output
* Error
  
* Validation Error

* Model ready for deployment on unseen movie data

## 
