

Avocado Harvesters 
Using Machine Learning to Predict Avocado Sales 
3-19-2025




📖 Project Description & Goals
The focus of this project was to leverage machine learning tools to predict avocado sales in the continental US.  Utilizing data from The Haas Avocado Board and Kaggle Avocado Sales (which was sourced from The Haas Avocado Board).  Our main working dataset was the Kaggle Avocado Sales dataset, which contained information on US Avocado sales from Jan 2015 through March 2018.  Our goals for this project were to:
Prepare the data (cleanup) for the machine model applications in parallel (Kaggle dataset from January 2015 through March of 2018)
Independent preparation, then fit, and apply a selection of time based machine learning models to predict sales for the remainder of 2018 (based on 2015 through 2017 data).  We used a variety of models so that we could compare the accuracy of each.
Models used:
Linear Regression
Random Forest
Facebook Prphet
Model accuracy predictions were then made and we compared our results.  The individual outcomes utilized the same base data and both similarities and differences were found (see the conclusion section for our findings) 
Kristin then calculated the price elasticity of demand for Total Avocados
Data collection, cleanup, and exploration processes.  
Notes on Decisions made early on:
Created an additional YEAR and MONTH column to separate this data out for future analysis needs.
Decided to remove REGIONS and only analyze the data based on CITIES. If we deleted CITIES and only analyzed REGIONS, our dataset would have become too small to work with.
Decided to Drop 5k+ rows of XLarge Avocados (out of 18k rows) due to zero values and the nature of this data

One of the interesting things about running our analysis in parallel is that we approached both data cleanup and machine learning applications differently.  See the comparison of box plots below.  
___________________________________________________________________________________

Initial investigations - Visualizations and Accuracy Calculations
In the first box plot, Kristin focused on Small and Large Avocados and compared Total Avocado Unit Sales / Year for each size.  
In the second box plot, Raymond compared the price of Total Avocados by City.  
Both Kristin & Raymond found a large number of outliers in our preliminary Box Plots, with Kristin having many more.  As her data was focused on _____ and Raymond on _____

Kristin’s Box Plot of Avocado sales volume per year:

Raymond’s Boxplot of Average Price vs. City (Region Name):


Raymond also created a box plot to analyze the distribution of Average Avocado Price by Season.  In this plot we can see Fall has the highest median price, and Winter has a lower median price when compared to Fall and Summer.  The price distributions of Spring and Summer look the same with more outliers in the Fall and Spring. 


Heatmaps:
Kristin chose to do a correlation heatmap comparing Avocado Prices by Season, and found that the price of Avocados in Spring and Summer have the highest correlation (0.93), meaning that prices tend to move similarly within the seasons.  There is also a high correlation between the prices in Fall and Spring (.81). Fall and Winter seasons have a negative correlation (-0.50), so prices move in opposite directions. Winter does not have a strong correlation to any season overall, which means that Winter prices tend to behave differently and follow a different pattern than in any other season.



Bringing the two different approaches together, we can see that each chart is pointing to the same key takeaways: if prices are high in Fall, they will likely also be high in Spring and Summer. Fall prices tend to be highest, and Winter prices tend to move in a different pattern (low correlation, and different median in the box plot)
____________________________________________________________________________
Machine Learning Time Series Models
Model 1: Linear Regression Model
Both Raymond and Krisin ran a Linear Regression model on their respective Avocado datasets.
Raymond approached his Linear Regression model by utilizing OneHotEncoder to encode the City names, Seasons and .  He went from 6 columns to 67 (!) and basically 10x his data in the dataset, introducing lots of duplicates. This is the reason when you look at his regression model, it is basically a sea of blue dots.  If this model was strong, the blue dots would be tightly clustered around the red line. His prediction model has high variance and is also underpredicting higher actual prices because there is a large gap 
 

Linear Regression Calculation Results:
Mean Squared Error: 0.1251
Root of the Mean Squared Error: 0.3537
Mean Absolute Error: 0.2814
R^ Squared: 0.2644
From these values you can see that Raymond’s model is not very accurate…this is because he introduced a lot of duplication inside the dataset, and the model read each repeated datapoint. 
Kristin approached the analysis slightly differently, focusing only on Total Large Avocado Sales. 

Mean Squared Error: 170549560405.2831
R-squared Score: 0.9192
You can see that Kristin’s model visually follows a 45* angled line, and has an R-squared score of 0.91, which means it is pretty accurate.  However, not all points fall directly on the line, which means that it may not be able to predict higher sales volumes as accurately. 
Kristin then decided to experiment with the model by splitting the data before testing and training it.  She again used Linear Regression, but split her dataset into training data (full years 2015, 2016, and 2017) and testing data (2018, partial data set through 3/31/18 only).  Using this approach yielded a slightly less accurate R-squared score (0.89).  This model would struggle even more with accurately forecasting higher volume Large Avocado sales.

Mean Squared Error: 276894562322.6763
R-squared Score: 0.8992
What we learned from comparing Kristin’s and Raymond’s approaches is having more datapoints in the dataset does always equal a more accurate model.  
Kristin also an additional machine learning model to compare model accuracy:
Model 1A: Random Forest Regression Graphs
Taking the standard approach to the Random Forest Regression model(test and train on full dataset), generated an R-squared score of 0.88.  This model is less accurate than the Linear Regression model, but the general pattern of the scatterplot still follows the 45* line.  This model will also have issues predicting higher volumes with accuracy. 
Mean Squared Error: 241597325561.9456
R-squared Score: 0.8855
When using Random Forest Regression combined with the split data approach (train on full years 2015-17, test on partial year 2018), the results continue to deteriorate due to the very small amount of data available in the test dataset.  You can see below that while the shape of this scatterplot is vaguely up and to the right, the R-squared score is less than the prior models (0.87).

Mean Squared Error: 350487682901.1068
R-squared Score: 0.8724
When using Random Forest Regression combined with the split data approach (train on full years 2015-17, test on partial year 2018), the results continue to deteriorate due to the very small amount of data available in the test dataset.  You can see below that while the shape of this scatterplot is vaguely up and to the right, the R-squared score is less than the prior models (0.87).

Model 2: Prophet Forecasting
Both Raymond and Kristin utilized Facebook Prophet to forecast future Avocado Sales.  You can see from the graphs below that the approaches and the results varied significantly.  We’ll discuss Raymond’s model first, where he uses Facebook Prophet to forecast the price of Avocados for the remainder of 2018.
Raymond decided to do some additional work on his dataset, and used OneHotEncoder to break out each of the cities.  This resulted in a 10x increase in columns, basically multiplying each datapoint by 10.  The impact to his Facebook Prophet model is that the model itself had to run through 10x more data points (so took upwards of 10 minutes to run each time), and still had a very low accuracy forecast.  

Mean Squared Error: 0.1854
Root of the Mean Squared Error: 0.4306
Mean Absolute Error: 0.3623
R^ Squared: -0.0601
You can see in Raymond’s visualization, not only are there a LOT of data points, but there is a ton of variance in the forecast range.  The farther out the model predicts, the bigger the swing of ‘potential results’ it provides. It also shows negative volume forecast, which doesn’t fit with what we know to be the purchasing trend on avocados…avocados are sold throughout the year (there would never be a season where absolutely no avocados are sold, nor would there be a time in the year when the retailer would pay the customer to take home avocados…).
Kristin decided to forecast Small Avocado sales volume for the remainder of 2018, as full year data was missing from the dataset.  On Kristin’s forecast model, you can see that the variance intervals are much tighter, and the volume never goes below zero.  The sales volume pattern roughly follows a similar seasonality pattern for earlier years.  There are several outliers present in the graph, but the forecast model is much more accurate.

 
 
Kristin also ran several over analyses (analyzing & forecasting organic vs. conventional avocado volumes, etc.) but because the number of organic avocados was such a small portion of the dataset, the results were not really meaningful. 


Whoops!

________________________________________________________________________________
Price Elasticity of Demand (Using Log-Log Regression)
One of the more exciting parts of this project was using our learnings to calculate the price elasticity of demand for Avocados.  Price elasticity of demand represents the relationship between the price and the quantity sold. Because we were analyzing a consumer good with changing prices over different months and seasons of the year, it was important to calculate how ‘price sensitive’ Avocados were during this period.
In order to determine what the overall price elasticity of demand trend was on Total Avocados, Kristin ran a Log-Log regression to determine the elasticity value – this means taking the log of the values in order to create a linear regression. Kristin then used an OLS (Ordinary Least Squares) regression to compare the relationship between Price and Volume.  
The resulting elasticity was -4.69.  This means that a 1% increase in price results in -4.69% decrease in quantity purchased (and conversely, when you decrease the price by 1%, you’ll see 4.69% increase in quantity sold).  
In the chart below, you can see this negative relationship in the red line going down to the right…as the price increases, the volume of avocados goes down. 

The final piece of analysis that Kristin did for this project was to compare price elasticity between 2 years…in the dataset we noted that 2017 not only had higher volumes of avocados sold, but there were more avocados sold at higher price points than in 2015.  Comparing the price elasticity of demands for both years, you can see on the chart below that in 2016, the market was very price sensitive to avocados. (elasticity of -6.70, and the x-axis tops out at just about $1.00). However, in 2017, the market was not only willing to pay more but also to buy more avocados at a higher price elasticity of -3.57, and the x-axis goes to $1.25).  2017 marked the height of the Avocado Toast Millennial trend, and this bears out in the data.
 

Ideas for Further Investigation
Investigate Organic vs Conventional Avocado pricing patterns & correlations by season
Find a way to predict Organic Avocado sales, and dig into whether Organic Avocado sales were responsible for more overall Avocado sales in 2017.
Run the same analysis on the full 2015-2025 dataset (which we found was available only after completing this project)
Dependencies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from prophet import Prophet

Location:  GitHub repository: URL
🌟 Acknowledgments
Thank you to avocados for being so delicious, and to the Haas Avocado Board for making this interesting data freely available.
Thank you to our instructors in the AI & Machine Learning Bootcamp
Thank you to Kaggle for being a great source of data for students learning to code!
📬 Contact
Kristin Peters
Raymond Stover

