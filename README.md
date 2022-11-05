# UDACITY - Enron Dataset Analysis

Analysis of the Enron Dataset to the Udacity Course - Data Science Foundations

The purpose of this project is to use financial and e-mail data from executives of the Enron company, which was released by the US government during the federal investigation, to arrive at a predictive model that could identify people involved in the fraud.
The files include financial and e-mail data of 146 people (records), classified as People of Interest (POI), most of whom are senior management, 14 financial resources, and 6 e-mail resources. Within these records, 18 were labeled as a "person of interest" (POI).

This notebook shows how data analysis on enron dataset can be done. The goal of the analysis is to find the best machine learning algorithm with the best precision and recal metric values. Each algorithms' job on the way is to correctly classify poi(person of interest) from the dataset. POIs are who I am interested in since I think they are strongly related to Enron Scandal. POIs are chosen mannually and provided by Udacity's "Intro to Machine Learning" course. You can think of this notebook as a part of the assignment for the final project from the course.

## Approach:
    Perform basic data analysis
    Choose features of my interest
    Find outliers, and remove them when needed
    Perform various machine learning algorithms
    Compare each results
    Confirm the best result
    
## Machine Learning Part

    Perform basic DecisionTree classifier on raw data
    Perform basic DecisionTree classifier on data that outliers are removed
    Define a function to measure accuracy, precision, and recall metrics
    Define a function to run Pipeline with SelectKBest, and GridSearchCV
    Run different kinds of ML algorithms with a number of different parameters
    
## To Do:
  - Insert the algorithm analysis
