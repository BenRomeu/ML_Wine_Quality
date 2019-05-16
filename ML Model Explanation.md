---
title: "ML Model Explanation"
author: "Benjamin Romeu"
date: "13/05/2019"
output: html_document
---

## Machine Learning Model Explanation

### k-nearest neighbors algorithm (k-NN) for classification

A k-nearest-neighbor is a data classification algorithm that attempts to determine what group a data point is in by looking at the data points around it.

An algorithm, looking at one point on a grid, trying to determine if a point is in group A or B, looks at the states of the points that are near it. The range is arbitrarily determined, but the point is to take a sample of the data. If the majority of the points are in group A, then it is likely that the data point in question will be A rather than B, and vice versa.

The k-nearest-neighbor is an example of a "lazy learner" algorithm because it does not generate a model of the data set beforehand. The only calculations it makes are when it is asked to poll the data point's neighbors. This makes k-nn very easy to implement for data mining.

![](http://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1531424125/KNN_final1_ibdm8a.png)

![](https://github.com/BenRomeu/ML_Wine_Quality/blob/master/knn_strength.PNG)


### Decision Trees for classification

It is build in a model in the form of a tree structure. 
The model itself comprises a series of logical decisions, with decision nodes 
that indicate a decision to be made on an attribute. These split into branches 
that indicate the decision's choices. The tree is terminated by leaf nodes 
that denote the result of following a combination of decisions.

![](https://datastudentblog.files.wordpress.com/2014/01/redwinedtree.png)

### Random Forest

Random forests are an ensemble learning method for classification, regression and other tasks that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Random decision forests correct for decision trees habit of overfitting to their training set.

The training algorithm for random forests applies the general technique of bootstrap aggregating, or bagging, to tree learners. Given a training set X with responses Y, bagging repeatedly selects a random sample with replacement of the training set. After training, predictions for unseen samples can be made by averaging the predictions from all the individual regression trees or by taking the majority vote in the case of decision trees


![](https://www.quantinsti.com/wp-content/uploads/2019/03/Random-Forest-Algorithm.jpg)



### Aritificial Neural Network

An Artificial Neural Network (ANN) models the relationship between a set of input
signals and an output signal using a model derived from our understanding of how
a biological brain responds to stimuli from sensory inputs. Just as a brain uses a
network of interconnected cells called neurons to create a massive parallel processor,
the ANN uses a network of artificial neurons or nodes to solve learning problems.

![](https://groupfuturista.com/blog/wp-content/uploads/2019/03/Artificial-Neural-Networks-Man-vs-Machine.jpeg)







