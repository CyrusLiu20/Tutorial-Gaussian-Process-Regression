# Tutorial-Gaussian-Process-Regression
Gaussian Process Regression (GPR) differs from traditional linear regression, which aims to fit a fixed set of parameters for a given dimensionality of the data. In linear regression, the goal is to find the best-fit line (or hyperplane) that minimizes the error between the observed data and the model's predictions. Specifically, for an 
-dimensional problem, a linear regression model fits 
 parameters (including an intercept).

In contrast, the central idea behind Gaussian Process Regression is that we are not fitting parameters but rather learning a distribution over all possible functions that can describe the data. This probabilistic framework allows GPR to provide not only predictions but also a measure of uncertainty for those predictions. A Gaussian Process (GP) is a subset of stochastic process (a collection of random variables), any finite number of which have a joint Gaussian distribution. It is completely specified by its mean function 
 and covariance function 
, also called a kernel. In GPR, we treat the function we are trying to predict as a sample from a Gaussian Process.

The key strengths of Gaussian Process Regression (GPR) lie in its flexibility and ability to model complex, non-linear relationships without assuming a specific functional form. As a non-parametric method, GPR adapts to data through a covariance function (kernel) that defines how data points are related. It provides not only predictions but also a measure of uncertainty, offering confidence intervals for each prediction. Additionally, by selecting different kernels, GPR can be customized to model a wide variety of patterns, making it highly versatile in handling real-world data.

This repository illustrates the inner workings of Gaussian Process Regression using only NumPy for demonstration purposes and the following tutorial is adapted from this lecture note (https://cs229.stanford.edu/section/cs229-gaussian_processes.pdf).

<img src="GPR Training.png" alt="Gaussian Process Diagram" width="1500">

