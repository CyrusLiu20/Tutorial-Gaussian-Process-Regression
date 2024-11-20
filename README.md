# Tutorial-Gaussian-Process-Regression.ipynb (Tutorial 1/3)
Gaussian Process Regression (GPR) differs from traditional linear regression, which aims to fit a fixed set of parameters for a given dimensionality of the data. In linear regression, the goal is to find the best-fit line (or hyperplane) that minimizes the error between the observed data and the model's predictions. Specifically, for an 
-dimensional problem, a linear regression model fits 
 parameters (including an intercept).

In contrast, the central idea behind Gaussian Process Regression is that we are not fitting parameters but rather learning a distribution over all possible functions that can describe the data. This probabilistic framework allows GPR to provide not only predictions but also a measure of uncertainty for those predictions. A Gaussian Process (GP) is a subset of stochastic process (a collection of random variables), any finite number of which have a joint Gaussian distribution. It is completely specified by its mean function 
 and covariance function 
, also called a kernel. In GPR, we treat the function we are trying to predict as a sample from a Gaussian Process.

The key strengths of Gaussian Process Regression (GPR) lie in its flexibility and ability to model complex, non-linear relationships without assuming a specific functional form. As a non-parametric method, GPR adapts to data through a covariance function (kernel) that defines how data points are related. It provides not only predictions but also a measure of uncertainty, offering confidence intervals for each prediction. Additionally, by selecting different kernels, GPR can be customized to model a wide variety of patterns, making it highly versatile in handling real-world data.

This repository illustrates the inner workings of Gaussian Process Regression using only NumPy for demonstration purposes and the following tutorial is adapted from this lecture note (https://cs229.stanford.edu/section/cs229-gaussian_processes.pdf).

<img src="pics/GPR Training.png" alt="Gaussian Process Diagram" width="1500">


# Tutorial-Bayesian-Optimization.ipynb (Tutorial 2/3)
Optimizing a non-linear function $f(\mathbf{x})$ within a compact set $\mathcal{A}$ is of great scientific interest due to its versatility and real-world applicability. However, problems arise when the function has no closed-form analytical solution or is computationally expensive to evaluate. This gives rise to Bayesian optimization to mitigate such problems.

$$
\max _{\mathbf{x} \in \mathcal{A} \subset \mathbb{R}^d} f(\mathbf{x})
$$

Bayesian optimization is a surrogate-based optimization strategy, typically using a Gaussian process regression model to approximate the unknown function. This strategy is particularly useful when the objective function is difficult to evaluate. Initially, we have some prior beliefs about the objective function. As we accumulate more information, $\mathcal{D}_{1: t}$, we obtain a posterior distribution of the function, $P\left(f \mid \mathcal{D}{1}\right)$, which facilitates optimization.


Given a Gaussian process regression model that provides a posterior mean $\mu(\mathbf{x})$ and a posterior covariance $\Sigma(\mathbf{x})$ for the objective function $f(\mathbf{x})$, the next sampling point $\mathbf{x}_{t+1}$ is chosen by maximizing an acquisition function $u(\mathbf{x}\mid \mathcal{D})$ that balances exploration and exploitation.

$$
\mathbf{x}_{\text{next}} = \underset{\mathbf{x} \in \mathcal{A}}{\arg\max} \, u(\mathbf{x}\mid \mathcal{D})
$$


# Tutorial-Batch-Bayesian-Optimization.ipynb (Tutorial 3/3)

Batch Bayesian Optimization is an extension of Bayesian optimization that evaluates multiple points in parallel, making it ideal for scenarios where function evaluations are costly but computational resources allow parallelism. Like standard Bayesian optimization, it uses a surrogate model, typically a Gaussian process, to approximate the objective function and an acquisition function to select points balancing exploration and exploitation. However, in BBO, the acquisition function has to be modified to ensure efficient and diverse sampling. This approach significantly reduces the overall wall-clock time, making it valuable for tasks such as hyperparameter tuning, experimental design, and other computationally expensive optimization problems.

Batch Bayesian Optimization can be mathematically defined as an optimization framework that aims to minimize an expensive-to-evaluate objective function $ f: \mathcal{X} \to \mathbb{R} $ by selecting a batch of $ q $ points $ \mathbf{X}_q = \{ \mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_q \} \subset \mathcal{X} $ for simultaneous evaluation at each iteration. The batch is selected by maximizing an acquisition function, $ \alpha(\mathbf{X} \mid \mathcal{D}) $. The goal is:
$$
\mathbf{X}_{batch}^* = \mathop{\operatorname{argmax}}\limits_{\mathbf{X}_{\text{batch}} \subset \mathcal{X}, |\mathbf{X}_{\text{batch}}| = q}
 \left[\alpha(\mathbf{X \mid \mathcal{D} })\right]
$$