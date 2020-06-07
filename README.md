# Implementación de GAMs en Python
 Ejemplo de aplicación GAMs, presentado en el curso Statistical learning de la Universidad de Antioquía

# Jupyter Notebook HTML

https://wropero.github.io/GAMs_UdeA/

# By PyGAM

Generalized Additive Models (GAMs) are smooth semi-parametric models of the form:

![alt tag](http://latex.codecogs.com/svg.latex?g\(\mathbb{E}\[y|X\]\)=\beta_0+f_1(X_1)+f_2(X_2)+\dots+f_p(X_p))

where `X.T = [X_1, X_2, ..., X_p]` are independent variables, `y` is the dependent variable, and `g()` is the link function that relates our predictor variables to the expected value of the dependent variable.

The feature functions `f_i()` are built using **penalized B splines**, which allow us to **automatically model non-linear relationships** without having to manually try out many different transformations on each variable.

<img src=imgs/pygam_basis.png>

GAMs extend generalized linear models by allowing non-linear functions of features while maintaining additivity. Since the model is additive, it is easy to examine the effect of each `X_i` on `Y` individually while holding all other predictors constant.

The result is a very flexible model, where it is easy to incorporate prior knowledge and control overfitting.

## Citing pyGAM
Please consider citing pyGAM if it has helped you in your research or work:

Daniel Servén, & Charlie Brummitt. (2018, March 27). pyGAM: Generalized Additive Models in Python. Zenodo. [DOI: 10.5281/zenodo.1208723](http://doi.org/10.5281/zenodo.1208723)

BibTex:
```
@misc{daniel\_serven\_2018_1208723,
  author       = {Daniel Servén and
                  Charlie Brummitt},
  title        = {pyGAM: Generalized Additive Models in Python},
  month        = mar,
  year         = 2018,
  doi          = {10.5281/zenodo.1208723},
  url          = {https://doi.org/10.5281/zenodo.1208723}
}
```
