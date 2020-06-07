# GAMs in Python
 It's an Example of applications in Generalized Additive Models (GAMs), presented in the course statistical learning. It is taught  at university of Antioquía, Medellín, Colombia.

# Jupyter Notebook HTML and Google Colab

Here, you can see two examples for regression and classification in jupyter Notebook.
https://wropero.github.io/GAMs_UdeA/

Here, you can run the code in python.
https://colab.research.google.com/drive/1_eOdHpCysUUP59EItXF8KBogwnSISpLC?usp=sharing

# By PyGAM

Generalized Additive Models (GAMs) are smooth semi-parametric models of the form:

![alt tag](http://latex.codecogs.com/svg.latex?g\(\mathbb{E}\[y|X\]\)=\beta_0+f_1(X_1)+f_2(X_2)+\dots+f_p(X_p))

where `X.T = [X_1, X_2, ..., X_p]` are independent variables, `y` is the dependent variable, and `g()` is the link function that relates our predictor variables to the expected value of the dependent variable.

The feature functions `f_i()` are built using **penalized B splines**, which allow us to **automatically model non-linear relationships** without having to manually try out many different transformations on each variable.

<img src=https://pygam.readthedocs.io/en/latest/_images/pygam_basis.png>

GAMs extend generalized linear models by allowing non-linear functions of features while maintaining additivity. Since the model is additive, it is easy to examine the effect of each `X_i` on `Y` individually while holding all other predictors constant.

The result is a very flexible model, where it is easy to incorporate prior knowledge and control overfitting.

## Citing pyGAM
Please consider citing pyGAM if it has helped you in your research or work:

Daniel Servén, & Charlie Brummitt. (2018, March 27). pyGAM: Generalized Additive Models in Python. Zenodo. [DOI: 10.5281/zenodo.1208723](http://doi.org/10.5281/zenodo.1208723)

# References

<ul>
<li>Green, P. J., & Silverman, B. W. (1993). Nonparametric regression and generalized linear models: a roughness penalty approach. Chapman and Hall/CRC.</li>
<li>Hastie, T. and Tibshirani, R. (1990). Generalized Additive Models, volume 43. CRC Press, 1990.</li>
<li>Wood, S. N. (2017). Generalized additive models: an introduction with R. Chapman and Hall/CRC.</li>
<li>Hastie, T., and Tibshirani, R. (1987). Generalized additive models: some applications. Journal of the American Statistical Association, 82(398), 371-386.</li>
    <li><a href = "https://cran.r-project.org/package=gam">https://cran.r-project.org/package=gam</a></li>
    <li><a href="https://cran.r-project.org/package=mgcv">https://cran.r-project.org/package=mgcv</a></li>
<li><a href="https://www.statsmodels.org/stable/gam.html">https://www.statsmodels.org/stable/gam.html</a></li>
    <li><a href="https://pygam.readthedocs.io/">https://pygam.readthedocs.io/</a></li>
    <li><a href="https://github.com/dswah/PyData-Berlin-2018-pyGAM/">https://github.com/dswah/PyData-Berlin-2018-pyGAM/</a></li>
</ul>
