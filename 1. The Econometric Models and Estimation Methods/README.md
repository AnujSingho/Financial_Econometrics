# Types of Regression Models, its Applications and Limitations 

## 1. OLS Regression, Assumptions and Limitations 
  
  ### 1.1 OLS Regression 

  #### 🔹 1.1.1 Simple OLS (Single Regressor)
  The model is defined as:  

  $$
  y_i = \beta_0 + \beta_1 x_i + \varepsilon_i
  $$  

  where:  
  - $$\( y_i \)$$ = dependent variable  
  - $$\( x_i \)$$ = independent variable  
  - $$\( \beta_0, \beta_1 \)$$ = regression coefficients  
  - $$\( \varepsilon_i \)$$ = error term  

  The OLS estimator minimizes the **sum of squared residuals**:  

  $$
  \hat{\beta} = \arg\min_{\beta} \sum_{i=1}^n (y_i - \beta_0 - \beta_1 x_i)^2
  $$  

  ---

  #### 🔹 1.1.2 Multiple OLS (Multivariable Regression)
  For multiple regressors, the model is expressed as:  

  $$
  y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \cdots + \beta_k x_{ik} + \varepsilon_i
  $$  

  In **matrix form**:  

  $$
  \mathbf{y} = \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\varepsilon}
  $$  

  with the OLS solution:  

  $$
  \hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}
  $$  

  ### 1.2 OLS Assumptions 

  #### 1.2.1 Linear Relationship 

  The OLS regression model assumes that the dependent variable has a linear relationship with independent variables. If $Y$ is the dependent     variable and $X_1$, $X_2$ and $X_3$ are independent variables and $e$ is the error term, we can write an OLS regression model as follows:

  $$ Y = \beta_0 + \beta_1  X_1 + \beta_2 X_2 + \beta_3 X_3 + e $$

  #### 1.2.2 No Multicollinearity

  This assumption states that independent variables should be linearly independent to each other. In reality, many independent variables are     correlated to each other.

  #### 1.2.3 Independent Error Term

  This assumption for OLS means that the error terms from each observation will be independent of each other.

  Assume we have these two observations from the dataset $$( y_1, X_{11}, X_{21}, X_{31} )$$ and $$( y_{2}, X_{12}, X_{22}, X_{32} )$$. We can   plug these two observations into an OLS regression model system.
  
  $$ y_{1} = \beta_{0} + \beta_{1}  X_{11} + \beta_{2}  X_{21} + \beta_{3} X_{31} + e_{1} $$
  
  $$ y_{2} = \beta_{0} + \beta_{1} X_{12} + \beta_{2} X_{22} + \beta_{3} X_{32} + e_{2} $$
  
  This assumption states that $e_{1}$ and $e_{2}$ are independent of each other. If they are not independent, we call them **autocorrelated**.

  #### 1.2.4 Error Terms are Normally Distributed 

  This means that $e_{1}$ and $e_{2}$ have to be normally distributed.
  
  $$[e \sim N (0, \sigma^{2})]$$

  #### 1.2.5 Expected Error terms are 0

  This means that $e_{1}$ and $e_{2}$ have to be 0.

  #### 1.2.6 Error Terms has same variance **(Homoscedasticity)**

  When the variances of error terms are constant, we call the error terms **homoscedastic.** If the error terms are not constant, we call them   **heteroskedastic.**

  ### 1.3 IID : Independent and Indentically Distributed 

  As per the assumptions (1.2.4),(1.2.5), and (1.2.6) we can deduce a relationship:

  $$[e \sim N (0, \sigma^{2})]$$

  Now we can write the multiple regression mean function as follows:

  $$ 
  E (Y|X) = \beta X 
  $$
  
  And variances of the errors are constant.
  
  $$ 
  \mathrm{Var}(Y|X)= \mathrm{Var}(e|X) = \delta^{2} 
  $$
  
  Because the variance function is for multiple regression, $\mathrm{Var}(e)$ is actually a covariance matrix.
  
  $$ 
  \mathrm{Var}(e) = \begin{pmatrix}
    \delta ^{2} & 0 & 0 & \cdots & 0 \\ 
    0 & \delta ^{2} & 0 & \cdots & 0 \\ 
    0 & 0 & \delta ^{2} & \cdots & 0 \\ 
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    0 & 0 & 0 & \cdots & \delta ^{2} \\ 
  \end{pmatrix} 
  $$
  
  The values on the diagonal of the matrix are the constant variance, and the values off the diagonal of the matrix are $0$. Because the assumption error terms are independent, the covariances of different error terms are $0$.
  
  Now we want to see what happens when any of the above assumptions do not exist. 
  
  ### 1.4 Limitations of OLS 
  
  THE situations when the assumptions are violated. 
  
  A. Linear relationship.  
  B. Low multicollinearity.   
  C. Homoscedasticity.    

  **When the homoscedasticity assumption does not hold up and when heteroskedasticity exists. Particularly, we will introduce a weighted least square (WLS) regression to handle heteroskedasticity.**

## 2. WLS Regression, Assumptions and Limitation

**Residuals** are the differences between **measured values** and **predicted values** of **dependent variables** from the OLS regression model.  
Before moving to the WLS Regression we need to detect Heteroskedasticity by:  
a. **Scatter Plot**  
b. **Breusch-Pagan (BP) Test** (Chi-Square Test)    
c. **White Test**

  ### 2.1 Basics of WLS  
  When the error terms are not constant and started to spread on the scatter plot as the variance increases. It indicates that the Heteroskedasticity exist.  
  
  $$ 
  \mathrm{Var}(e) = \begin{pmatrix}
    \delta ^{2} & 0 & 0 & \cdots & 0 \\ 
    0 & \delta ^{2} & 0 & \cdots & 0 \\ 
    0 & 0 & \delta ^{2} & \cdots & 0 \\ 
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    0 & 0 & 0 & \cdots & \delta ^{2} \\ 
  \end{pmatrix} 
  $$

 where  $\delta_{1}^{2} \neq \delta_{2}^{2} \neq \delta_{3}^{2} \neq \cdots \neq \delta_{n}^{2}$  

 Let's assume the variance of the error term of observation $i$ is as follows:

  $$ 
  \mathrm{Var}(e_{i}) = \delta_{i}^{2} = \frac{\sigma ^{2}}{w_{i}} 
  $$
  
  Where $w_{1}, \cdots, w_{n}$ are known positive numbers
  
  For weighted least square regression, we need to solve for coefficients by minimizing the following sum of weighted squared residuals.
  
  $$ 
  RSS(\beta_{0}, \beta_{1}) = \sum_{i=1}^{n} w_{i} (Y_{i} - \beta_{0} - \beta_{1}  X_{i})^{2} 
  $$
  
  The above function is called an **objective function**. An **objective function** is a function we would like to maximize or minimize.  
  the weight $w_{i}$ is inversely proportional to the variance of each observation. It means for an observation that is highly variable, it will be given a smaller weight for its weighted squared residual. Hence, this observation will have less importance in the minimization process. For an observation that has small variability, its weight will be larger. Its weighted squared residual will be larger, and it will be more important in the minimization process.  

### 2.2 Derivation of Coefficient for WLS Regression
We first use a single regression model to derive coefficients for weighted least square regression.

Assume we have the following simple regression model for n observations:

$$ 
Y_{i} = \beta_{0} + \beta_{1} X_{i} + e_{i}, \quad  i = 1, …, n  
$$

Where $e \sim N( 0, \frac{\sigma ^{2}}{w_{i}} )$ and $w_1,…,w_n$ are known positive numbers

For weighted least square regression, we are looking for $\beta_{0}$ and $\beta_{1}$ that will minimize the sum of weighted squared residuals. Here is the objective function:

$$ 
RSS (\beta_{0}, \beta_{1})= \sum_{i=1}^{n}w_{i} ( Y_{i} - \beta_{0} - \beta_{1} X_{i} )^{2} 
$$

We take partial derivatives of the objective function with respect to $\beta_{0}$ and $\beta_{1}$ to get the WLS estimates of $\hat{\beta_{0}}$ and $\hat{\beta_{1}}$

$$ 
\begin{align*}
  \hat{\beta_{0}}  &= \overline{Y_{w}} - \hat{\beta_{1}} \overline{X_{w}} \\
  \hat{\beta_{1}}  &= \frac{\sum_{i=1}^{n}  w_{i}  (X_{i} - \overline{X_{w}}) (Y_{i} -\overline{Y_{w}})}{\sum_{i=1}^{n}  w_{i}  (X_{i} - \overline{X_{w}})^{2}} 
\end{align*} 
$$

Where $\overline{X_{w}}$ and $\overline{Y_{w}}$ are weighted averages of $X$ and $Y$ with weights $w$.

$$ 
\overline{X_{w}} = \frac{\sum_{i=1}^{n} X_{i} w_{i}}{\sum_{i=1}^{n} w_{i}}  \quad \text{  and  } \quad  \overline{Y_{w}} = \frac{\sum_{i=1}^{n} Y_{i} w_{i}}{\sum_{i=1}^{n} w_{i}} 
$$

We can apply the same process above to solve for coefficients for weighted least square regression with multiple independent variables. This is a general form of derivation because we don't restrict the number of independent variables. Since we have multiple independent variables now, we will use the matrix form to solve the minimization problem.

For multiple weighted least regression, we have the model:

$$ 
\mathbf{Y} = \mathbf{X} \boldsymbol{\beta } + \mathbf{e} 
$$

$$ 
\mathrm{Var}(e) = \sigma^{2} \mathbf{W}^{-1} 
$$

We minimize the following sum of weighted squared residuals in matrix form for multiple weighted least square regression models. Here is the objective function:

$$ 
RSS(\boldsymbol{\beta }) = \sum_{i=1}^{n} w_{i} (Y_{i} - X_{i}^{t} \boldsymbol{\beta})^{2} = (\mathbf{Y} - \mathbf{X} \boldsymbol{\beta})^{t} \mathbf{W} (\mathbf{Y} - \mathbf{X} \boldsymbol{\beta}) 
$$

We take the partial derivatives of the above objective function with respect to $\boldsymbol{\beta}$ and set the equations to $0$ to solve for $\boldsymbol{\beta}$.

$$ 
\sum_{i=1}^{n} w_{i} e_{i} X_{i}^{t}= 0 
$$

The general form of WLS estimator is given by

$$ 
\hat{\boldsymbol{\beta}} = (\mathbf{X}^{t} \mathbf{W} \mathbf{X})^{-1}  (\mathbf{X}^{t} \mathbf{W \textbf{Y}}) 
$$

### 2.3 Estimating Weights 

We don't always know the numbers of the weight. 

$$ w_{i}= \frac{\sigma^{2}}{\sigma_{i}^{2}} $$

Since $w_{i}$ is the weight and $\sigma^{2}$ is an unknown constant and we only care about relationship among weights, not the scale, **we can assume $\sigma^{2}$ to be $1$**. As such, we can try to estimate the following weight function.
  
$$ w_{i}= \frac{1}{\sigma_{i}^{2}} $$

However, we still don't know the variance for each error term of the observation ($\sigma_{i}^{2}$). We will need to estimate the variances of error terms.

steps to run a WLS regression model:

**Step 1:** Run an OLS regression model.

**Step 2:** Draw a scatter plot with fitted values from OLS on the $X$ axis and residuals from OLS on the $Y$ axis.

**Step 3:** Inspect the scatter plot to see if there is a cone or megaphone pattern in the plot to indicate heteroskedasticity. You can also use the **Breusch-Pagan test** to check for heteroskedasticity, as discussed in the last section.

**Step 4:** If you determine that there is heteroskedasticity in the OLS model, you will have to estimate the weights. We know from the last section that the weight for each observation is the inverse of the variance. We can use squared residuals from OLS to estimate the variances. It also means to use residuals from OLS to estimate the standard deviations. In order to estimate squared residuals, we will run an OLS regression model with absolute values of the residuals as the dependent variable and the fitted values from OLS model in **Step 1** as an independent variable as follows:

$$ | \text{residuals from OLS} | = 1 + \gamma \cdot \text{fitted values from OLS} + e $$

We use the fitted value from the above OLS regression as the proxy for standard deviation. Variance is the squared standard deviation. We then plug the estimated standard deviation into the following equation to get the weight variable.

$$ \text{weight} =\frac{1}{\text{standard deviation}^{2}} $$

**Step 5:** Once we get the weight variable, we can plug this weight variable into any statistical software to run a weighted least square regression model.  

** Source - WorldQuant University**


 
  




  

  

  


