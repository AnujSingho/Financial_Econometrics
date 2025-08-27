# Financial_Econometrics
Repository dedicated to my learning and projects related to Financial Econometrics. It's a branch that applies statistical analysis and mathematical models over the financial data. It caters to "Test Theories, Build Models, and Forecast" asset price, volatility, and risks. 

## 📌 Financial Econometrics Overview

Financial Econometrics is the discipline that applies **statistical and mathematical methods** to financial data.  
It provides the tools to **test theories, build models, and forecast** asset prices, volatility, and risk. 


---

## 🔹 Core Elements of Econometrics

### 1. **The Econometric Model**
At its core, econometrics links a dependent variable (Y) with independent variables (X):

$$
Y_i = \beta_0 + \beta_1 X_{i1} + \beta_2 X_{i2} + \cdots + \beta_k X_{ik} + \varepsilon_i
$$

- $Y_i$: dependent variable (e.g., asset return)  
- $X_{ij}$: explanatory variables (e.g., market return, interest rate)  
- $\beta_j$: parameters to estimate  
- $\varepsilon_i$: error term

---

### 2. Estimation Methods

**Ordinary Least Squares (OLS)** — minimizes squared residuals:

$$
\hat{\beta} = \arg\min_{\beta} \sum_{i=1}^n \big(Y_i - X_i\beta\big)^2
$$

Closed-form (normal equation, matrix form):

$$
\hat{\beta} = (X^\top X)^{-1} X^\top Y
$$

**Maximum Likelihood Estimation (MLE)** — chooses parameters $\theta$ that maximize the likelihood $L(\theta; \text{data})$.

---

### 3. Time Series Econometrics
Financial data is often time-indexed. Common models:

**AR(p) — AutoRegressive:**

$$
Y_t = \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + \cdots + \phi_p Y_{t-p} + \varepsilon_t
$$

**MA(q) — Moving Average:**

$$
Y_t = \varepsilon_t + \theta_1 \varepsilon_{t-1} + \cdots + \theta_q \varepsilon_{t-q}
$$

**ARIMA(p,d,q)** — integrates differencing $(1-B)^d$ to remove trends:

$$
\phi(B)(1-B)^d Y_t = \theta(B)\varepsilon_t
$$

where $B$ is the lag operator, $\phi(B)$ and $\theta(B)$ are polynomials.

**GARCH(1,1)** — models time-varying volatility (volatility clustering):

$$
\sigma_t^2 = \alpha_0 + \alpha_1 \varepsilon_{t-1}^2 + \beta_1 \sigma_{t-1}^2
$$

---

### 4. Hypothesis Testing & Inference

- Null hypothesis example: $H_0: \beta_1 = 0$  
- Alternative: $H_1: \beta_1 \neq 0$

t-statistic for coefficient $\beta_j$:

$$
t = \frac{\hat{\beta}_j - \beta_{j,0}}{\widehat{\mathrm{SE}}(\hat{\beta}_j)}
$$

Use F-tests for joint hypotheses and likelihood-ratio tests for nested models.

---

### 5. Model Diagnostics & Good Practices

- Check residuals for autocorrelation (e.g., Ljung–Box) and heteroskedasticity (e.g., Breusch–Pagan).  
- Use AIC / BIC for model selection.  
- Stationarity tests: ADF, KPSS.  
- Robust standard errors (e.g., Newey–West) when errors have autocorrelation or heteroskedasticity.

---

## Applications in Quantitative Finance

- **Asset Pricing:** CAPM, factor models (Fama–French).  
- **Volatility Forecasting:** GARCH family for risk/option pricing.  
- **Portfolio Construction:** estimate expected returns / covariances.  
- **High-frequency / Microstructure:** modeling order flow and intraday volatility.  
- **Risk Management:** Value-at-Risk (VaR), Expected Shortfall.

---

## 🚀 Why It Matters for Quantitative Finance
Financial Econometrics enables quant students and practitioners to:
- Build **data-driven trading strategies**  
- Forecast **risk and volatility**  
- Test and validate **financial theories**  
- Turn **raw market data into actionable insights**  

---

📖 *This repository will contain implementations of these models in Python/R for practical applications in Quantitative Finance.*
