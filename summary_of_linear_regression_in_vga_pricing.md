# Application of Linear Regression, MAP Estimation, and Multicollinearity Handling in VGA Hardware Pricing

## I. Algorithm Explanation and Mathematical Foundation (Course MAI391)

**Linear Regression (Chapter 9, Deisenroth et al.)** is selected as the core algorithm to solve the graphics card (VGA) price prediction problem, comprehensively integrating the mathematical foundations of the semester: Vector Calculus (Chapter 5), Subspace Geometry (Chapter 3), Matrix Decomposition (Chapter 4), and Probabilistic Statistics MLE/MAP (Chapters 6 & 9).

### 1. Mathematical Nature of the VGA Pricing Problem

* **Hedonic Pricing Model (Engineering Economics):** The retail price of a VGA is not random, but rather the aggregation of the marginal values of individual hardware specifications and brand equity.

* **Mathematical Representation:** Given $N$ samples ($N \approx 400 - 500$) and $D$ encoded technical features, each $n$-th VGA is represented by a feature vector $\mathbf{x}_n = [1, x_{n1}, x_{n2}, \dots, x_{nD}]^T \in \mathbb{R}^{D+1}$ (where the first element $1$ corresponds to the intercept $w_0$). The linear model assumes:
  

  $$
  y_n = \mathbf{x}_n^T \mathbf{w} + \epsilon_n, \quad \text{with } \epsilon_n \sim \mathcal{N}(0, \sigma^2)
  $$

* **Why predict the target variable** $\tilde{y} = \ln(\text{Price})$ **instead of the Original Price** $y$**?**

  1. **Resolving Violations of Homoscedasticity:** Under the MLE assumption in Chapter 9, the noise variance $\sigma^2$ must remain constant across all price levels. In reality, price variance scales proportionally with the price tier ($y_n = \exp(\mathbf{x}_n^T \mathbf{w}) \cdot \eta_n$). Taking the natural logarithm transforms multiplicative noise into additive noise with stable variance: $\tilde{y}_n = \ln(y_n) = \mathbf{x}_n^T \mathbf{w} + \ln(\eta_n)$.

  2. **Eliminating Physically Nonsensical Negative Predictions:** A linear hyperplane on raw prices can easily predict negative values for low-end cards (e.g., GT 730, GT 1030). With $\ln(y)$, the inverse transformation $\hat{y} = \exp(\mathbf{x}^T \mathbf{w})$ guarantees strictly positive price predictions.

  3. **Economic Interpretability of Weights** $w_j$**:** When the target is $\ln(y)$, the coefficient $w_j$ represents the percentage change in price ($w_j \times 100\%$) when feature $x_j$ increases by $1$ unit.

### 2. Task 1: Designing Matrix $\mathbf{X}$ & Preprocessing via Linear Algebra

When placing $N$ samples into the design matrix $\mathbf{X} \in \mathbb{R}^{N \times (D+1)}$, the quality of the solution $\mathbf{w}$ directly depends on the matrix rank $\text{rank}(\mathbf{X})$ and the condition number $\kappa(\mathbf{X}^T\mathbf{X})$.

#### 2.1. Feature Categorization and Normalization

| Feature Group | Typical VGA Specifications | Mathematical Issue if Left Unprocessed | Mandatory Preprocessing Method | 
| ----- | ----- | ----- | ----- | 
| **1. Core Quantitative** | VRAM ($\text{GB}$), Boost Clock ($\text{MHz}$), Core Count (CUDA / Stream), Memory Bus ($\text{bit}$), TDP ($\text{W}$) | Extreme scale disparities cause the Hessian matrix $\mathbf{H} = \frac{1}{N}\mathbf{X}^T\mathbf{X}$ to have a massive eigenvalue ratio $\frac{\lambda_{\max}}{\lambda_{\min}}$, creating elongated elliptical loss contours. | **Z-score Standardization:** $x_{nj}^{\text{std}} = \frac{x_{nj} - \mu_j}{\sigma_j}$ (computing $\mu_j, \sigma_j$ strictly on the Train set). Accelerates Gradient Descent convergence and allows direct comparison of $\lvert w_j \rvert$. | 
| **2. Technological Ordinal** | Memory Type (DDR4 $\to$ GDDR6X), PCIe Interface ($3.0 \to 5.0$) | Represented as strings, but possess a strict hierarchical order in terms of bandwidth and technology generation. | **Ordinal Encoding** based on transmission speed order, or combining indicator variables with theoretical bandwidth. | 
| **3. Categorical Nominal** | GPU Chipset Brand (NVIDIA, AMD, Intel), AIB Partner (ASUS, MSI, Gigabyte...), Cooling Type ($1, 2, 3$ fans) | Assigning integers $1, 2, 3$ misleads the linear model into assuming false quantitative ratios (e.g., $\text{Zotac} = 3 \times \text{ASUS}$). | **One-Hot Encoding:** For $K$ categories, generate strictly $K - 1$ binary columns to prevent matrix singularity. | 
| **4. Noise / Redundant** | Product SKU, Exact Physical Dimensions ($L \times W \times H\text{ mm}$), Standard HDMI/DP Port Counts | Inflates dimensionality $D$ without adding core pricing signal, causing **Overfitting** on small sample sizes. | Drop during feature selection or compress into a single binary/summary feature (e.g., `fan_count`, `is_flagship_size`). | 

#### 2.2. Explaining the "Dummy Variable Trap" via Linear Dependence

If all $3$ One-Hot columns for GPU brands are kept alongside the intercept column $\mathbf{x}_0 = \mathbf{1}$, a strict linear dependence arises:

$$
\mathbf{x}_{\text{NVIDIA}} + \mathbf{x}_{\text{AMD}} + \mathbf{x}_{\text{Intel}} = \mathbf{x}_0
$$


Consequence: $\text{rank}(\mathbf{X}) < D + 1 \implies \text{rank}(\mathbf{X}^T\mathbf{X}) < D + 1 \implies \det(\mathbf{X}^T\mathbf{X}) = 0$. The matrix $\mathbf{X}^T\mathbf{X}$ has at least one eigenvalue $\lambda = 0$ and becomes non-invertible (singular).

#### 2.3. Resolving Core Architecture Differences (Interaction Term)

$1$ NVIDIA CUDA Core and $1$ AMD Stream Processor (or Intel Xe Core) are not architecturally or economically equivalent. Merging them into a single `Cores` column skews the regression hyperplane.
* **Analytical Solution:** Introduce an interaction term $x_{\text{interaction}} = x_{\text{Cores}} \times \mathbb{I}_{\text{Brand=AMD}}$. The model remains linear with respect to parameters $\mathbf{w}$ while allowing the price slope per core to vary flexibly between NVIDIA and AMD.

#### 2.4. Diagnosing Multicollinearity via VIF (Variance Inflation Factor)

In VGA hardware, VRAM capacity, Bus Width, and recommended PSU/TDP scale almost simultaneously across product tiers. The Variance Inflation Factor for the $j$-th feature is computed as:

$$
\text{VIF}_j = \frac{1}{1 - R_j^2}
$$


where $R_j^2$ is the coefficient of determination when regressing column $\mathbf{x}_j$ against all remaining feature columns. Any feature with $\text{VIF}_j > 10$ provides empirical proof that $\mathbf{X}^T\mathbf{X}$ is ill-conditioned, directly motivating **Ridge Regression (MAP)** in Task 2.

### 3. Task 2: Objective Functions & 4 Optimization Methods

#### 3.1. Justifying the Objective Function Selection

1. **Mean Squared Error (MSE - Derived from MLE under Gaussian Noise):**
   

   $$
   \mathcal{L}_{\text{MSE}}(\mathbf{w}) = \frac{1}{2N} \Vert \mathbf{y} - \mathbf{X}\mathbf{w} \Vert_2^2
   $$

   * *Mathematical Advantage:* Infinitely differentiable ($C^\infty$) everywhere; the Hessian $\mathbf{H} = \frac{1}{N}\mathbf{X}^T\mathbf{X}$ is positive semi-definite, guaranteeing global convexity and a unique closed-form solution when $\mathbf{X}$ has full column rank.

   * *Drawback on VGA Data:* Because residuals are squared $(y_n - \hat{y}_n)^2$, extreme outliers (e.g., Workstation cards like RTX A4000/A6000 or limited editions priced $2\times$ higher than standard models) generate massive gradients that pull and distort the entire regression hyperplane.

2. **$L_2$-Regularized MSE - Tikhonov / Ridge Objective (Derived from MAP with Gaussian Prior** $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \tau^2\mathbf{I})$**):**
   

   $$
   \mathcal{L}_{\text{Ridge}}(\mathbf{w}) = \frac{1}{2N} \Vert \mathbf{y} - \mathbf{X}\mathbf{w} \Vert_2^2 + \frac{\alpha}{2} \Vert \mathbf{w} \Vert_2^2
   $$

   * *Mathematical Advantage:* Adding $\alpha\mathbf{I}$ to the Hessian makes the objective function **strictly convex** with all eigenvalues $\ge \alpha$, guaranteeing invertibility regardless of multicollinearity in $\mathbf{X}$.

#### 3.2. Comparison of 4 Optimization Methods in Linear Regression

| Optimization Method | Mathematical Formula / Update Rule | MML Textbook Reference | Performance on VGA Dataset | 
| ----- | ----- | ----- | ----- | 
| **1. Closed-Form OLS (Normal Equation)** | $\mathbf{w}_{\text{OLS}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ | Chap 3.8 (Orthogonal Projection) & Chap 9.2.1 (MLE) | Extremely fast since dimensionality $D$ is small ($20 - 30$ encoded columns). Highly susceptible to numerical instability if multicollinear features are kept or the Dummy Trap occurs. | 
| **2. SVD Pseudoinverse (Least Squares)** | $\mathbf{X} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T \implies \mathbf{w}_{\text{SVD}} = \mathbf{V}\mathbf{\Sigma}^+\mathbf{U}^T\mathbf{y}$ | Chap 4.5 (Singular Value Decomposition) | Always finds the minimum-norm solution $\Vert \mathbf{w} \Vert_2$ even when $\mathbf{X}^T\mathbf{X}$ is singular (by setting the reciprocal of singular values $\sigma_i \approx 0$ to $0$). Underlying mechanism of `Scikit-learn`. | 
| **3. Batch Gradient Descent (Iterative)** | $\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \frac{1}{N}\mathbf{X}^T(\mathbf{X}\mathbf{w}_t - \mathbf{y})$ | Chap 5.2 (Vector Calculus) & Chap 7.1 (Gradient Descent) | Without Z-score normalization, causes numerical overflow ($\text{NaN}$) unless $\eta \approx 10^{-9}$. When standardized, converges smoothly to $\mathbf{w}_{\text{OLS}}$ if $0 < \eta < \frac{2}{\lambda_{\max}}$. | 
| **4. Closed-Form MAP (Ridge Regression)** | $\mathbf{w}_{\text{MAP}} = (\mathbf{X}^T\mathbf{X} + \alpha\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$ | Chap 6.3 (Bayes) & Chap 9.2.2 (MAP Estimation) | Suppresses weight explosion among strongly correlated specs (VRAM, Bus Width, Core Count), significantly reducing Generalization Error on the Test set. | 

### 4. Task 3: Algorithm Evaluation, Residual Diagnostics & Runtime Analysis

#### 4.1. Dual-Scale Error Reporting & Jensen's Inequality Correction

* **Dual Evaluation Levels:** Evaluate mathematical convergence on the training space $\ln(\text{Price})$ ($\text{RMSE}_{\log}, R_{\log}^2$) and practical business accuracy on the original currency scale ($\text{VND}$).

* **Jensen's Inequality Correction:** Since $\tilde{y} \sim \mathcal{N}(\mu, \sigma^2)$, the actual price $y = \exp(\tilde{y})$ follows a Log-Normal distribution. By Jensen's Inequality for convex functions $\exp(\cdot)$, the expected actual price is:
  

  $$
  \mathbb{E}[y \mid \mathbf{x}] = \exp\left(\mu + \frac{\sigma^2}{2}\right)
  $$

  
  Multiplying by the correction factor $\exp\left(\frac{\hat{\sigma}^2}{2}\right)$ (where $\hat{\sigma}^2$ is the Train residual variance) prevents systematic underestimation in the high-end GPU segment.

* **4 Mandatory Evaluation Metrics:**

  1. **MAE (**$\text{Million VND}$**):** Average absolute monetary deviation per VGA.

  2. **RMSE (**$\text{Million VND}$**):** Heavily penalizes large prediction errors in the flagship segment.

  3. **MAPE (**$\%$**):** $\text{MAPE} = \frac{100\%}{N}\sum_{n=1}^N \left\lvert \frac{y_n - \hat{y}_n}{y_n} \right\rvert$ (the fairest metric across price tiers, as a $1\text{ Million VND}$ error is $2.5\%$ on a $40\text{M}$ card but $33.3\%$ on a $3\text{M}$ card).

  4. **Adjusted** $R^2$**:** Penalizes adding meaningless features that artificially expand the subspace $\text{Im}(\mathbf{X})$:
     

     $$
     R_{\text{adj}}^2 = 1 - \frac{(1 - R^2)(N - 1)}{N - D - 1}
     $$

#### 4.2. Model Diagnostics & Error Analysis

* **5-Fold Cross-Validation:** Report $\text{RMSE}_{\text{CV}} = \text{Mean} \pm \text{Std}$ on the Train set to mitigate lucky/unlucky $80/20$ splits on rare cards and to tune the hyperparameter $\alpha$ for Ridge.

* **Residual Plot (**$e_n = y_n - \hat{y}_n$ **vs.** $\hat{y}_n$**):** Place two plots side-by-side: Raw Price regression (cone/funnel-shaped spread violating homoscedasticity) vs. $\ln(\text{Price})$ regression (points distributed randomly in a uniform horizontal band around $0$).

* **Error Analysis on Top 5 Largest Outliers:**

  * *Underestimation (*$y_n \gg \hat{y}_n$*):* Professional Workstation cards (NVIDIA RTX Workstation/Quadro with certified drivers and ECC memory) or special limited editions (ROG Strix White, integrated AIO liquid cooling).

  * *Overestimation (*$y_n \ll \hat{y}_n$*):* Older-architecture clearance cards (GTX 16-series, RTX 20-series) that possess relatively high nominal VRAM/clock speeds but lack modern technologies (DLSS 3, Frame Generation), causing steep retail discounts.

#### 4.3. Computational Complexity (Theoretical Big-O vs. Empirical Runtime)

* **Normal Equation:** Costs $\mathcal{O}(ND^2)$ for $\mathbf{X}^T\mathbf{X}$ multiplication and $\mathcal{O}(D^3)$ for matrix inversion/Cholesky decomposition. With $N \approx 400, D \approx 25$, requires $\approx 400 \times 25^2 + 25^3 \approx 2.65 \times 10^5$ FLOPs ($\text{runtime} < 1\text{ ms}$).

* **SVD Decomposition:** Costs $\mathcal{O}(ND^2)$ with a constant factor $2 - 4\times$ larger than the Normal Equation to compute orthogonal matrices $\mathbf{U}, \mathbf{V}$, trading slight speed for absolute numerical stability.

* **Batch Gradient Descent:** Costs $\mathcal{O}(ND)$ per iteration, totaling $\mathcal{O}(KND)$ after $K$ epochs. At $K = 5000$, runs $20 - 50\times$ slower than closed-form solutions when $D$ is small; only becomes advantageous in time and memory when $D$ is massive (tens of thousands of features).

### 5. Tasks 4 & 5: Core Q&A Scenarios (Syllabus CQ42 $\to$ CQ49)

| MAI391 Syllabus Question | Core Mathematical Answer | Direct Empirical Link to VGA Pricing Project | 
| ----- | ----- | ----- | 
| **CQ42, CQ43:** Formulate univariate and multivariate linear regression in matrix form? | Express $\mathbf{y} = \mathbf{X}\mathbf{w} + \boldsymbol{\epsilon}$ with dimensions: $\mathbf{y}, \boldsymbol{\epsilon} \in \mathbb{R}^{N \times 1}$, $\mathbf{X} \in \mathbb{R}^{N \times (D+1)}$, $\mathbf{w} \in \mathbb{R}^{(D+1) \times 1}$. | *Univariate:* Predicting $\ln(\text{Price})$ solely from VRAM (2D line). *Multivariate:* Expanding into a hyperplane across a $D$-dimensional hardware specification space. | 
| **CQ44:** What are the fundamental assumptions of the linear regression model? | 1\. Linearity in parameters $\mathbf{w}$. 2. i.i.d. samples. 3. $\text{rank}(\mathbf{X}) = D + 1$ (no perfect multicollinearity). 4. Zero-mean noise $\mathbb{E}[\boldsymbol{\epsilon} \mid \mathbf{X}] = \mathbf{0}$ and constant variance $\text{Cov}(\boldsymbol{\epsilon}) = \sigma^2\mathbf{I}$. | Raw VGA data violates Assumption (3) due to multicollinearity (VRAM, Bus, Cores) and Assumption (4) due to heteroscedastic price variance at high tiers $\implies$ Resolved via $\ln(\text{Price})$, VIF filtering, and Ridge. | 
| **CQ45, CQ47:** Unbiased Estimator & Gauss-Markov Theorem? | Prove $\mathbb{E}[\mathbf{w}_{\text{OLS}}] = \mathbf{w}$. Gauss-Markov states that within the class of linear unbiased estimators, OLS achieves the minimum variance (BLUE). | Under severe multicollinearity, although OLS is BLUE, its variance becomes excessively large $\implies$ We trade a small amount of Bias via Ridge (MAP) to drastically reduce total Test MSE. | 
| **CQ46, CQ48:** Deriving estimators via OLS and Maximum Likelihood (MLE)? | Solve $\nabla_{\mathbf{w}} \Vert \mathbf{y} - \mathbf{X}\mathbf{w} \Vert_2^2 = \mathbf{0}$ and maximize likelihood $\prod_{n=1}^N \mathcal{N}(y_n \mid \mathbf{x}_n^T\mathbf{w}, \sigma^2)$. | The MLE noise variance estimator $\hat{\sigma}_{\text{MLE}}^2 = \frac{1}{N}\sum_{n=1}^N (y_n - \mathbf{x}_n^T\mathbf{w})^2$ corresponds directly to the training MSE of the VGA pricing model. | 
| **CQ49:** Core idea of Bayesian / MAP Linear Regression? | Treat weights $\mathbf{w}$ not as fixed constants, but as random variables governed by a prior distribution $p(\mathbf{w}) = \mathcal{N}(\mathbf{0}, \tau^2\mathbf{I})$. | Imposing a zero-centered Gaussian prior prevents the model from assigning extreme weights to any single hardware spec or AIB brand. | 

## II. Application Code Pipeline Explanation

The code pipeline is built from scratch using `NumPy` matrix algebra on VGA pricing data collected over a recent $3$-week window, cross-validated against `scikit-learn`, and configured to automatically export $6$ analytical plots:

1. **Imputing Sparse and Heterogeneous Matrices (**$963\text{ columns} \to 24\text{ standard features}$**):**

   * Instead of calling `dropna()` and losing valuable samples, `load_and_preprocess_data()` constructs a median reference lookup table from $3,762$ standardized *BuildCores* records indexed by `(Clean_Chipset, VRAM_GB)`.

   * Accurately reconstructs missing physical specifications (`Bus_Bit`, `Core_Count`) for retail listings scraped from *sp-one.vn* and *phongvu.vn*.

2. **Log-Linear Transformation & Jensen's Inequality Correction (`predict_vnd()`):**

   * Applies the mathematical expectation of the Log-Normal distribution by multiplying predictions by the correction factor $\exp\left(\frac{1}{2}\hat{\sigma}_{\text{MLE}}^2\right)$, where $\hat{\sigma}_{\text{MLE}}^2 = \frac{1}{N - D}\Vert \mathbf{y} - \mathbf{X}\mathbf{w} \Vert_2^2$, eliminating systematic underprediction on flagship RTX 4090 / RTX 5090 cards.

3. **Unregularized Intercept** $w_0$ **in MAP / Ridge Estimation (`I_0[0, 0] = 0.0`):**

   * Inside `MML_LinearRegression_Scratch` (`ridge_map` branch), the top-left element of the identity matrix $\mathbf{I}_0$ is explicitly set to $0$ before adding to $\mathbf{X}^T\mathbf{X}$.

   * Ensures the Gaussian prior $p(\mathbf{w}) = \mathcal{N}(\mathbf{0}, \tau^2\mathbf{I})$ only shrinks feature slopes toward $0$ without forcing the regression hyperplane through the origin (which would distort the market-wide average VGA price). This guarantees an exact mathematical match with `sklearn.linear_model.Ridge(fit_intercept=True)`.

4. **Verifying Mathematical Equivalence Among OLS, SVD, and Gradient Descent:**

   * Prints the $L_2$ norm of the weight difference vector $\Vert \mathbf{w}_{\text{SVD\_Scratch}} - \mathbf{w}_{\text{Sklearn}} \Vert_2 \approx 10^{-14}$ (effectively $0$ within $64\text{-bit}$ floating-point precision).

   * Exports `Slide3_Gradient_Descent_Convergence.png`, demonstrating that the Batch Gradient Descent loss curve asymptotically converges to the exact closed-form OLS baseline.

5. **Empirical Multicollinearity Demonstration via Transformed Features:**

   * Deliberately includes both raw and log-transformed feature pairs (`VRAM_GB` & `Log_VRAM`, `Core_Count` & `Log_Cores`) in matrix $\mathbf{X}$, driving up VIF scores and the condition number $\kappa(\mathbf{X}^T\mathbf{X})$.

   * Creates an ideal empirical scenario for Slides 2 and 4 to illustrate why Normal Equation weights fluctuate wildly under collinearity and how MAP estimation (Ridge Regression) stabilizes the entire weight vector $\mathbf{w}$.