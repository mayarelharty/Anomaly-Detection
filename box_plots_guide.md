# Complete Guide to Box Plots & Outlier Detection in OEE Analysis

---

## 1. What is a Box Plot?

A **Box Plot** (also called a **Box-and-Whisker Plot**, invented by statistician John Tukey in 1977) is a standardized, visual way of displaying the distribution of data based on a **five-number summary**:

1. **Minimum** (smallest non-outlier value)
2. **First Quartile ($Q_1$ / 25th percentile)**: 25% of the data falls below this point.
3. **Median ($Q_2$ / 50th percentile)**: The exact middle value of the data.
4. **Third Quartile ($Q_3$ / 75th percentile)**: 75% of the data falls below this point.
5. **Maximum** (largest non-outlier value)

---

## 2. Anatomy of a Box Plot

```text
       Extreme Outlier     Mild Outlier                                                    Mild Outlier
             *                   *      |---------|===================|---------|            *
                                      Lower      Q1      Median      Q3       Upper
                                      Fence     (25%)     (50%)     (75%)     Fence
                                                |<------- IQR ------->|
                                      |<--------- 1.5 × IQR --------->|
```

### Key Components Explained:
* **The Box**: Spans from $Q_1$ to $Q_3$. This represents the **middle 50%** of your dataset.
* **The Center Line**: The **Median**. If it is not in the exact center of the box, the data is skewed.
* **IQR (Interquartile Range)**: The height/width of the box:
  $$\text{IQR} = Q_3 - Q_1$$
* **The Whiskers**: Lines extending from the box to the farthest data points that are still within the **"fences"** ($1.5 \times \text{IQR}$).
* **The Dots (Outliers)**: Any data points falling outside the fences are plotted individually as points.

---

## 3. The Mathematical Rule for Outliers (Tukey's Fences)

Instead of guessing which numbers are abnormal, a box plot uses strict mathematical boundaries called **fences**:

### Step 1: Calculate the Interquartile Range (IQR)
$$\text{IQR} = Q_3 - Q_1$$

### Step 2: Calculate the Fences
* **Lower Fence** = $Q_1 - (1.5 \times \text{IQR})$
* **Upper Fence** = $Q_3 + (1.5 \times \text{IQR})$

### Step 3: Classify Outliers
* **Mild Outlier**: Any point lying outside the $1.5 \times \text{IQR}$ fences:
  $$\text{Value} < Q_1 - 1.5 \times \text{IQR} \quad \text{OR} \quad \text{Value} > Q_3 + 1.5 \times \text{IQR}$$
* **Extreme Outlier**: Any point lying outside the $3.0 \times \text{IQR}$ fences:
  $$\text{Value} < Q_1 - 3.0 \times \text{IQR} \quad \text{OR} \quad \text{Value} > Q_3 + 3.0 \times \text{IQR}$$

---

## 4. Box Plot (IQR) vs. Standard Deviation (Z-Score): Are They the Same?

> **Short Answer:** Both are statistical methods, but they are **fundamentally different in their mathematics and assumptions**.

| Feature | Standard Deviation Method ($\mu \pm 3\sigma$) | Box Plot Method ($\text{IQR}$ / Tukey) |
| :--- | :--- | :--- |
| **Statistical Family** | **Parametric** (assumes normal distribution) | **Non-Parametric** (distribution-free) |
| **Center Measurement** | **Mean ($\mu$)** | **Median ($Q_2$)** |
| **Spread Measurement** | **Standard Deviation ($\sigma$)** | **Interquartile Range ($\text{IQR} = Q_3 - Q_1$)** |
| **Outlier Threshold** | Typically $\mu \pm 2\sigma$ or $\mu \pm 3\sigma$ | $Q_1 - 1.5 \times \text{IQR}$ and $Q_3 + 1.5 \times \text{IQR}$ |
| **Sensitivity to Outliers** | ⚠️ **Very High** (a single extreme zero pulls the mean down and inflates $\sigma$) |  **Robust** (median and quartiles are not affected by extreme values) |
| **Best Used When** | Data is symmetric and bell-shaped | Data is skewed, bounded (0% to 100%), or has extreme spikes |

### Why Academic Evaluators & Doctors Prefer Box Plots for Factory Data:
Real factory data (like Availability, Performance, Quality, and OEE) is **never a perfect bell curve**:
* Quality is naturally clustered near $95\% - 99\%$, with a long tail of rare, catastrophic breakdown shifts reaching down to $0\%$.
* In a skewed distribution, the **Mean** and **Standard Deviation** get distorted by the very outliers you are trying to find!
* The **Box Plot** uses ranks/percentiles, making it completely immune to this distortion.

---

## 5. Real Outlier Findings in Our Dataset

Using our cleaned, validated dataset ($N = 1,921$ shifts), here are the exact five-number summaries and outlier counts:

| Metric | $Q_1$ (25%) | Median (50%) | $Q_3$ (75%) | $\text{IQR}$ | Lower Fence ($1.5 \times \text{IQR}$) | Upper Fence ($1.5 \times \text{IQR}$) | Mild Outliers ($1.5 \times \text{IQR}$) | Extreme Outliers ($3.0 \times \text{IQR}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Availability (A)** | 0.4558 | 0.5893 | 0.7032 | 0.2475 | **0.0846** | 1.0744 | **5 shifts** (0.26%) | **0 shifts** (0.0%) |
| **Performance (P)** | 0.7012 | 0.8557 | 0.9331 | 0.2319 | **0.3533** | 1.2810 | **10 shifts** (0.52%) | **0 shifts** (0.0%) |
| **Quality (Q)** | 0.9000 | 0.9358 | 0.9620 | 0.0620 | **0.8070** | 1.0550 | **109 shifts** (5.67%) | **47 shifts** (2.45%) |
| **OEE** | 0.3356 | 0.4463 | 0.5526 | 0.2171 | **0.0100** | 0.8782 | **3 shifts** (0.16%) | **0 shifts** (0.0%) |

*(Generated plot saved at `outputs/phase2_anomaly/boxplot_outliers.png`)*

---

## 6. Key Industrial Engineering Insights from These Box Plots

### 1. Quality ($Q$) is the Most Volatile Dimension
* Availability and Performance have very few statistical outliers (only 5 and 10 shifts respectively).
* However, **Quality has 109 outliers (5.67%)**, of which **47 shifts are extreme outliers** ($Q < 0.7140$, or less than 71.4% good parts).
* **Engineering Meaning:** In this manufacturing facility, machines usually run at expected speeds and availability, but when an anomaly occurs, it manifests predominantly as a **scrap/defect crisis** rather than just a minor speed dip.

### 2. Physical Upper Limits (100% Ceilings)
* Notice that the theoretical Upper Fences for $A$, $P$, and $Q$ calculate to numbers like $1.0744$ or $1.2810$.
* Since efficiency cannot physically exceed $1.00$ ($100\%$), **all real outliers in manufacturing are one-sided (Lower Tail / Performance Drops)**.

---

## 7. How This Connects to Phase 2 Anomaly Detection

In Phase 2 of our project, we discussed how to set the threshold for training our AI:

1. **Benchmark Mode ($A < 90\%, P < 95\%, Q < 99.9\%$)**:
   * Completely unrealistic. Looking at the box plot, the 75th percentile for Availability is only $70.3\%$. Asking for $90\%$ means almost 100% of the plant's history is labeled as an anomaly.
2. **Train Percentile Mode (Bottom 10%)**:
   * Captures the bottom tail of historical performance.
3. **Box Plot Outlier Mode ($Q_1 - 1.5 \times \text{IQR}$)**:
   * Sets strict mathematical statistical boundaries ($A < 8.5\%$, $P < 35.3\%$, $Q < 80.7\%$). This labels only the truly severe anomalies (the "red dots" on the chart).

---

## 8. Presentation Script for Your Doctor / Professor

If your doctor asks you to present the box plot results, here is a concise explanation:

> *"We generated box plots for the four core dimensions: Availability, Performance, Quality, and OEE. Because manufacturing data is bounded between 0 and 1 and exhibits significant skewness, we utilized Tukey's non-parametric Interquartile Range (IQR) method rather than parametric Z-scores to prevent outlier distortion.*
> 
> *Our analysis revealed that Availability, Performance, and OEE operate within consistent statistical fences, yielding less than 0.5% outliers. In contrast, Quality exhibited 109 statistical outliers (5.67%) with a lower fence at 80.7%, including 47 extreme outliers below 71.4%. This quantitatively proves that process variance in this facility is heavily driven by defect spikes rather than downtime alone."*
