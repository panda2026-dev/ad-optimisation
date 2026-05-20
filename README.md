# Ad Click-Through Rate (CTR) Optimization using Wide & Deep Learning

This repository features a custom Wide & Deep Neural Network implemented in PyTorch designed for digital advertising platforms to maximize conversion performance while reducing wasted ad spend. The pipeline balances low-order feature memorization with high-order feature generalization to predict user click-through probabilities.

---

## Project Overview

In digital marketing, delivering the right advertisement to the right audience segment in real-time determines structural profitability. This project simulates an ad-tech analytics platform that processes multi-dimensional logs (user demographics, continuous activity metrics, platform channels, and context signals) to predict binary conversions (Click vs. No Click).

By computing precise click probabilities, the platform allows real-time bidding architectures to selectively trigger ads for High-Value Visits (Probability > 70%), effectively minimizing baseline programmatic bounce rates.

### Core Architecture Concepts
* **Memorization (Wide Component):** Tracks historical direct feature impacts—such as specific ad slots or distinct platform channels—to capture static business rules.
* **Generalization (Deep Component):** Captures multi-layered, non-linear feature interactions across dense continuous metrics and sparse interest clusters using deep hidden layers.

---

## Dataset Structure & Feature Mapping

The pipeline streams a high-dimensional source array via scikit-learn's `fetch_covtype` dataset containing 50,000 sampled observations as a processing proxy for large-scale, anonymized digital interaction logs.

The system organizes 54 feature columns into two structural pipelines:

### 1. Continuous Dense Features (10 Dimensions)
Standardized via `StandardScaler` to handle divergent scaling factors:
* Ad_Slot_Height / Ad_Slot_Width
* Vertical_Scroll_Depth (User engagement indicator)
* Morning_Traffic_Volume
* Historical_User_Engagement_Score

### 2. Categorical/Binary Sparse Features (44 Dimensions)
Binary indicators representing architectural channels and demographic segments:
* **Platform Channels (4 features):** Desktop_Web, Mobile_App, Tablet_UI, etc.
* **User Demographics & Interests (40 features):** Distinct profile clusters including Tech_Enthusiast, Fashion_Buyer, Gamers, Automotive_Inquirer, etc.

---

## Model Architecture

The deep learning system implements a custom, dual-channel `WideAndDeep` class extending `torch.nn.Module`. 

The Wide component handles linear memorization of sparse inputs, while the Deep component handles non-linear generalization by combining dense and sparse components into a deep feedforward path.

### Architectural Sub-Components
1. **Wide Channel:** Single `nn.Linear(sparse_dim, 1)` mapping directly from categorical signals.
2. **Deep Channel:** Feedforward sequence processing concatenated global dimensions:
   Linear(54 to 64) -> ReLU -> Dropout(p=0.2) -> Linear(64 to 32) -> ReLU -> Linear(32 to 1)
3. **Activation Framework:** Joint wide and deep linear outputs are summed and passed through a final `Sigmoid` activation layer to scale values within a realistic (0, 1) probability range.

---

## Technical Requirements & Setup

### Requirements
* Python 3.8+
* PyTorch
* Scikit-Learn
* Pandas
* Numpy
* Matplotlib
* Seaborn

### Installation
Clone this repository and install the dependencies:
```bash
git clone [https://github.com/your-username/ad-ctr-optimization-wide-deep.git](https://github.com/your-username/ad-ctr-optimization-wide-deep.git)
cd ad-ctr-optimization-wide-deep
pip install torch pandas numpy scikit-learn matplotlib seaborn datasets

