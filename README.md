# Simulating-the-2026-FIFA-World-Cup-with-Regularized-XGBoost


Predicting an international tournament is incredibly volatile. With the upcoming expansion to a massive 48-team format, the 2026 FIFA World Cup introduces an unprecedented layer of variance and operational complexity. 

**CupPredictor-26** is a machine learning pipeline built to simulate the entire tournament—from initial group match dynamics to the final trophy presentation—by treating match outcomes not just as arbitrary wins or losses, but as continuous scoreline distributions.

---

##  Datasets Used

The predictive model relies on the consolidation of three distinct datasets:
* `results.csv`: Decades of historical international match logs to track long-term team performance, historic head-to-heads, and tournament competitive form.
* `male_players.csv`: Granular individual player attributes aggregated to the national squad level, establishing a realistic, up-to-date baseline of team depth and squad quality.
* `FIFA2026_schedule_Fixtures_Finalized.csv`: The official 48-team tournament structure used to programmatically map out group placements and knockout paths.

---

##  Training & Optimization Strategy

Unlike standard classification setups that simply predict a Win/Loss/Draw outcome, this architecture models exact team goal tallies (`Team A Score` and `Team B Score`) using parallel regression paths. This approach is vital for tournament modeling where goal differences directly dictate group stage progression.

### Model Evaluation & Hyperparameter Tuning
We established a clear baseline and iteratively drove down validation errors:

* **Baseline Model (Decision Tree Regressor):** Established a performance floor with a Total Mean Absolute Error (MAE) of **2.1194**.
* **Final Model (Regularized XGBoost):** To control variance and mitigate historical bias (e.g., favoring traditional giants who may currently have weaker squads), an XGBoost framework was deployed. 

Using `RandomizedSearchCV` across 50 distinct parameter configurations with **5-fold cross-validation**, the optimized boosting pipeline achieved a superior Total MAE of **2.0258**. The model converged on a conservative learning rate (`~0.019`) combined with strict $L_1$ and $L_2$ regularization (`reg_alpha` and `reg_lambda`) to ensure predictions heavily penalize historic reputation in favor of actual player performance biometrics.

---

##  Simulation Architecture & Tournament Logic

Once the regression models are trained, the pipeline feeds predicted score distributions into a custom tournament simulator that handles the official FIFA framework:

1.  **Group Stage Accumulation:** Simulates all group stage fixtures, tallies points ($3$ for a win, $1$ for a draw), and applies tie-breaker logic to determine group winners and runners-up.
2.  **Knockout Phase Routing:** Programmatically seeds the Round of 16, Quarter-finals, Semi-finals, and Final according to the official bracket parameters.
3.  **Extra Time & Penalty Shootouts:** For knockout ties, the simulator invokes an algorithmic penalty shootout model ($5\text{-}4$ baseline distribution) to guarantee a stable progression vector.

---

##  How to Run the Project

### Prerequisites
Ensure you have the following libraries installed:
```bash
pip install numpy pandas scikit-learn xgboost
