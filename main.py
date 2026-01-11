import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor

# -------------------------
# Load Dataset
# -------------------------
df = pd.read_csv(r"E:/ALL AI Course Files/research paper/data/student-mat.csv", sep=";")

# Encode categorical features
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

# Feature Engineering
df['avg_prev_grades'] = (df['G1'] + df['G2']) / 2
df['study_efficiency'] = df['studytime'] / (df['absences'] + 1)
df['health_factor'] = df['health'] * df['Walc']

X = df.drop("G3", axis=1)
y = df["G3"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Baseline Models
lr = LinearRegression().fit(X_train, y_train)
dt = DecisionTreeRegressor(random_state=42).fit(X_train, y_train)

# Random Forest Hyperparameter Tuning
rf = RandomForestRegressor(random_state=42)
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}
rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=rkf, n_jobs=-1, scoring='r2')
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

# Gradient Boosting
gbr = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42).fit(X_train, y_train)

# Voting Regressor Ensemble
voting = VotingRegressor([('lr', lr), ('rf', best_rf), ('gbr', gbr)]).fit(X_train, y_train)

# Predictions
predictions = {
    "Linear Regression": lr.predict(X_test),
    "Decision Tree": dt.predict(X_test),
    "Random Forest Tuned": best_rf.predict(X_test),
    "Gradient Boosting": gbr.predict(X_test),
    "Voting Regressor": voting.predict(X_test)
}

# -------------------------
# 1. Final Comparison Table
# -------------------------
results = pd.DataFrame({
    "Model": list(predictions.keys()),
    "RMSE": [np.sqrt(mean_squared_error(y_test, pred)) for pred in predictions.values()],
    "R2": [r2_score(y_test, pred) for pred in predictions.values()]
})
print("\nFinal Model Comparison:\n", results)

# -------------------------
# 2. Random Forest Feature Importance Plot
# -------------------------
importances = best_rf.feature_importances_
feat_names = X.columns
feat_df = pd.DataFrame({"Feature": feat_names, "Importance": importances}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x="Importance", y="Feature", data=feat_df, palette="viridis")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig("Feature_Importance.png", dpi=300)
plt.show()

# -------------------------
# 3. Predicted vs Actual (Voting Regressor)
# -------------------------
plt.figure(figsize=(6,6))
plt.scatter(y_test, predictions["Voting Regressor"], alpha=0.7, color="blue")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
plt.xlabel("Actual Grades")
plt.ylabel("Predicted Grades")
plt.title("Predicted vs Actual (Voting Regressor)")
plt.tight_layout()
plt.savefig("Predicted_vs_Actual.png", dpi=300)
plt.show()

# -------------------------
# 4. Save Results to CSV (Optional)
# -------------------------
results.to_csv("Model_Comparison.csv", index=False)

print("\n✅ Graphs saved: Feature_Importance.png & Predicted_vs_Actual.png")
print("✅ Model comparison saved: Model_Comparison.csv")
