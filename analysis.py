import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# =========================
# 1. Load & Clean Data
# =========================
df = pd.read_csv("students.csv")

df = df.dropna()
df = df.drop_duplicates()

# =========================
# 2. Features & Target
# =========================
X = df.drop("GradeClass", axis=1)
y = df["GradeClass"]

# =========================
# 3. Train/Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 4. Model Comparison
# =========================
models = {
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier()
}

print("Model Comparison Results:")
for name, m in models.items():
    m.fit(X_train, y_train)
    preds = m.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name}: {acc:.4f}")

# =========================
# 5. Final Model (Random Forest)
# =========================
model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Accuracy
print("\nFinal Model Accuracy:", accuracy_score(y_test, y_pred))

# =========================
# 6. Confusion Matrix
# =========================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =========================
# 7. Feature Importance
# =========================
importances = model.feature_importances_

plt.figure(figsize=(10,5))
plt.barh(X.columns, importances)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.show()

# =========================
# 8. Cross Validation
# =========================
scores = cross_val_score(model, X, y, cv=5)

print("\nCross Validation Scores:", scores)
print("Average CV Score:", scores.mean())

# =========================
# 9. Hyperparameter Tuning
# =========================
params = {
    "n_estimators": [50, 100],
    "max_depth": [None, 10, 20]
}

grid = GridSearchCV(RandomForestClassifier(), params, cv=3)
grid.fit(X_train, y_train)

print("\nBest Parameters:", grid.best_params_)
