# Student Performance Prediction 🤖📊

## 📌 Objective
The goal of this project is to predict student grade classification using machine learning techniques based on academic, demographic, and behavioral features.

---

## ⚠️ Problem: Overfitting
During model development, we observed that some models were overfitting the training data, especially Decision Tree and an unconstrained Random Forest.

This was identified by:
- Very high training accuracy compared to test accuracy
- Instability in cross-validation results
- Models performing too perfectly on training data

---

## 🔧 Solutions Applied to Reduce Overfitting

### 1. Model Regularization (Random Forest Tuning)
We controlled model complexity using:
- max_depth
- min_samples_split
- min_samples_leaf

This helped the model generalize better instead of memorizing the training data.

---

### 2. Cross Validation
We used 5-fold cross-validation to evaluate model stability across different data splits, ensuring better generalization.

---

### 3. Hyperparameter Tuning
We applied GridSearchCV to find the optimal parameters for Random Forest and improve performance without overfitting.

---

## 🌳 Why Decision Tree Was Kept
Decision Tree was included as a baseline model for comparison purposes.

Although it achieved very high accuracy, it is highly prone to overfitting because:
- It can fully memorize training data
- It does not generalize well on unseen data

Therefore, it is useful for comparison but not selected as the final model.

---

## 🛠️ Tools & Technologies
- Python
- Pandas
- Matplotlib / Seaborn
- Scikit-learn

---

## ⚙️ Project Workflow
- Data Cleaning (handling missing values & duplicates)
- Exploratory Data Analysis (EDA)
- Data Visualization
- Feature Selection
- Model Training

---

## 🤖 Models Used
- Random Forest Classifier (Final Model)
- Logistic Regression
- Decision Tree (Baseline Model)

---

## 📊 Model Evaluation
- Accuracy Score
- Confusion Matrix
- Cross Validation

---

## 📈 Results
- Random Forest achieved the best balance between accuracy and generalization
- Cross-validation improved model reliability and reduced overfitting

---

## 🔍 Key Insights
- Study time has a strong positive impact on performance
- High absences negatively affect student grades
- Parental support contributes significantly to better outcomes

---

## 🚀 Future Improvements
- Try advanced models (XGBoost, LightGBM)
- Perform feature engineering
- Deploy the model using Streamlit or Flask

---

## 🧾 Conclusion

This project demonstrates a complete machine learning pipeline for predicting student performance, starting from data cleaning and exploration to model building, evaluation, and optimization.

We identified and addressed overfitting issues by controlling model complexity, applying cross-validation, and tuning hyperparameters.

The final Random Forest model provided the best balance between accuracy and generalization, making it the most reliable choice for this problem.

Overall, this project highlights the importance of not only achieving high accuracy but also ensuring that the model generalizes well to unseen data.

---

## 👤 Author
Yousef Badr  
Data Science Postgraduate Student
