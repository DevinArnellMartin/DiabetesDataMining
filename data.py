import sys
import os
sys.path.insert(0, os.path.abspath('F:/PythonPackages'))
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,f1_score
from ucimlrepo import fetch_ucirepo  # type: ignore Sys.Path Insert Fixes Warning
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE, RandomOverSampler  # type: ignore Sys.Path Insert Fixes Warning
from imblearn.under_sampling import RandomUnderSampler  # type: ignore Sys.Path Insert Fixes Warning
from imblearn.combine import SMOTEENN # type: ignore Sys.Path Insert Fixes Warning

cdc_diabetes_health_indicators = fetch_ucirepo(id=891)


X = cdc_diabetes_health_indicators.data.features
Y = cdc_diabetes_health_indicators.data.targets  

target_names = ['Healthy', 'Pre-Diabetic/Diabetic']



#Accuracy is not good for initialliy imbalanced datasets


plt.hist(Y, bins=np.arange(len(np.unique(Y)) + 1) - 0.5, align='left', rwidth=0.8)

plt.xticks(ticks=np.arange(len(target_names)), labels=target_names)
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.title('Class Distribution')
plt.show()
plt.show()

correlation_matrix = X.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, vmin=-1, vmax=1, annot=True, cmap='coolwarm')
plt.title('All:Correlation Heatmap', fontdict={'fontsize': 18}, pad=12)
plt.show()

X_clean = X.dropna() #.drop_duplicates().reset_index(drop=True)
Y_clean = Y.dropna() 



# X_clean = X_clean[]
# Y_clean = Y_clean[]

scaler =  StandardScaler() # Using because not weak to outliers & also more independent of size of the data
X_standardized = scaler.fit_transform(X_clean)


#KEEP if cannot process all instances at same time ;else drop
X_clean, _, Y_clean, _ = train_test_split(
    X_clean, Y_clean, test_size=0.95, stratify=Y_clean, random_state=42
)

trainX, testX, trainY, testY = train_test_split(
    X_clean, Y_clean, test_size=0.3, stratify=Y_clean, random_state=42
)

models = {
    "Support Vector Machine:Linear Kernel": SVC(kernel="linear", probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=1000, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=1000, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
}

accuracies = []

def test_model(name, model, balance=False): 
    """Balance : class weight for cost_sensitive learning"""
    if balance and hasattr(model, "class_weight"):
        model.set_params(class_weight="balanced")
    
    model.fit(trainX, trainY)
    cv_scores = cross_val_score(model, trainX, trainY, cv=5)
    print(f"Cross-Validation Accuracy Scores ({name}): {cv_scores}")
    print(f"Mean Cross-Validation Accuracy ({name}): {cv_scores.mean()}")
    train_prediction = model.predict(trainX)
    test_prediction = model.predict(testX)
    
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(testX)[:, 1]
        print(f"{name}'s ROC-AUC Score:", roc_auc_score(testY, probas))
    else:
        print(f"{name} : NA Prediction")

    # Accuracy
    training_accuracy = accuracy_score(trainY, train_prediction)
    testing_accuracy = accuracy_score(testY, test_prediction)
    accuracies.append(testing_accuracy)
    
    print(f"{name} Training Accuracy: {training_accuracy}")
    print(f"{name} Testing Accuracy: {testing_accuracy}")
    

    conf_matrix = confusion_matrix(testY, test_prediction)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Reds")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    
    if hasattr(model, "feature_importances_"):
        feature_importances = model.feature_importances_
        feature_names = X_clean.columns  
        plt.figure(figsize=(10, 6))
        sns.barplot(x=feature_importances, y=feature_names, orient="h", palette="viridis")
        plt.title(f"{name} Feature Importances")
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.show()

for model_name, model_instance in models.items():
    test_model(model_name, model_instance, balance=True)

smote_enn = SMOTEENN(random_state=42)
X_resampled, Y_resampled = smote_enn.fit_resample(trainX, trainY)


def test_model_advSampling(name, model):
    print(f"\nTesting model: {name}")
    model.fit(X_resampled, Y_resampled)
    cv_scores = cross_val_score(model, X_resampled, Y_resampled, cv=5)
    print(f"Cross-Validation Accuracy Scores ({name}): {cv_scores}")
    print(f"Mean Cross-Validation Accuracy ({name}): {cv_scores.mean()}")
    train_prediction = model.predict(trainX)
    test_prediction = model.predict(testX)
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(testX)[:, 1]
        print(f"{name}'s ROC-AUC Score:", roc_auc_score(testY, probas))
    else:
        print(f"{name} does not support probability predictions.")
    training_accuracy = accuracy_score(trainY, train_prediction)
    testing_accuracy = accuracy_score(testY, test_prediction)
    accuracies.append(testing_accuracy)

    print(f"{name} Training Accuracy: {training_accuracy}")
    print(f"{name} Testing Accuracy: {testing_accuracy}")
    conf_matrix = confusion_matrix(testY, test_prediction)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Reds")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    
    if hasattr(model, "feature_importances_"):
        feature_importances = model.feature_importances_
        feature_names = X_clean.columns  
        plt.figure(figsize=(10, 8))
        sns.barplot(x=feature_importances, y=feature_names, orient="h", palette="viridis")
        plt.title(f"{name} Feature Importances")
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.show()



for model_name, model_instance in models.items():
    test_model_advSampling(model_name, model_instance)


# With SMOTE 

# Cross-Validation Accuracy Scores (Support Vector Machine:Linear Kernel): [0.81761006 0.86118598 0.868823   0.87191011 0.88      ]
# Mean Cross-Validation Accuracy (Support Vector Machine:Linear Kernel): 0.8599058319957196
# Support Vector Machine:Linear Kernel's ROC-AUC Score: 0.7832077775474002
# Support Vector Machine:Linear Kernel Training Accuracy: 0.621423744086506
# Support Vector Machine:Linear Kernel Testing Accuracy: 0.6103520756699947


# Cross-Validation Accuracy Scores (Random Forest): [0.90251572 0.9442947  0.94833783 0.95910112 0.95460674]
# Mean Cross-Validation Accuracy (Random Forest): 0.941771222629395
# Random Forest's ROC-AUC Score: 0.7754181353237959
# Random Forest Training Accuracy: 0.7728091912592926
# Random Forest Testing Accuracy: 0.6862848134524435

# Cross-Validation Accuracy Scores (Gradient Boosting): [0.84950584 0.92138365 0.93351303 0.93977528 0.94382022]
# Mean Cross-Validation Accuracy (Gradient Boosting): 0.9175996042682495
# Gradient Boosting's ROC-AUC Score: 0.7560197663971249
# Gradient Boosting Training Accuracy: 0.7140121649020049
# Gradient Boosting Testing Accuracy: 0.6576458223857068


# Cross-Validation Accuracy Scores (Logistic Regression): [0.81176999 0.86028751 0.86837376 0.87685393 0.88044944]
# Mean Cross-Validation Accuracy (Logistic Regression): 0.8595469275265757
# Logistic Regression's ROC-AUC Score: 0.7790511899002465
# Logistic Regression Training Accuracy: 0.6252534354584366
# Logistic Regression Testing Accuracy: 0.6169206516027326

# Using Balanced Class 

# Cross-Validation Accuracy Scores (Support Vector Machine:Linear Kernel): [0.70777027 0.70326577 0.7134009  0.71549296 0.71887324]
# Mean Cross-Validation Accuracy (Support Vector Machine:Linear Kernel): 0.7117606268240071
# Support Vector Machine:Linear Kernel's ROC-AUC Score: 0.8124513327343518
# Support Vector Machine:Linear Kernel Training Accuracy: 0.7132236990313133
# Support Vector Machine:Linear Kernel Testing Accuracy: 0.7088807146610615

# Cross-Validation Accuracy Scores (Random Forest): [0.85641892 0.8597973  0.85810811 0.86253521 0.85464789]
# Mean Cross-Validation Accuracy (Random Forest): 0.8583014845831748
# Random Forest's ROC-AUC Score: 0.8001693275278181
# Random Forest Training Accuracy: 0.9995494480738906
# Random Forest Testing Accuracy: 0.8641618497109826


# Cross-Validation Accuracy Scores (Gradient Boosting): [0.85135135 0.84290541 0.85191441 0.85295775 0.84901408]
# Mean Cross-Validation Accuracy (Gradient Boosting): 0.8496286004314173
# Gradient Boosting's ROC-AUC Score: 0.8011202110258715
# Gradient Boosting Training Accuracy: 0.90831268303672
# Gradient Boosting Testing Accuracy: 0.8552285864424592

# Cross-Validation Accuracy Scores (Logistic Regression): [0.72635135 0.72635135 0.73648649 0.73690141 0.73690141]
# Mean Cross-Validation Accuracy (Logistic Regression): 0.7325984012181195
# Logistic Regression's ROC-AUC Score: 0.8118483193954892
# Logistic Regression Training Accuracy: 0.7342870015769317
# Logistic Regression Testing Accuracy: 0.7312138728323699
