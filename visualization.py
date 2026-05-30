import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
os.makedirs('visualizations', exist_ok=True)


X_test_pca = np.load('preprocessed_data/X_test_pca.npy')
Y_test = np.load('preprocessed_data/Y_test.npy')
label_encoder = joblib.load('preprocessed_data/label_encoder.pkl')

rf_classifier = joblib.load('models/rf.pkl')
dt_classifier = joblib.load('models/dt.pkl')
xgb_classifier = joblib.load('models/xgb.pkl')
voting_classifier = joblib.load('models/voting.pkl')
#taking predictions for all models
Y_pred_rf = rf_classifier.predict(X_test_pca)
Y_pred_dt = dt_classifier.predict(X_test_pca)
Y_pred_xgb = xgb_classifier.predict(X_test_pca)
Y_pred_voting = voting_classifier.predict(X_test_pca)

#confusion matrix
cm = confusion_matrix(Y_test, Y_pred_voting)
plt.figure(figsize=(14, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix — Voting Classifier', fontsize=16)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('visualizations/confusion_matrix.png', dpi=150)
plt.close()
print("Confusion matrix saved")

#feature importance 
importances = rf_classifier.feature_importances_
indices = np.argsort(importances)[::-1][:15]  # top 15 features
plt.figure(figsize=(10, 6))
plt.barh(range(15), importances[indices][::-1], color='steelblue')
plt.yticks(range(15), [f'PCA Component {indices[::-1][i]+1}' for i in range(15)])
plt.title('Top 15 Feature Importances — Random Forest', fontsize=16)
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('visualizations/feature_importance.png', dpi=150)
plt.close()
print("Feature importance saved")
#model comparison bar graph 
models = ['Decision Tree', 'Random Forest', 'XGBoost', 'Voting Classifier']
accuracies = [
    accuracy_score(Y_test, Y_pred_dt),
    accuracy_score(Y_test, Y_pred_rf),
    accuracy_score(Y_test, Y_pred_xgb),
    accuracy_score(Y_test, Y_pred_voting)
]
colors = ['#e74c3c', '#2ecc71', '#3498db', '#9b59b6']
plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracies, color=colors, edgecolor='black', linewidth=0.5)
plt.ylim(0.95, 1.0)
plt.title('Model Accuracy Comparison', fontsize=16)
plt.ylabel('Accuracy Score')
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{acc:.4f}', ha='center', va='bottom', fontsize=11)
plt.tight_layout()
plt.savefig('visualizations/model_comparison.png', dpi=150)
plt.close()
print("Model comparison saved")

print("\nAll visualizations saved to visualizations/ folder")