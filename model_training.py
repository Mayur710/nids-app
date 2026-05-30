import os 
import numpy as np 
import joblib
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
from sklearn.tree import DecisionTreeClassifier   
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from sklearn.ensemble import VotingClassifier

#loading pre processed data 
# X_train_scaled = np.load('preprocessed_data/X_train_scaled.npy')
# X_test_scaled = np.load('preprocessed_data/X_test_scaled.npy')
Y_train = np.load('preprocessed_data/Y_train.npy')
Y_test = np.load('preprocessed_data/Y_test.npy')
label_encoder = joblib.load('preprocessed_data/label_encoder.pkl')
# print(X_train_scaled.shape)
# print(X_test_scaled.shape)
# print(Y_train.shape)
# print(Y_test.shape)

# pca = PCA(n_components=0.95) #keeping variance = 95%
# X_train_pca = pca.fit_transform(X_train_scaled)
# X_test_pca = pca.transform(X_test_scaled)
# # print(X_train_pca.shape)
# # print(X_test_pca.shape)
# joblib.dump(pca, 'preprocessed_data/pca.pkl')
# np.save('preprocessed_data/X_train_pca.npy', X_train_pca)
# np.save('preprocessed_data/X_test_pca.npy', X_test_pca)

#now that the pca is trained we just load the pca transformed data
X_train_pca = np.load('preprocessed_data/X_train_pca.npy')
X_test_pca = np.load('preprocessed_data/X_test_pca.npy')

#processing 2million rows of data is taking too long so we sample the data to 300k rows and then train the model on that data 
X_train_pca, Y_train = resample(X_train_pca, Y_train, n_samples=300000, random_state=42, stratify=Y_train)
#since data is imbalanced we use SMOTE to take care of the imbalance and then train the model on the balanced data
# smote_strategy = {
#     1: 5000,   # Bot
#     3: 5000,   # DoS GoldenEye
#     5: 5000,   # DoS Slowhttptest
#     6: 5000,   # DoS slowloris
#     7: 5000,   # FTP-Patator
#     11: 5000,  # SSH-Patator
#     12: 5000,  # Web Attack Brute Force
#     14: 5000,  # Web Attack XSS
# }
# smote = SMOTE(random_state=42, sampling_strategy=smote_strategy)
# X_train_pca, Y_train = smote.fit_resample(X_train_pca, Y_train) 
# # Train a Random Forest Classifier
# rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_classifier.fit(X_train_pca, Y_train)
# # Predict on the test set
# Y_pred_rf = rf_classifier.predict(X_test_pca)

# #train a Decision Tree Classifier
# dt_classifier = DecisionTreeClassifier(random_state=42)
# dt_classifier.fit(X_train_pca, Y_train)
# Y_pred_dt = dt_classifier.predict(X_test_pca)

# #train xgboost classifier
# xgb_classifier = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')
# xgb_classifier.fit(X_train_pca, Y_train)
# Y_pred_xgb = xgb_classifier.predict(X_test_pca)

# #save the models
# os.makedirs('models', exist_ok=True) 
# joblib.dump(rf_classifier, 'models/rf.pkl')
# joblib.dump(dt_classifier, 'models/dt.pkl')
# joblib.dump(xgb_classifier, 'models/xgb.pkl')

# Evaluate the models
# print("Random Forest Classifier:")
# print(classification_report(Y_test, Y_pred_rf, target_names=label_encoder.classes_))
# print("Decision Tree Classifier:")
# print(classification_report(Y_test, Y_pred_dt, target_names=label_encoder.classes_))
# print("XGBoost Classifier:")
# print(classification_report(Y_test, Y_pred_xgb, target_names=label_encoder.classes_))

"""now that we have trained the models we don't need to run their code again and for voting classifier we can just load models into it"""
rf_classifier = joblib.load('models/rf.pkl')
dt_classifier = joblib.load('models/dt.pkl')
xgb_classifier = joblib.load('models/xgb.pkl')
voting_classifier = VotingClassifier(estimators=[
    ('rf', rf_classifier),
    ('dt', dt_classifier),
    ('xgb', xgb_classifier)
], voting='soft')
voting_classifier.fit(X_train_pca, Y_train)
Y_pred_voting = voting_classifier.predict(X_test_pca)
print("Voting Classifier:")
print(classification_report(Y_test, Y_pred_voting, target_names=label_encoder.classes_))
joblib.dump(voting_classifier, 'models/voting.pkl')