import pandas as pd 
import glob
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
import numpy as np 
# Load the dataset
data = glob.glob('MachineLearningCVE/*.csv')  # adjust the path and file extension as needed
df = pd.concat((pd.read_csv(file) for file in data), ignore_index=True) #we loaded all the files and concatenated them into a single dataframe

"""Data cleaning , dataset has too many whitespaces and this line of code will remove all the whitespaces from the dataset and replace them with a single space"""
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
df.columns = df.columns.str.strip() #removing whitespaces from column names
"""removing infinity values"""
df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
"""dropping rows with missing values"""
df.dropna(inplace=True)
"""dropping duplicate rows"""
df.drop_duplicates(inplace=True)
"""dropping constant columns"""
df = df.loc[:, ~(df == df.iloc[0]).all()]#df.iloc takes first row, df.iloc[0] compares every cell in df to correspoding value in that first row
"""checking label column"""
# print(df['Label'].value_counts())
# print(df.shape)
# print(df.isnull().sum().sum())
"""doing the encoding for the web attack column"""
df['Label'] = df['Label'].str.replace('â\x80\x93', '-', regex=False)
"""splitting the dataset into features and target variable"""
X = df.drop('Label', axis=1)
Y = df['Label']
scaler = StandardScaler()
"""doing label encoding for the target variable"""
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

os.makedirs('preprocessed_data', exist_ok=True)
np.save('preprocessed_data/X_train_scaled.npy',X_train_scaled, )
np.save('preprocessed_data/X_test_scaled.npy',X_test_scaled, )           
np.save('preprocessed_data/Y_train.npy',Y_train, )
np.save('preprocessed_data/Y_test.npy',Y_test )
np.save('preprocessed_data/label_encoder.npy',label_encoder)

joblib.dump(label_encoder, 'preprocessed_data/label_encoder.pkl')
joblib.dump(scaler, 'preprocessed_data/scaler.pkl')
joblib.dump(list(X.columns), 'preprocessed_data/feature_columns.pkl')