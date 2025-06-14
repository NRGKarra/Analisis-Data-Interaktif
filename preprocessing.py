import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest
import scipy.stats as stats
import streamlit as st

class DataPreprocessor:
    def __init__(self, data):
        self.data = data
        self.scalers = {}
        self.encoders = {}
        
    def detect_outliers(self, columns, method='z-score', threshold=3):
        outliers = []
        for col in columns:
            if method == 'z-score':
                z_scores = np.abs(stats.zscore(self.data[col]))
                outliers.extend(self.data[z_scores > threshold].index.tolist())
            elif method == 'iqr':
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers.extend(self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)].index.tolist())
            elif method == 'isolation forest':
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_pred = iso_forest.fit_predict(self.data[[col]])
                outliers.extend(self.data[outlier_pred == -1].index.tolist())
        return list(set(outliers))
    
    def apply_scaling(self, data, columns, method='standardscaler'):
        scaled_data = data.copy()
        if method.lower() == 'standardscaler':
            scaler = StandardScaler()
        elif method.lower() == 'minmaxscaler':
            scaler = MinMaxScaler()
        elif method.lower() == 'robustscaler':
            scaler = RobustScaler()
        else:
            return scaled_data
        scaled_data[columns] = scaler.fit_transform(scaled_data[columns])
        self.scalers[method] = scaler
        return scaled_data
    
    def handle_missing_values(self, data, strategies):
        processed_data = data.copy()
        for column, strategy in strategies.items():
            if strategy == 'knn':
                imputer = KNNImputer(n_neighbors=5)
                processed_data[column] = imputer.fit_transform(processed_data[[column]])
            else:
                imputer = SimpleImputer(strategy=strategy)
                processed_data[column] = imputer.fit_transform(processed_data[[column]]).ravel()
        return processed_data
    
    def encode_categorical(self, data, columns, method='one-hot encoding'):
        encoded_data = data.copy()
        for column in columns:
            if column not in encoded_data.columns:
                st.warning(f"Kolom {column} tidak ditemukan dalam data.")
                continue
            if method == 'label encoding':
                le = LabelEncoder()
                encoded_data[column] = le.fit_transform(encoded_data[column].astype(str))
                self.encoders[column] = le
            elif method == 'one-hot encoding':
                dummies = pd.get_dummies(encoded_data[column], prefix=column, drop_first=True)
                encoded_data = pd.concat([encoded_data.drop(column, axis=1), dummies], axis=1)
            elif method == 'target encoding':
                if hasattr(self, 'target_column'):
                    target_mean = encoded_data.groupby(column)[self.target_column].mean()
                    encoded_data[column] = encoded_data[column].map(target_mean)
        return encoded_data
