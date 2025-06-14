import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE, SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder

class FeatureSelector:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.is_classification = len(y.unique()) < 20 and y.dtype == 'object'
        self.importance_scores = {}
        
    def rfe_selection(self, n_features):
        X_processed = self.X.copy()
        for col in X_processed.columns:
            if X_processed[col].dtype == 'object':
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col].astype(str))
        if self.is_classification:
            estimator = LogisticRegression(random_state=42, max_iter=1000)
        else:
            estimator = LinearRegression()
        rfe = RFE(estimator, n_features_to_select=n_features)
        rfe.fit(X_processed, self.y)
        selected_features = self.X.columns[rfe.support_].tolist()
        self.importance_scores = dict(zip(self.X.columns, rfe.ranking_))
        return selected_features
    
    def mutual_info_selection(self, k_features):
        if self.is_classification:
            mi_scores = mutual_info_classif(self.X, self.y, random_state=42)
        else:
            from sklearn.feature_selection import mutual_info_regression
            mi_scores = mutual_info_regression(self.X, self.y, random_state=42)
        selector = SelectKBest(score_func=lambda X, y: mi_scores, k=k_features)
        selector.fit(self.X, self.y)
        selected_features = self.X.columns[selector.get_support()].tolist()
        self.importance_scores = dict(zip(self.X.columns, mi_scores))
        return selected_features
    
    def tree_importance_selection(self, threshold=0.01):
        if self.is_classification:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(self.X, self.y)
        importances = model.feature_importances_
        selected_features = self.X.columns[importances > threshold].tolist()
        self.importance_scores = dict(zip(self.X.columns, importances))
        return selected_features
    
    def statistical_selection(self, k_features):
        if self.is_classification:
            if self.X.dtypes.apply(lambda x: x == 'object').any():
                selector = SelectKBest(score_func=chi2, k=k_features)
            else:
                selector = SelectKBest(score_func=f_classif, k=k_features)
        else:
            from sklearn.feature_selection import f_regression
            selector = SelectKBest(score_func=f_regression, k=k_features)
        selector.fit(self.X, self.y)
        selected_features = self.X.columns[selector.get_support()].tolist()
        self.importance_scores = dict(zip(self.X.columns, selector.scores_))
        return selected_features
    
    def get_feature_importance(self):
        return self.importance_scores
