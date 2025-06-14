from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = XGBRegressor = None

class ModelTrainer:
    def __init__(self, task_type):
        self.task_type = task_type.lower()
        self.models = {}
        
    def get_model(self, model_name):
        if self.task_type == 'classification':
            if model_name == 'Logistic Regression':
                return LogisticRegression(random_state=42, max_iter=1000)
            elif model_name == 'Decision Tree':
                return DecisionTreeClassifier(random_state=42)
            elif model_name == 'Random Forest':
                return RandomForestClassifier(random_state=42, n_estimators=100)
            elif model_name == 'SVM':
                return SVC(random_state=42, probability=True)
            elif model_name == 'XGBoost' and XGBClassifier:
                return XGBClassifier(random_state=42)
        else:
            if model_name == 'Linear Regression':
                return LinearRegression()
            elif model_name == 'Ridge Regression':
                return Ridge(random_state=42)
            elif model_name == 'Lasso Regression':
                return Lasso(random_state=42)
            elif model_name == 'Random Forest':
                return RandomForestRegressor(random_state=42, n_estimators=100)
            elif model_name == 'SVR':
                return SVR()
            elif model_name == 'XGBoost' and XGBRegressor:
                return XGBRegressor(random_state=42)
        raise ValueError(f"Model {model_name} not supported for {self.task_type}")
    
    def train_model(self, model_name, X_train, y_train):
        model = self.get_model(model_name)
        model.fit(X_train, y_train)
        self.models[model_name] = model
        return model
    
    def train_multiple_models(self, model_names, X_train, y_train):
        trained_models = {}
        for model_name in model_names:
            model = self.train_model(model_name, X_train, y_train)
            trained_models[model_name] = model
        return trained_models
