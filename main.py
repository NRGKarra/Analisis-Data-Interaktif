import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import RFE, SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')

from preprocessing import DataPreprocessor
from feature_selection import FeatureSelector
from model_training import ModelTrainer
from evaluation import ModelEvaluator
from utils import create_sample_datasets

st.set_page_config(
    page_title="ML Pipeline Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🤖 Machine Learning Pipeline Analyzer</h1>', unsafe_allow_html=True)
    
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.selectbox(
        "Pilih Tahapan:",
        ["🏠 Home", "📁 Upload Dataset", "🔧 Preprocessing", "🎯 Feature Selection", 
         "🤖 Model Selection", "⚙️ Hyperparameter Tuning", "📊 Model Evaluation", "🧠 Model Interpretability"]
    )
    
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'selected_features' not in st.session_state:
        st.session_state.selected_features = None
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
    
    if page == "🏠 Home":
        show_home()
    elif page == "📁 Upload Dataset":
        show_upload()
    elif page == "🔧 Preprocessing":
        show_preprocessing()
    elif page == "🎯 Feature Selection":
        show_feature_selection()
    elif page == "🤖 Model Selection":
        show_model_selection()
    elif page == "⚙️ Hyperparameter Tuning":
        show_hyperparameter_tuning()
    elif page == "📊 Model Evaluation":
        show_evaluation()
    elif page == "🧠 Model Interpretability":
        show_interpretability()

def show_home():
    st.markdown('<h2 class="section-header">Selamat Datang di ML Pipeline Analyzer!</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🎯 Fitur Utama:
        - **Upload Dataset**: Unggah data CSV dengan preview interaktif
        - **Preprocessing**: Otomatis & manual handling untuk outliers, scaling, encoding
        - **Feature Selection**: RFE, Mutual Information, Tree-based importance
        - **Model Selection**: Berbagai algoritma klasifikasi & regresi
        - **Hyperparameter Tuning**: Grid Search & Random Search
        - **Evaluasi Komprehensif**: Metrik lengkap dengan visualisasi
        - **Interpretabilitas**: SHAP values untuk explainable AI
        """)
    with col2:
        st.markdown("""
        ### 📊 Dataset yang Didukung:
        - **Drug Dataset**: Prediksi jenis obat berdasarkan karakteristik pasien
        - **Mushroom Dataset**: Klasifikasi jamur beracun vs dapat dimakan
        - **Custom Dataset**: Upload dataset CSV Anda sendiri
        
        ### 🚀 Cara Penggunaan:
        1. Mulai dengan upload dataset di menu "Upload Dataset"
        2. Lakukan preprocessing sesuai kebutuhan
        3. Pilih fitur yang relevan
        4. Bandingkan berbagai model
        5. Tune hyperparameter untuk performa optimal
        6. Evaluasi dan interpretasi hasil
        """)
    st.markdown('<h3 class="section-header">📈 Dataset Sampel</h3>', unsafe_allow_html=True)
    if st.button("🧪 Load Drug Dataset"):
        drug_data = create_sample_datasets()['drug']
        st.session_state.data = drug_data
        st.session_state.target_column = 'Drug'
        st.success("Drug dataset berhasil dimuat!")
        st.dataframe(drug_data.head())
    if st.button("🍄 Load Mushroom Dataset"):
        mushroom_data = create_sample_datasets()['mushroom']
        st.session_state.data = mushroom_data
        st.session_state.target_column = 'class'
        st.success("Mushroom dataset berhasil dimuat!")
        st.dataframe(mushroom_data.head())

def show_upload():
    st.markdown('<h2 class="section-header">📁 Upload Dataset</h2>', unsafe_allow_html=True)
    upload_option = st.radio("Pilih metode upload:", ["Upload file CSV", "Gunakan dataset sampel"])
    if upload_option == "Upload file CSV":
        uploaded_file = st.file_uploader("Pilih file CSV", type="csv")
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.session_state.data = data
                st.success(f"Dataset berhasil dimuat! Shape: {data.shape}")
                st.dataframe(data.head(10))
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📊 Informasi Dataset")
                    st.write(f"**Jumlah baris:** {data.shape[0]}")
                    st.write(f"**Jumlah kolom:** {data.shape[1]}")
                    st.write(f"**Missing values:** {data.isnull().sum().sum()}")
                with col2:
                    st.subheader("🏷️ Tipe Data")
                    dtype_df = pd.DataFrame({
                        'Column': data.columns,
                        'Type': data.dtypes,
                        'Non-Null Count': data.count(),
                        'Unique Values': [data[col].nunique() for col in data.columns]
                    })
                    st.dataframe(dtype_df)
                st.subheader("🎯 Pilih Target Column")
                target_col = st.selectbox("Pilih kolom target untuk prediksi:", data.columns)
                st.session_state.target_column = target_col
                if target_col:
                    st.write(f"Target yang dipilih: **{target_col}**")
                    st.write(f"Target distribution:")
                    st.write(data[target_col].value_counts())
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    else:
        dataset_choice = st.selectbox("Pilih dataset sampel:", ["Drug Dataset", "Mushroom Dataset"])
        if st.button("Load Dataset"):
            datasets = create_sample_datasets()
            if dataset_choice == "Drug Dataset":
                st.session_state.data = datasets['drug']
                st.session_state.target_column = 'Drug'
                st.success("Drug dataset berhasil dimuat!")
            else:
                st.session_state.data = datasets['mushroom']
                st.session_state.target_column = 'class'
                st.success("Mushroom dataset berhasil dimuat!")
            st.dataframe(st.session_state.data.head())

def show_preprocessing():
    if st.session_state.data is None:
        st.warning("⚠️ Silakan upload dataset terlebih dahulu!")
        return
    st.markdown('<h2 class="section-header">🔧 Data Preprocessing</h2>', unsafe_allow_html=True)
    data = st.session_state.data.copy()
    preprocessor = DataPreprocessor(data)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔍 Deteksi Outliers")
        outlier_method = st.selectbox("Pilih metode deteksi outlier:", ["Z-Score", "IQR", "Isolation Forest"])
        if st.button("Deteksi Outliers"):
            try:
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                outliers = preprocessor.detect_outliers(numeric_cols, method=outlier_method.lower())
                st.write(f"Outliers terdeteksi: {len(outliers)} baris")
                if len(outliers) > 0:
                    st.dataframe(data.loc[outliers])
                    if st.button("Hapus Outliers"):
                        data = data.drop(outliers)
                        st.success(f"Berhasil menghapus {len(outliers)} outliers")
            except Exception as e:
                st.error(f"Error deteksi outliers: {str(e)}")
    with col2:
        st.subheader("📏 Scaling & Normalization")
        scaling_method = st.selectbox("Pilih metode scaling:", ["StandardScaler", "MinMaxScaler", "RobustScaler", "None"])
        if scaling_method != "None":
            numeric_cols = st.multiselect("Pilih kolom numerik untuk scaling:", data.select_dtypes(include=[np.number]).columns)
            if st.button("Apply Scaling"):
                try:
                    data = preprocessor.apply_scaling(data, numeric_cols, method=scaling_method)
                    st.success(f"Scaling {scaling_method} berhasil diterapkan!")
                except Exception as e:
                    st.error(f"Error scaling: {str(e)}")
    st.subheader("🔧 Handling Missing Values")
    missing_cols = data.columns[data.isnull().any()].tolist()
    if missing_cols:
        st.write(f"Kolom dengan missing values: {missing_cols}")
        for col in missing_cols:
            st.write(f"**{col}**: {data[col].isnull().sum()} missing values")
            if data[col].dtype in ['object']:
                impute_method = st.selectbox(f"Metode imputation untuk {col}:", ["most_frequent", "constant"], key=f"impute_{col}")
            else:
                impute_method = st.selectbox(f"Metode imputation untuk {col}:", ["mean", "median", "most_frequent", "knn"], key=f"impute_{col}")
            if st.button(f"Apply imputation untuk {col}", key=f"apply_{col}"):
                try:
                    data = preprocessor.handle_missing_values(data, {col: impute_method})
                    st.success(f"Missing values pada {col} berhasil ditangani!")
                except Exception as e:
                    st.error(f"Error imputation: {str(e)}")
    st.subheader("🏷️ Categorical Encoding")
    cat_cols = data.select_dtypes(include=['object']).columns.tolist()
    if hasattr(st.session_state, 'target_column') and st.session_state.target_column in cat_cols:
        cat_cols.remove(st.session_state.target_column)
    if cat_cols:
        encoding_method = st.selectbox("Pilih metode encoding:", ["One-Hot Encoding", "Label Encoding", "Target Encoding"])
        cols_to_encode = st.multiselect("Pilih kolom kategorikal untuk encoding:", cat_cols)
        if st.button("Apply Encoding"):
            try:
                data = preprocessor.encode_categorical(data, cols_to_encode, method=encoding_method.lower())
                st.success(f"{encoding_method} berhasil diterapkan!")
            except Exception as e:
                st.error(f"Error encoding: {str(e)}")
    if st.button("💾 Simpan Preprocessing"):
        try:
            st.session_state.processed_data = data
            st.success("Data preprocessing berhasil disimpan!")
            st.dataframe(data.head())
            st.write(f"Shape setelah preprocessing: {data.shape}")
        except Exception as e:
            st.error(f"Error simpan preprocessing: {str(e)}")

def validate_and_preprocess_data(data, target_col):
    processed_data = data.copy()
    try:
        X = processed_data.drop(columns=[target_col])
        y = processed_data[target_col]
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
        processed_data = X.copy()
        processed_data[target_col] = y
        return processed_data
    except Exception as e:
        st.error(f"Error validasi/preprocessing: {str(e)}")
        return None

def show_feature_selection():
    if st.session_state.processed_data is None:
        st.warning("⚠️ Silakan lakukan preprocessing terlebih dahulu!")
        return
    st.markdown('<h2 class="section-header">🎯 Feature Selection</h2>', unsafe_allow_html=True)
    try:
        data = validate_and_preprocess_data(
            st.session_state.processed_data,
            st.session_state.target_column
        )
        if data is None:
            st.error("Gagal validasi/preprocessing data. Silakan cek log error.")
            return
        target_col = st.session_state.target_column
        X = data.drop(columns=[target_col])
        y = data[target_col]
        feature_selector = FeatureSelector(X, y)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Feature Selection Methods")
            method = st.selectbox("Pilih metode feature selection:", ["Recursive Feature Elimination", "Mutual Information", "Tree-based Importance", "Statistical Tests"])
            if method == "Recursive Feature Elimination":
                n_features = st.slider("Jumlah fitur yang diinginkan:", 1, len(X.columns), len(X.columns)//2)
                if st.button("Apply RFE"):
                    try:
                        selected_features = feature_selector.rfe_selection(n_features)
                        st.session_state.selected_features = selected_features
                        st.success(f"RFE selesai! {len(selected_features)} fitur dipilih.")
                    except Exception as e:
                        st.error(f"Error RFE: {str(e)}")
            elif method == "Mutual Information":
                k_features = st.slider("Top K fitur:", 1, len(X.columns), len(X.columns)//2)
                if st.button("Apply Mutual Information"):
                    try:
                        selected_features = feature_selector.mutual_info_selection(k_features)
                        st.session_state.selected_features = selected_features
                        st.success(f"Mutual Information selesai! {len(selected_features)} fitur dipilih.")
                    except Exception as e:
                        st.error(f"Error Mutual Information: {str(e)}")
            elif method == "Tree-based Importance":
                threshold = st.slider("Threshold importance:", 0.0, 1.0, 0.01)
                if st.button("Apply Tree Importance"):
                    try:
                        selected_features = feature_selector.tree_importance_selection(threshold)
                        st.session_state.selected_features = selected_features
                        st.success(f"Tree importance selesai! {len(selected_features)} fitur dipilih.")
                    except Exception as e:
                        st.error(f"Error Tree Importance: {str(e)}")
        with col2:
            st.subheader("📈 Feature Importance Visualization")
            if st.session_state.selected_features is not None:
                try:
                    importance_scores = feature_selector.get_feature_importance()
                    if not importance_scores:
                        st.warning("Belum ada data feature importance.")
                    else:
                        importance_df = pd.DataFrame(list(importance_scores.items()), columns=['Feature', 'Importance'])
                        importance_df = importance_df.sort_values('Importance', ascending=False)
                        fig = px.bar(importance_df, x='Feature', y='Importance', title="Feature Importance Scores")
                        st.plotly_chart(fig, use_container_width=True)
                        st.subheader("✅ Selected Features")
                        st.write(st.session_state.selected_features)
                except Exception as e:
                    st.error(f"Error visualisasi feature importance: {str(e)}")
        st.subheader("🔥 Correlation Heatmap")
        if st.button("Show Correlation"):
            try:
                numeric_data = X.select_dtypes(include=[np.number])
                if len(numeric_data.columns) > 1:
                    corr_matrix = numeric_data.corr()
                    fig = px.imshow(corr_matrix, text_auto=True, title="Feature Correlation Matrix")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Minimal 2 kolom numerik diperlukan untuk heatmap.")
            except Exception as e:
                st.error(f"Error membuat heatmap: {str(e)}")
    except Exception as e:
        st.error(f"Error feature selection: {str(e)}")

def show_model_selection():
    if st.session_state.processed_data is None:
        st.warning("⚠️ Silakan lakukan preprocessing terlebih dahulu!")
        return
    st.markdown('<h2 class="section-header">🤖 Model Selection & Training</h2>', unsafe_allow_html=True)
    try:
        data = validate_and_preprocess_data(
            st.session_state.processed_data,
            st.session_state.target_column
        )
        if data is None:
            st.error("Gagal validasi/preprocessing data. Silakan cek log error.")
            return
        st.session_state.model_ready_data = data
        target_col = st.session_state.target_column
        if st.session_state.selected_features is not None:
            X = data[st.session_state.selected_features]
        else:
            X = data.drop(columns=[target_col])
        y = data[target_col]
        is_classification = len(y.unique()) < 20 and y.dtype == 'object'
        task_type = "Classification" if is_classification else "Regression"
        st.write(f"**Task Type:** {task_type}")
        st.write(f"**Features shape:** {X.shape}")
        st.write(f"**Target shape:** {y.shape}")
        test_size = st.slider("Test size ratio:", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        model_trainer = ModelTrainer(task_type)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🎯 Available Models")
            if is_classification:
                available_models = ["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "XGBoost"]
            else:
                available_models = ["Linear Regression", "Ridge Regression", "Lasso Regression", "Random Forest", "SVR"]
            selected_models = st.multiselect("Pilih model untuk training:", available_models)
            if st.button("🚀 Train Models"):
                try:
                    progress_bar = st.progress(0)
                    trained_models = {}
                    for i, model_name in enumerate(selected_models):
                        st.write(f"Training {model_name}...")
                        model = model_trainer.train_model(model_name, X_train, y_train)
                        trained_models[model_name] = {
                            'model': model,
                            'X_test': X_test,
                            'y_test': y_test,
                            'predictions': model.predict(X_test)
                        }
                        progress_bar.progress((i + 1) / len(selected_models))
                    st.session_state.trained_models = trained_models
                    st.success("🎉 Semua model berhasil ditraining!")
                except Exception as e:
                    st.error(f"Error training model: {str(e)}")
        with col2:
            st.subheader("📋 Model Descriptions")
            model_descriptions = {
                "Logistic Regression": "Linear model untuk klasifikasi dengan probabilitas output",
                "Linear Regression": "Model linear sederhana untuk prediksi kontinu",
                "Decision Tree": "Model berbasis aturan yang mudah diinterpretasi",
                "Random Forest": "Ensemble dari banyak decision trees",
                "SVM": "Support Vector Machine untuk klasifikasi/regresi",
                "Ridge Regression": "Linear regression dengan L2 regularization",
                "Lasso Regression": "Linear regression dengan L1 regularization",
                "XGBoost": "Gradient boosting yang powerful dan efisien"
            }
            for model in available_models:
                with st.expander(f"ℹ️ {model}"):
                    st.write(model_descriptions.get(model, "Model description not available"))
    except Exception as e:
        st.error(f"Error model selection: {str(e)}")

def show_hyperparameter_tuning():
    if not st.session_state.trained_models:
        st.warning("⚠️ Silakan training model terlebih dahulu!")
        return
    st.markdown('<h2 class="section-header">⚙️ Hyperparameter Tuning</h2>', unsafe_allow_html=True)
    model_names = list(st.session_state.trained_models.keys())
    selected_model = st.selectbox("Pilih model untuk tuning:", model_names)
    if selected_model:
        st.subheader(f"🔧 Tuning {selected_model}")
        param_grids = {
            "Random Forest": {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            },
            "Logistic Regression": {
                'C': [0.1, 1, 10],
                'solver': ['liblinear', 'saga']
            },
            "SVM": {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf']
            }
        }
        if selected_model in param_grids:
            st.write("**Parameter Grid:**")
            st.json(param_grids[selected_model])
            search_method = st.radio("Pilih metode search:", ["Grid Search", "Random Search"])
            cv_folds = st.slider("Cross-validation folds:", 3, 10, 5)
            if st.button("🚀 Start Tuning"):
                try:
                    model_data = st.session_state.trained_models[selected_model]
                    X_train = st.session_state.processed_data.drop(columns=[st.session_state.target_column])
                    y_train = st.session_state.processed_data[st.session_state.target_column]
                    if st.session_state.selected_features:
                        X_train = X_train[st.session_state.selected_features]
                    X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
                    base_model = model_data['model']
                    param_grid = param_grids[selected_model]
                    if search_method == "Grid Search":
                        search = GridSearchCV(base_model, param_grid, cv=cv_folds, scoring='accuracy')
                    else:
                        search = RandomizedSearchCV(base_model, param_grid, cv=cv_folds, n_iter=10, scoring='accuracy')
                    search.fit(X_train, y_train)
                    st.success("✅ Tuning completed!")
                    st.write(f"**Best Score:** {search.best_score_:.4f}")
                    st.write(f"**Best Parameters:**")
                    st.json(search.best_params_)
                    st.session_state.trained_models[selected_model]['tuned_model'] = search.best_estimator_
                    st.session_state.trained_models[selected_model]['best_params'] = search.best_params_
                except Exception as e:
                    st.error(f"Error tuning hyperparameter: {str(e)}")

def show_evaluation():
    if not st.session_state.trained_models:
        st.warning("⚠️ Silakan training model terlebih dahulu!")
        return
    st.markdown('<h2 class="section-header">📊 Model Evaluation</h2>', unsafe_allow_html=True)
    try:
        data = st.session_state.processed_data
        target_col = st.session_state.target_column
        y = data[target_col]
        is_classification = len(y.unique()) < 20 and y.dtype == 'object'
        evaluator = ModelEvaluator()
        results = {}
        for model_name, model_data in st.session_state.trained_models.items():
            y_test = model_data['y_test']
            y_pred = model_data['predictions']
            if is_classification:
                metrics = evaluator.classification_metrics(y_test, y_pred)
            else:
                metrics = evaluator.regression_metrics(y_test, y_pred)
            results[model_name] = metrics
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📈 Model Comparison")
            metrics_df = pd.DataFrame(results).T
            st.dataframe(metrics_df)
            if is_classification:
                best_model = metrics_df['accuracy'].idxmax()
                st.success(f"🏆 Best Model: {best_model} (Accuracy: {metrics_df.loc[best_model, 'accuracy']:.4f})")
            else:
                best_model = metrics_df['r2_score'].idxmax()
                st.success(f"🏆 Best Model: {best_model} (R² Score: {metrics_df.loc[best_model, 'r2_score']:.4f})")
        with col2:
            st.subheader("📊 Visualizations")
            if is_classification:
                fig = px.bar(x=list(results.keys()), y=[results[model]['accuracy'] for model in results.keys()], title="Model Accuracy Comparison")
            else:
                fig = px.bar(x=list(results.keys()), y=[results[model]['r2_score'] for model in results.keys()], title="Model R² Score Comparison")
            st.plotly_chart(fig, use_container_width=True)
        st.subheader("🔍 Detailed Evaluation")
        selected_model = st.selectbox("Pilih model untuk evaluasi detail:", list(results.keys()))
        if selected_model:
            model_data = st.session_state.trained_models[selected_model]
            y_test = model_data['y_test']
            y_pred = model_data['predictions']
            if is_classification:
                st.subheader("🎯 Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig = px.imshow(cm, text_auto=True, title=f"Confusion Matrix - {selected_model}")
                st.plotly_chart(fig, use_container_width=True)
                st.subheader("📋 Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)
                if len(np.unique(y_test)) == 2:
                    st.subheader("📈 ROC Curve")
                    model = model_data['model']
                    if hasattr(model, 'predict_proba'):
                        y_proba = model.predict_proba(model_data['X_test'])[:, 1]
                        fpr, tpr, _ = roc_curve(y_test, y_proba)
                        roc_auc = auc(fpr, tpr)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC Curve (AUC = {roc_auc:.2f})'))
                        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random'))
                        fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.subheader("📊 Residual Plot")
                residuals = y_test - y_pred
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers', name='Residuals'))
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                fig.update_layout(title="Residual Plot", xaxis_title="Predicted Values", yaxis_title="Residuals")
                st.plotly_chart(fig, use_container_width=True)
                st.subheader("🎯 Actual vs Predicted")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predictions'))
                fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode='lines', name='Perfect Prediction'))
                fig.update_layout(title="Actual vs Predicted", xaxis_title="Actual Values", yaxis_title="Predicted Values")
                st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error evaluasi model: {str(e)}")

def show_interpretability():
    if not st.session_state.trained_models:
        st.warning("⚠️ Silakan training model terlebih dahulu!")
        return
    st.markdown('<h2 class="section-header">🧠 Model Interpretability</h2>', unsafe_allow_html=True)
    try:
        import shap
        import lime
        import lime.lime_tabular
    except ImportError:
        st.error("📦 SHAP dan LIME libraries diperlukan untuk interpretability. Install dengan: pip install shap lime")
        return
    model_names = list(st.session_state.trained_models.keys())
    selected_model = st.selectbox("Pilih model untuk interpretasi:", model_names)
    if selected_model:
        model_data = st.session_state.trained_models[selected_model]
        model = model_data['model']
        X_test = model_data['X_test']
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🎯 SHAP Values")
            if st.button("Generate SHAP Explanation"):
                with st.spinner("Generating SHAP values..."):
                    try:
                        explainer = shap.Explainer(model, X_test)
                        shap_values = explainer(X_test[:100])
                        st.subheader("📊 SHAP Summary Plot")
                        fig, ax = plt.subplots()
                        shap.summary_plot(shap_values, X_test[:100], show=False)
                        st.pyplot(fig)
                        st.subheader("📈 SHAP Feature Importance")
                        fig, ax = plt.subplots()
                        shap.summary_plot(shap_values, X_test[:100], plot_type="bar", show=False)
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error generating SHAP explanation: {str(e)}")
        with col2:
            st.subheader("🔍 LIME Explanation")
            sample_idx = st.slider("Pilih sample untuk dijelaskan:", 0, len(X_test)-1, 0)
            if st.button("Generate LIME Explanation"):
                with st.spinner("Generating LIME explanation..."):
                    try:
                        explainer = lime.lime_tabular.LimeTabularExplainer(
                            X_test.values,
                            feature_names=X_test.columns,
                            mode='classification' if hasattr(model, 'predict_proba') else 'regression'
                        )
                        explanation = explainer.explain_instance(
                            X_test.iloc[sample_idx].values,
                            model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
                            num_features=min(10, len(X_test.columns))
                        )
                        st.subheader(f"📋 Explanation for Sample {sample_idx}")
                        exp_list = explanation.as_list()
                        exp_df = pd.DataFrame(exp_list, columns=['Feature', 'Impact'])
                        fig = px.bar(exp_df, x='Impact', y='Feature', orientation='h', title=f"LIME Explanation for Sample {sample_idx}")
                        st.plotly_chart(fig, use_container_width=True)
                        sample_df = pd.DataFrame({
                            'Feature': X_test.columns,
                            'Value': X_test.iloc[sample_idx].values
                        })
                        st.dataframe(sample_df)
                    except Exception as e:
                        st.error(f"Error generating LIME explanation: {str(e)}")

if __name__ == "__main__":
    main()
