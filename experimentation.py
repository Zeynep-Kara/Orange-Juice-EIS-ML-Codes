# === Importlar ===
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report, f1_score, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


# opsiyonel kütüphaneler (kurulu değilse yoruma al)
try:
    from xgboost import XGBClassifier, XGBRegressor
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:
    XGBClassifier = XGBRegressor = LGBMClassifier = LGBMRegressor = None

import warnings
warnings.filterwarnings("ignore")

# === Enhanced Unified Model Training with Dimensionality Reduction (Combined Identifier CSV) ===
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.metrics import r2_score, accuracy_score, classification_report
from keras import layers, models
import numpy as np
import pandas as pd

import time

def return_identity(x_train, x_test, dim=10):
    return x_train, x_test

# === 1️⃣ Helper: Apply FFT ===
def apply_fft(x_train, x_test, dim=10):
    # Validate input data
    try:
        x_train = np.asarray(x_train, dtype=np.float32)
        x_test = np.asarray(x_test, dtype=np.float32)
    except ValueError as e:
        raise ValueError(f"Invalid input data for FFT: {str(e)}")
    
    # Check dimensions
    if x_train.ndim != 2 or x_test.ndim != 2:
        raise ValueError("Input arrays must be 2-dimensional")
    
    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError("Training and test data must have same number of features")
    
    try:
        return np.abs(np.fft.fft(x_train, axis=1)), np.abs(np.fft.fft(x_test, axis=1))
    except Exception as e:
        raise RuntimeError(f"FFT computation failed: {str(e)}")

# === 2️⃣ Helper: AutoEncoder ===
def apply_autoencoder(x_train, x_test, dim=10, epochs=10, batch_size=32):
    # Validate input data
    try:
        x_train = np.asarray(x_train, dtype=np.float32)
        x_test = np.asarray(x_test, dtype=np.float32)
    except ValueError as e:
        raise ValueError(f"Invalid input data for autoencoder: {str(e)}")
    
    # Check dimensions
    if x_train.ndim != 2 or x_test.ndim != 2:
        raise ValueError("Input arrays must be 2-dimensional")
    
    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError("Training and test data must have same number of features")
    
    input_dim = x_train.shape[1]
    
    # Validate dim parameter
    if dim >= input_dim:
        raise ValueError(f"dim ({dim}) must be less than input dimension ({input_dim})")
    
    # Validate batch_size
    if batch_size > len(x_train):
        batch_size = len(x_train)
        print(f"Warning: batch_size adjusted to {batch_size} due to small sample size")

    try:
        encoder_input = layers.Input(shape=(input_dim,))
        encoded = layers.Dense(64, activation='relu')(encoder_input)
        encoded = layers.Dense(32, activation='relu')(encoded)
        latent = layers.Dense(dim, activation='relu')(encoded)
        decoded = layers.Dense(32, activation='relu')(latent)
        decoded = layers.Dense(64, activation='relu')(decoded)
        decoder_output = layers.Dense(input_dim, activation='sigmoid')(decoded)

        autoencoder = models.Model(encoder_input, decoder_output)
        encoder = models.Model(encoder_input, latent)

        autoencoder.compile(optimizer='adam', loss='mse')
        autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=batch_size,verbose=0)
        return encoder.predict(x_train), encoder.predict(x_test)
    except Exception as e:
        raise RuntimeError(f"Autoencoder training failed: {str(e)}")

def apply_pca(x_train, x_test, dim=10):
    # Validate input data
    try:
        x_train = np.asarray(x_train, dtype=np.float32)
        x_test = np.asarray(x_test, dtype=np.float32)
    except ValueError as e:
        raise ValueError(f"Invalid input data for PCA: {str(e)}")
    
    # Check dimensions
    if x_train.ndim != 2 or x_test.ndim != 2:
        raise ValueError("Input arrays must be 2-dimensional")
    
    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError("Training and test data must have same number of features")
    
    # Validate dim parameter
    if dim >= x_train.shape[1]:
        raise ValueError(f"dim ({dim}) must be less than number of features ({x_train.shape[1]})")
    
    try:
        pca = PCA(n_components=dim)
        x_train_pca = pca.fit_transform(x_train)
        x_test_pca = pca.transform(x_test)
        return x_train_pca, x_test_pca
    except Exception as e:
        raise RuntimeError(f"PCA computation failed: {str(e)}")

import numpy as np
from umap import UMAP
from sklearn.utils import check_array

def apply_umap(x_train, x_test, dim=2, n_neighbors=15, min_dist=0.1, random_state=42):
    n_components = dim
    # Validate input data
    try:
        x_train = check_array(x_train, dtype=np.float32, accept_sparse=True)
        x_test = check_array(x_test, dtype=np.float32, accept_sparse=True)
    except ValueError as e:
        raise ValueError(f"Invalid input data: {str(e)}")
    
    # Check minimum dimensions
    n_samples, n_features = x_train.shape
    if n_samples < 2:
        raise ValueError("At least 2 samples required")
    if n_features < 2:
        raise ValueError("At least 2 features required")
        
    # Validate n_components
    if n_components >= n_features:
        raise ValueError(f"n_components ({n_components}) must be less than n_features ({n_features})")
        
    # Validate n_neighbors
    if n_neighbors >= n_samples:
        n_neighbors = max(2, n_samples - 1)
        print(f"Warning: n_neighbors adjusted to {n_neighbors} due to small sample size")
    
    # Fit UMAP
    umap_model = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        n_jobs=1
    )
    
    try:
        embedding = umap_model.fit_transform(x_train)
        x_test_embedding = umap_model.transform(x_test)
        return embedding, x_test_embedding  
    except Exception as e:
        raise RuntimeError(f"UMAP fitting failed: {str(e)}")

def apply_supervised_umap(x_train, x_test, dim=2, labels=None, n_neighbors=15, min_dist=0.1, random_state=42):
    n_components = dim
    # Validate input data
    try:
        x_train = check_array(x_train, dtype=np.float32, accept_sparse=True)
        x_test = check_array(x_test, dtype=np.float32, accept_sparse=True)
        labels = np.asarray(labels)
    except ValueError as e:
        raise ValueError(f"Invalid input data: {str(e)}")
    
    # Check data and labels dimensions match
    if len(x_train) != len(labels):
        raise ValueError(f"Data has {len(x_train)} samples but {len(labels)} labels provided")

    # Check minimum dimensions
    n_samples, n_features = x_train.shape
    if n_samples < 2:
        raise ValueError("At least 2 samples required")
    if n_features < 2:
        raise ValueError("At least 2 features required")
        
    # Validate n_components
    if n_components >= n_features:
        raise ValueError(f"n_components ({n_components}) must be less than n_features ({n_features})")
        
    # Validate n_neighbors
    if n_neighbors >= n_samples:
        n_neighbors = max(2, n_samples - 1)
        print(f"Warning: n_neighbors adjusted to {n_neighbors} due to small sample size")
    
    # Fit supervised UMAP
    umap_model = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        n_jobs=1
    )
    
    try:
        embedding = umap_model.fit_transform(x_train, y=labels)
        x_test_embedding = umap_model.transform(x_test)
        return embedding, x_test_embedding
    except Exception as e:
        raise RuntimeError(f"Supervised UMAP fitting failed: {str(e)}")




# === Tüm modeller ===
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor,
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

# === Data Manipulation Models ===

data_models = { #6 models
    'default': return_identity,
    'fft': apply_fft,
    'auto_encoder': apply_autoencoder,
    'pca': apply_pca,
    'umap': apply_umap,
    'umap_supervised': apply_supervised_umap  
}

# === Regression Models ===
regression_models = { # 10 models (with 6 data models, 5 folds and 50 iterations, total 15000 rows of data)
    'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42,n_jobs=1),
    'LinearRegression': LinearRegression(n_jobs=1),
    'SVR': SVR(kernel='linear', C=1.0),
    'GradientBoostingRegressor': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'AdaBoostRegressor': AdaBoostRegressor(n_estimators=100, random_state=42),
    'XGBRegressor': XGBRegressor(n_estimators=100, random_state=42,n_jobs=1),
    'CatBoostRegressor': CatBoostRegressor(iterations=100, random_state=42, verbose=0,thread_count=1),
    'Lasso': Lasso(),
    'Ridge': Ridge(),
    'MLPRegressor': MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500)


}

# === Classification Models ===
classification_models = { # 11 models (with 6 data models, 5 folds and 50 iterations, total 16500 rows of data)
    'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=42,n_jobs=1),
    'LogisticRegression': LogisticRegression(max_iter=2000, solver='lbfgs',  n_jobs=1),
    'SVC': SVC(kernel='rbf', C=1.0, probability=True,cache_size=1000),
    'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=5,n_jobs=1),
    'GradientBoostingClassifier': GradientBoostingClassifier(n_estimators=100, random_state=42,max_depth=3), 
    'AdaBoostClassifier': AdaBoostClassifier(n_estimators=100, random_state=42),
    'XGBClassifier': XGBClassifier(n_estimators=100, random_state=4,n_jobs=1),
    'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42),
    'GaussianNB': GaussianNB(),
    'CatBoostClassifier': CatBoostClassifier(iterations=100, random_state=42, verbose=10,thread_count=1), # -- başlangıcı çalışıyor gibi görünüyor süre fazla uzun olduğu için sonuna kadar test etmedim
    'MLPClassifier': MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=500)

}

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold 
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def eval(data, data_y, seed=42, iteration_count=5, k_fold_n_splits=5, regress=True, classify=True, split_by_grouped_data=False, save_file="evaluation_results.csv"):
    if not regress and not classify:
        raise ValueError("At least one of 'regress' or 'classify' must be True.")

    results = []
    for name, data_model in data_models.items():
        # print(f"Evaluating {name} on...")
        for i in range(iteration_count):
            # print(f"Iteration {i + 1}/{iteration_count}")
            seed += 100
            
            kf = KFold(n_splits=k_fold_n_splits, shuffle=True, random_state=seed + i)
            
            
            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(data)):
            #for fold_idx, (train_idx, test_idx) in enumerate(kf.split(data, data_y)):   
                # print(f"  Fold {fold_idx + 1}/{k_fold_n_splits}")
                
                # Split data and target for current fold
                if split_by_grouped_data:
                    print(train_idx)
                    print(test_idx)
                    X_train_groups = [data[i] for i in train_idx]
                    X_test_groups = [data[i] for i in test_idx]
                    y_train_groups = [data_y[i] for i in train_idx]
                    y_test_groups = [data_y[i] for i in test_idx]

                    X_train = np.vstack(X_train_groups)
                    X_test = np.vstack(X_test_groups)
                    y_train = np.hstack(y_train_groups)
                    y_test = np.hstack(y_test_groups)

                    print(f"Shapes after grouping - X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")
                else:
                    X_train = data[train_idx]
                    X_test = data[test_idx]
                    y_train = data_y[train_idx]
                    y_test = data_y[test_idx]
                
                try:

                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                    
                    # Apply data transformation model
                    scale_start = time.perf_counter()
                    if 'supervised' in name:
                        X_train_transformed, X_test_transformed = data_model(X_train, X_test, labels=y_train)
                    else: 
                        X_train_transformed, X_test_transformed = data_model(X_train, X_test)
                    scale_end = time.perf_counter()

                    transform_time = scale_end - scale_start
                except Exception as e:
                    print(f"Error in {name} model, fold {fold_idx + 1}: {str(e)}")
                    continue
                    
                if regress:
                    # iterate over all regression models for this fold
                    for model_name, base_model in regression_models.items():
                        # print(f"    Model: {model_name}")
                        try:
                            model = clone(base_model)
                            time_start = time.perf_counter()
                            model.fit(X_train_transformed, y_train)
                            time_fit = time.perf_counter()
                            y_pred = model.predict(X_test_transformed)
                            time_end = time.perf_counter()
                            predict_time = time_end - time_fit
                            fit_time = time_fit - time_start
                            
                            mse = mean_squared_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            print(f"    {model_name} => MSE: {mse:.4f}, R2: {r2:.4f}")
                            results.append({
                                "DataModel": name,
                                "ModelType": "Regression",
                                "Model": model_name,
                                "Fold": fold_idx + 1,
                                "Iteration": i + 1,
                                "MSE": mse,
                                "R2": r2,
                                "TransformTime": transform_time, 
                                "FitTime": fit_time, 
                                "PredictTime": predict_time
                            })

                        except Exception as e:
                            print(f"    Training/prediction error for {model_name} in fold {fold_idx + 1}: {e}")

                if classify:
                    for model_name, base_model in classification_models.items():
                        # print(f"    Model: {model_name}")
                        try:
                            labeler = LabelEncoder()
                            y_train_labelled = labeler.fit_transform(y_train)
                            model = clone(base_model)
                            time_start = time.perf_counter()
                            model.fit(X_train_transformed, y_train_labelled)
                            time_fit = time.perf_counter()
                            y_pred = model.predict(X_test_transformed)
                            time_end = time.perf_counter()
                            predict_time = time_end - time_fit
                            fit_time = time_fit - time_start
                            y_pred = labeler.inverse_transform(y_pred)
                            
                            accuracy = accuracy_score(y_test, y_pred)
                            print(f"    {model_name} => Accuracy: {accuracy:.4f}")
                            results.append({
                                "DataModel": name,
                                "ModelType": "Classifier",
                                "Model": model_name,
                                "Fold": fold_idx + 1,
                                "Iteration": i + 1,
                                "Accuracy": accuracy,
                                "TransformTime": transform_time, 
                                "FitTime": fit_time, 
                                "PredictTime": predict_time
                            })
                        except Exception as e:
                            print(f"    Training/prediction error for {model_name} in fold {fold_idx + 1}: {e}")
                # if len(results) % 500 == 0:
                df = pd.DataFrame(results)
                df.to_csv(save_file, index=False)
    df = pd.DataFrame(results)
    return df

def get_average_results(df):
    try:
        avg_results = df.groupby(['DataModel', 'ModelType', 'Model']).agg({
            'MSE': 'mean',
            'R2': 'mean'
        }).reset_index()
        return avg_results
    except Exception as e:
        avg_results = df.groupby(['DataModel', 'ModelType', 'Model']).agg({
            'Accuracy': 'mean'
        }).reset_index()
        return avg_results