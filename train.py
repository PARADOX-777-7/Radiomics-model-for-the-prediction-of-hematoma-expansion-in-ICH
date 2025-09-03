import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, average_precision_score,
    accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
)
from sklearn.utils import resample
import os
import time
import warnings
warnings.filterwarnings('ignore')

# ML models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from tabpfn import TabPFNClassifier
import optuna
from sklearn.model_selection import cross_val_score
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Configuration
RANDOM_STATE = 42
N_BOOTSTRAPS_CI = 100
N_BOOTSTRAPS_ROC_PLOT = 500
OPTUNA_CV_FOLDS = 10
OPTUNA_SCORING = 'roc_auc'
OPTUNA_N_TRIALS = 50
OPTUNA_TIMEOUT = 600

# File paths
TRAIN_FILEPATH = 'D:/DATA OUTPUT/train.csv'
TEST_FILEPATH = 'D:/DATA OUTPUT/test.csv'
VAL_FILEPATH = 'D:/DATA OUTPUT/val.csv'

# Plot configuration
SAVE_PLOTS = True
PLOT_SAVE_DIR = "model_evaluation_plots"
PLOT_FORMAT = "pdf"
PLOT_DPI = 300

# CSV output
PREDICTED_PROBABILITIES_CSV_FILENAME = "predicted_probabilities_all_samples.csv"

if SAVE_PLOTS and not os.path.exists(PLOT_SAVE_DIR):
    os.makedirs(PLOT_SAVE_DIR)

# Data loading
df_train = pd.read_csv(TRAIN_FILEPATH)
df_test = pd.read_csv(TEST_FILEPATH)
df_val = pd.read_csv(VAL_FILEPATH)

# Data preparation
target_column = df_train.columns[0]
feature_columns = df_train.columns[1:]

X_train = df_train[feature_columns].copy()
y_train = df_train[target_column].copy()
X_test = df_test[feature_columns].copy()
y_test = df_test[target_column].copy()
X_val = df_val[feature_columns].copy()
y_val = df_val[target_column].copy()

# Convert to numeric if needed
def coerce_to_numeric(df):
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass
    return df

X_train = coerce_to_numeric(X_train)
X_test = coerce_to_numeric(X_test)
X_val = coerce_to_numeric(X_val)

# Model definitions
try:
    import tensorflow as tf
    if tf.config.list_physical_devices('GPU'):
        tabpfn_device = 'cuda'
    else:
        tabpfn_device = 'cpu'
except:
    tabpfn_device = 'cpu'

models_sklearn_base = {
    'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE, max_depth=10, min_samples_leaf=10),
    'SVM': SVC(probability=True, random_state=RANDOM_STATE, C=1.0),
    'Extra Trees': ExtraTreesClassifier(random_state=RANDOM_STATE, max_depth=10, min_samples_leaf=10),
    'LightGBM': lgb.LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1, max_depth=5, subsample=0.8, colsample_bytree=0.8),
    'Gradient Boosting': GradientBoostingClassifier(random_state=RANDOM_STATE, max_depth=4, subsample=0.8),
    'KNN': KNeighborsClassifier(n_neighbors=9),
    'XGBoost': xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', use_label_encoder=False, max_depth=4, subsample=0.8, colsample_bytree=0.8),
    'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=7, min_samples_leaf=20),
    'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, solver='lbfgs', penalty='l2'),
    'TabPFN': TabPFNClassifier(device=tabpfn_device, random_state=RANDOM_STATE)
}

# Parameter grids
param_grids = {
    'Random Forest': {
        'n_estimators': [50,100,200],
        'max_depth': [5, 10],
        'min_samples_split': [10, 20, 40],
        'min_samples_leaf': [5, 10, 20],
        'max_features': ['sqrt', 0.7]
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    },
    'Extra Trees': {
        'n_estimators': [50, 100,200],
        'max_depth': [5, 10],
        'min_samples_split': [10, 20, 40],
        'min_samples_leaf': [5, 10, 20],
        'max_features': ['sqrt', 0.7]
    },
    'LightGBM': {
        'n_estimators': [50,100,200],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [15, 20, 25],
        'max_depth': [4, 5],
        'reg_alpha': [0.01, 0.1, 1],
        'reg_lambda': [0.01, 0.1, 1],
        'subsample': [0.7, 0.8],
        'colsample_bytree': [0.7, 0.8]
    },
    'Gradient Boosting': {
        'n_estimators': [50,100,200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4],
        'subsample': [0.7, 0.8],
        'max_features': ['sqrt', 0.7]
    },
    'KNN': {
        'n_neighbors': [5, 9, 15, 21],
        'weights': ['uniform', 'distance'],
        'metric': ['minkowski', 'euclidean']
    },
    'XGBoost': {
        'n_estimators': [50,100,200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3,4,5],
        'gamma': [0, 0.1, 0.5],
        'subsample': [0.7, 0.8],
        'colsample_bytree': [0.7, 0.8],
        'reg_alpha': [0, 0.01, 0.1],
        'reg_lambda': [0.1, 1, 10]
    },
    'Decision Tree': {
        'criterion': ['gini', 'entropy'],
        'max_depth': [4, 6, 8],
        'min_samples_split': [20, 40, 60],
        'min_samples_leaf': [10, 20, 30]
    },
    'Logistic Regression': {
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'saga']
    }
}

model_names = list(models_sklearn_base.keys())

# Model colors
model_colors = {
    'Random Forest': '#934B43', 'SVM': '#D76364', 'Extra Trees': '#EF7A6D',
    'LightGBM': '#F1D77E', 'Gradient Boosting': '#B1CE46', 'KNN': '#63E398',
    'XGBoost': '#9394E7', 'Decision Tree': '#5F97D2', 'Logistic Regression': '#9DC3E7',
    'TabPFN': '#FF00FF'
}

default_color = '#aaaaaa'

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.labelcolor': 'black',
    'axes.titlecolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'legend.facecolor': 'white',
    'legend.edgecolor': 'lightgray',
    'grid.color': 'lightgray',
    'figure.dpi': 100
})

# Optuna hyperparameter optimization
def generate_optuna_params_sklearn(trial, model_name, param_grid):
    optuna_params = {}
    
    if model_name == 'Logistic Regression' and 'solver' in param_grid:
        current_solver = trial.suggest_categorical('solver', param_grid['solver'])
        optuna_params['solver'] = current_solver
    
    for p_name, p_values in param_grid.items():
        if not isinstance(p_values, list) or not p_values:
            continue
            
        if model_name == 'Logistic Regression':
            if p_name == 'solver':
                continue
            if p_name == 'penalty':
                compatible_penalties = []
                if current_solver == 'liblinear':
                    compatible_penalties = [p for p in p_values if p in ['l1', 'l2']]
                elif current_solver == 'saga':
                    compatible_penalties = [p for p in p_values if p in ['l1', 'l2', 'elasticnet']]
                else:
                    compatible_penalties = p_values
                if not compatible_penalties:
                    continue
                penalty_val = trial.suggest_categorical('penalty', compatible_penalties)
                optuna_params['penalty'] = penalty_val
            else:
                optuna_params[p_name] = trial.suggest_categorical(p_name, p_values)
        else:
            optuna_params[p_name] = trial.suggest_categorical(p_name, p_values)
    
    return optuna_params

def objective_factory(model_class, model_name, X_train_data, y_train_data, param_grid, base_params):
    def objective(trial):
        params = generate_optuna_params_sklearn(trial, model_name, param_grid)
        candidate_params = base_params.copy()
        candidate_params.update(params)
        
        if model_name == 'Logistic Regression':
            solver = candidate_params.get('solver')
            penalty = candidate_params.get('penalty')
            if solver == 'liblinear' and penalty not in ['l1', 'l2']:
                return -np.inf
            elif solver == 'lbfgs' and penalty != 'l2' and penalty is not None:
                return -np.inf
        
        model = model_class(**candidate_params)
        try:
            score = cross_val_score(model, X_train_data, y_train_data, n_jobs=-1, cv=OPTUNA_CV_FOLDS, scoring=OPTUNA_SCORING).mean()
        except:
            return -np.inf
        return score
    return objective

# Training
trained_models = {}
results = {
    'train': {'y_true': y_train, 'y_pred': {}, 'y_proba': {}},
    'test': {'y_true': y_test, 'y_pred': {}, 'y_proba': {}},
    'val': {'y_true': y_val, 'y_pred': {}, 'y_proba': {}}
}
dataset_map_X = {'train': X_train, 'test': X_test, 'val': X_val}
dataset_keys = list(results.keys())
best_params_all_models = {}

for name, model_template in models_sklearn_base.items():
    print(f"Processing {name}...")
    
    if name == 'TabPFN':
        try:
            X_train_np = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
            y_train_np = y_train.values if isinstance(y_train, pd.Series) else y_train
            model_template.fit(X_train_np, y_train_np)
            trained_models[name] = model_template
            best_params_all_models[name] = {"Info": "TabPFN: No hyperparameters tuned."}
        except Exception as e:
            print(f"ERROR training TabPFN: {e}")
            continue
    
    elif name in param_grids:
        current_param_grid = param_grids[name]
        model_class = model_template.__class__
        base_params = model_template.get_params()
        
        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
        objective_func = objective_factory(model_class, name, X_train, y_train, current_param_grid, base_params)
        
        try:
            study.optimize(objective_func, n_trials=OPTUNA_N_TRIALS, timeout=OPTUNA_TIMEOUT)
            best_params = study.best_trial.params
            
            final_params = base_params.copy()
            final_params.update(best_params)
            
            if name == 'Logistic Regression':
                solver = final_params.get('solver')
                penalty = final_params.get('penalty')
                if solver == 'lbfgs':
                    final_params['penalty'] = 'l2'
                elif solver == 'liblinear' and penalty not in ['l1', 'l2']:
                    final_params['penalty'] = 'l2'
            
            final_model = model_class(**final_params)
            final_model.fit(X_train, y_train)
            trained_models[name] = final_model
            best_params_all_models[name] = best_params
        except Exception as e:
            print(f"ERROR during Optuna tuning for {name}: {e}")
            continue
    
    # Predictions
    if name in trained_models:
        current_model = trained_models[name]
        for key in dataset_keys:
            X_data = dataset_map_X.get(key)
            
            try:
                if hasattr(current_model, "predict_proba"):
                    if name == 'TabPFN':
                        X_data_np = X_data.values if isinstance(X_data, pd.DataFrame) else X_data
                        y_proba_raw = current_model.predict_proba(X_data_np)
                    else:
                        y_proba_raw = current_model.predict_proba(X_data)
                    
                    if y_proba_raw.ndim == 2 and y_proba_raw.shape[1] == 2:
                        y_proba = y_proba_raw[:, 1]
                    else:
                        y_proba = y_proba_raw
                    
                    results[key]['y_proba'][name] = y_proba
                    results[key]['y_pred'][name] = (y_proba >= 0.5).astype(int)
                elif hasattr(current_model, "predict"):
                    if name == 'TabPFN':
                        X_data_np = X_data.values if isinstance(X_data, pd.DataFrame) else X_data
                        y_pred = current_model.predict(X_data_np)
                    else:
                        y_pred = current_model.predict(X_data)
                    results[key]['y_pred'][name] = y_pred
                    results[key]['y_proba'][name] = None
            except Exception as e:
                results[key]['y_proba'][name] = None
                results[key]['y_pred'][name] = None

active_model_names = list(trained_models.keys())

# Plotting functions
def plot_roc_curves_with_ci(results_dict, dataset_key, ax, model_names, colors, n_bootstraps=500):
    display_names = {'train': 'Train', 'test': 'Test', 'val': 'Validation'}
    display_name = display_names.get(dataset_key, dataset_key.capitalize())
    
    ax.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Chance')
    mean_fpr = np.linspace(0, 1, 100)
    y_true = results_dict.get('y_true')
    auc_ci_results = []
    
    if y_true is None:
        return auc_ci_results
    
    y_proba_map = results_dict.get('y_proba', {})
    valid_model_names = [name for name in model_names if name in y_proba_map and y_proba_map.get(name) is not None]
    
    for name in valid_model_names:
        y_proba = y_proba_map[name]
        if len(y_true) != len(y_proba) or len(np.unique(y_true)) < 2 or np.isnan(y_proba).all():
            continue
        
        try:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
        except:
            continue
        
        bootstrapped_tprs, bootstrapped_aucs = [], []
        rng = np.random.RandomState(RANDOM_STATE)
        y_proba_np, y_true_np = np.array(y_proba), np.array(y_true)
        mask_valid = ~np.isnan(y_proba_np)
        y_true_valid = y_true_np[mask_valid]
        y_proba_valid = y_proba_np[mask_valid]
        indices_valid = np.arange(len(y_true_valid))
        
        if len(indices_valid) >= 2 and len(np.unique(y_true_valid)) >= 2:
            for _ in range(n_bootstraps):
                if len(indices_valid) == 0:
                    continue
                try:
                    resampled_indices = rng.choice(indices_valid, size=len(indices_valid), replace=True)
                    y_true_boot = y_true_valid[resampled_indices]
                    y_proba_boot = y_proba_valid[resampled_indices]
                    if len(np.unique(y_true_boot)) < 2:
                        continue
                    fpr_b, tpr_b, _ = roc_curve(y_true_boot, y_proba_boot)
                    interp_tpr = np.interp(mean_fpr, fpr_b, tpr_b)
                    interp_tpr[0] = 0.0
                    bootstrapped_tprs.append(interp_tpr)
                    bootstrapped_aucs.append(auc(fpr_b, tpr_b))
                except:
                    continue
        
        if bootstrapped_tprs and bootstrapped_aucs:
            bootstrapped_tprs_arr = np.array(bootstrapped_tprs)
            bootstrapped_aucs_arr = np.array(bootstrapped_aucs)
            mean_tpr = bootstrapped_tprs_arr.mean(axis=0)
            mean_tpr[-1] = 1.0
            tprs_low, tprs_high = np.percentile(bootstrapped_tprs_arr, [2.5, 97.5], axis=0)
            auc_low, auc_high = np.percentile(bootstrapped_aucs_arr, [2.5, 97.5])
        else:
            mean_tpr = np.interp(mean_fpr, fpr, tpr)
            mean_tpr[0], mean_tpr[-1] = 0.0, 1.0
            tprs_low, tprs_high = mean_tpr, mean_tpr
            auc_low, auc_high = np.nan, np.nan
        
        auc_ci_results.append({'Dataset': display_name, 'Model': name, 'AUC': roc_auc, 'AUC_low': auc_low, 'AUC_high': auc_high})
        ax.plot(mean_fpr, mean_tpr, color=colors.get(name, default_color), lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
        
        if not np.isnan(auc_low) and not np.isnan(auc_high):
            ax.fill_between(mean_fpr, tprs_low, tprs_high, color=colors.get(name, default_color), alpha=0.2)
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curves ({display_name} Set)')
    ax.legend(loc='lower right', fontsize='x-small')
    ax.grid(False)
    ax.set_facecolor('white')
    return auc_ci_results

def plot_confusion_matrices(results_dict, dataset_key, fig, axes, model_names, cmap='Blues'):
    display_names = {'train': 'Train', 'test': 'Test', 'val': 'Validation'}
    display_name = display_names.get(dataset_key, dataset_key.capitalize())
    
    y_true = results_dict.get('y_true')
    y_pred_map = results_dict.get('y_pred')
    
    if y_true is None or y_pred_map is None or len(y_true) == 0:
        for ax in axes.flatten():
            fig.delaxes(ax)
        return
    
    axes_flat = axes.flatten()
    num_axes = len(axes_flat)
    num_cols = axes.shape[1] if axes.ndim == 2 else 1
    
    plotted_count = 0
    valid_model_names = [name for name in model_names if name in y_pred_map and y_pred_map.get(name) is not None]
    
    for name in valid_model_names:
        if plotted_count >= num_axes:
            break
        
        ax = axes_flat[plotted_count]
        y_pred = y_pred_map[name]
        
        if y_pred is None or len(y_true) != len(y_pred):
            ax.set_title(f"{name}\n(No Data)", fontsize=10)
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            plotted_count += 1
            continue
        
        try:
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
            disp.plot(ax=ax, cmap=cmap, colorbar=False, values_format='d')
            ax.set_title(name, fontsize=10)
            ax.grid(False)
            
            current_row = plotted_count // num_cols
            current_col = plotted_count % num_cols
            
            if current_col != 0:
                ax.set_ylabel('')
            else:
                ax.set_ylabel('True Label')
            
            last_plotted_row = (len(valid_model_names) - 1) // num_cols
            if current_row < last_plotted_row:
                ax.set_xlabel('')
            else:
                ax.set_xlabel('Predicted Label')
            
            ax.tick_params(axis='both', which='major', labelsize=8)
            plotted_count += 1
        except:
            ax.set_title(f"{name}\n(Error)", fontsize=10)
            ax.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            plotted_count += 1
    
    for j in range(plotted_count, num_axes):
        fig.delaxes(axes_flat[j])
    
    if plotted_count > 0:
        fig.suptitle(f'Confusion Matrices ({display_name} Set)', fontsize=16, y=1.02)

def plot_pr_curves(results_dict, dataset_key, ax, model_names, colors):
    display_names = {'train': 'Train', 'test': 'Test', 'val': 'Validation'}
    display_name = display_names.get(dataset_key, dataset_key.capitalize())
    
    y_true = results_dict.get('y_true')
    y_proba_map = results_dict.get('y_proba')
    
    if y_true is None or y_proba_map is None or len(y_true) == 0:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Precision-Recall Curves ({display_name} Set)\n(No Data)')
        return
    
    if len(np.unique(y_true)) < 2:
        baseline = np.mean(y_true)
        ax.text(0.5, 0.5, f'Single Class (Baseline={baseline:.2f})', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Precision-Recall Curves ({display_name} Set)\n(Single Class)')
        return
    
    baseline = np.sum(y_true) / len(y_true)
    ax.axhline(baseline, linestyle='--', color='grey', label=f'Baseline (Prev={baseline:.2f})')
    
    valid_model_names = [name for name in model_names if name in y_proba_map and y_proba_map.get(name) is not None]
    
    for name in valid_model_names:
        y_proba = y_proba_map[name]
        if len(y_true) != len(y_proba) or np.isnan(y_proba).all():
            continue
        
        mask_valid = ~np.isnan(y_proba)
        y_true_valid = np.array(y_true)[mask_valid]
        y_proba_valid = np.array(y_proba)[mask_valid]
        
        if len(y_true_valid) < 2 or len(np.unique(y_true_valid)) < 2:
            continue
        
        try:
            precision, recall, _ = precision_recall_curve(y_true_valid, y_proba_valid)
            ap_score = average_precision_score(y_true_valid, y_proba_valid)
            ax.plot(recall, precision, color=colors.get(name, default_color), lw=2, label=f'{name} (AP = {ap_score:.3f})')
        except:
            continue
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curves ({display_name} Set)')
    ax.legend(loc='lower left', fontsize='x-small')
    ax.grid(False)
    ax.set_facecolor('white')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])

# Calculate metrics
def calculate_bootstrap_metrics(y_true_boot, y_pred_boot, y_proba_boot):
    metrics = {}
    unique_true = np.unique(y_true_boot)
    n_samples = len(y_true_boot)
    
    if n_samples == 0:
        return {m: np.nan for m in ['Accuracy', 'AUC', 'Sensitivity', 'Specificity', 'Precision', 'F1 Score', 'PPV', 'NPV']}
    
    if y_pred_boot is None or np.isnan(y_pred_boot).any():
        metrics = {m: np.nan for m in ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1 Score', 'PPV', 'NPV']}
    else:
        try:
            metrics['Accuracy'] = accuracy_score(y_true_boot, y_pred_boot)
        except:
            metrics['Accuracy'] = np.nan
        
        if len(unique_true) < 2:
            single_class = unique_true[0]
            metrics['Sensitivity'] = 1.0 if single_class == 1 and np.all(y_pred_boot == 1) else 0.0
            metrics['Specificity'] = 1.0 if single_class == 0 and np.all(y_pred_boot == 0) else 0.0
            try:
                metrics['Precision'] = precision_score(y_true_boot, y_pred_boot, pos_label=1, zero_division=0)
                metrics['PPV'] = metrics['Precision']
            except:
                metrics['Precision'] = np.nan
                metrics['PPV'] = np.nan
            
            if single_class == 0:
                tn_single = np.sum(y_pred_boot == 0)
                fn_single = np.sum(y_pred_boot == 1)
                metrics['NPV'] = tn_single / (tn_single + fn_single) if (tn_single + fn_single) > 0 else 0.0
            elif single_class == 1:
                metrics['NPV'] = np.nan if np.sum(y_pred_boot == 0) == 0 else 0.0
            else:
                metrics['NPV'] = np.nan
            
            try:
                metrics['F1 Score'] = f1_score(y_true_boot, y_pred_boot, pos_label=1, zero_division=0)
            except:
                metrics['F1 Score'] = np.nan
        else:
            try:
                cm = confusion_matrix(y_true_boot, y_pred_boot, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()
                metrics['Sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                metrics['PPV'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                metrics['NPV'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
                metrics['Precision'] = metrics['PPV']
            except:
                metrics['Sensitivity'] = np.nan
                metrics['Specificity'] = np.nan
                metrics['PPV'] = np.nan
                metrics['NPV'] = np.nan
                metrics['Precision'] = np.nan
            
            try:
                metrics['F1 Score'] = f1_score(y_true_boot, y_pred_boot, pos_label=1, zero_division=0)
            except:
                metrics['F1 Score'] = np.nan
    
    if y_proba_boot is not None and not np.isnan(y_proba_boot).all() and len(unique_true) > 1:
        try:
            metrics['AUC'] = roc_auc_score(y_true_boot, y_proba_boot)
        except:
            metrics['AUC'] = np.nan
    else:
        metrics['AUC'] = np.nan
    
    return metrics

def calculate_metrics_with_bootstrap_ci(y_true, y_pred, y_proba, n_bootstraps=100, random_state=42):
    metric_names = ['Accuracy', 'AUC', 'Sensitivity', 'Specificity', 'Precision', 'F1 Score', 'PPV', 'NPV']
    results_nan = {m: (np.nan, np.nan, np.nan) for m in metric_names}
    
    if y_true is None or len(y_true) == 0:
        return results_nan
    
    y_true = np.array(y_true)
    n_samples = len(y_true)
    
    if y_pred is not None:
        y_pred = np.array(y_pred)
        if len(y_pred) != n_samples:
            return results_nan
    
    if y_proba is not None:
        y_proba = np.array(y_proba)
        if len(y_proba) != n_samples:
            y_proba = None
    
    rng = np.random.RandomState(random_state)
    bootstrap_metrics = {m: [] for m in metric_names}
    indices = np.arange(n_samples)
    
    for _ in range(n_bootstraps):
        resampled_indices = rng.choice(indices, size=n_samples, replace=True)
        y_true_boot = y_true[resampled_indices]
        y_pred_boot = y_pred[resampled_indices] if y_pred is not None else None
        y_proba_boot = y_proba[resampled_indices] if y_proba is not None else None
        
        metrics_boot = calculate_bootstrap_metrics(y_true_boot, y_pred_boot, y_proba_boot)
        for m in metric_names:
            bootstrap_metrics[m].append(metrics_boot.get(m, np.nan))
    
    final_metrics = {}
    point_estimates = calculate_bootstrap_metrics(y_true, y_pred, y_proba)
    
    for metric in metric_names:
        values = bootstrap_metrics[metric]
        valid_values = [v for v in values if pd.notna(v)]
        
        if len(valid_values) >= 2:
            try:
                ci_low, ci_high = np.percentile(valid_values, [2.5, 97.5])
            except:
                ci_low, ci_high = np.nan, np.nan
        else:
            ci_low, ci_high = np.nan, np.nan
        
        point_estimate = point_estimates.get(metric, np.nan)
        final_metrics[metric] = (point_estimate, ci_low, ci_high)
    
    return final_metrics

# Calculate metrics
metrics_data_ci = []
for dataset_key in dataset_keys:
    data = results[dataset_key]
    y_true_ds = data.get('y_true')
    y_pred_map = data.get('y_pred')
    y_proba_map = data.get('y_proba')
    
    display_names = {'train': 'Train', 'test': 'Test', 'val': 'Validation'}
    display_name = display_names.get(dataset_key, dataset_key.capitalize())
    
    for model_name in active_model_names:
        y_pred_ds = y_pred_map.get(model_name)
        y_proba_ds = y_proba_map.get(model_name)
        
        if y_pred_ds is None and y_proba_ds is None:
            continue
        
        metrics_with_ci = calculate_metrics_with_bootstrap_ci(
            y_true_ds, y_pred_ds, y_proba_ds,
            n_bootstraps=N_BOOTSTRAPS_CI, random_state=RANDOM_STATE
        )
        
        row = {'Dataset': display_name, 'Model': model_name}
        for metric_name, (point, low, high) in metrics_with_ci.items():
            row[metric_name] = point
            row[f"{metric_name}_low"] = low
            row[f"{metric_name}_high"] = high
        metrics_data_ci.append(row)

metrics_ci_df = pd.DataFrame(metrics_data_ci)

# Generate plots
plot_datasets = dataset_keys
num_datasets = len(plot_datasets)

if num_datasets > 0:
    plot_ncols = min(num_datasets, 3)
    plot_nrows = (num_datasets + plot_ncols - 1) // plot_ncols
    
    # ROC Curves
    fig_roc, axes_roc = plt.subplots(plot_nrows, plot_ncols,
                                     figsize=(6 * plot_ncols, 5 * plot_nrows),
                                     squeeze=False)
    axes_roc_flat = axes_roc.flatten()
    
    for idx, key in enumerate(plot_datasets):
        if key in results:
            ax = axes_roc_flat[idx]
            plot_roc_curves_with_ci(results[key], key, ax, active_model_names, model_colors, n_bootstraps=N_BOOTSTRAPS_ROC_PLOT)
    
    for i in range(len(plot_datasets), len(axes_roc_flat)):
        fig_roc.delaxes(axes_roc_flat[i])
    
    fig_roc.suptitle('ROC Curves with 95% CI', fontsize=16, y=1.03)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if SAVE_PLOTS:
        save_path = os.path.join(PLOT_SAVE_DIR, f"roc_curves.{PLOT_FORMAT}")
        fig_roc.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig_roc)
    
    # Confusion Matrices
    n_models = len(active_model_names)
    if n_models > 0:
        n_cols_cm = 3
        n_rows_cm = (n_models + n_cols_cm - 1) // n_cols_cm
        
        for key in plot_datasets:
            if key in results:
                fig_cm, axes_cm = plt.subplots(n_rows_cm, n_cols_cm,
                                              figsize=(3.5 * n_cols_cm, 3.0 * n_rows_cm),
                                              squeeze=False)
                plot_confusion_matrices(results[key], key, fig_cm, axes_cm, active_model_names)
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                
                if SAVE_PLOTS:
                    save_path = os.path.join(PLOT_SAVE_DIR, f"confusion_matrices_{key}.{PLOT_FORMAT}")
                    fig_cm.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
                plt.close(fig_cm)
    
    # PR Curves
    fig_pr, axes_pr = plt.subplots(plot_nrows, plot_ncols,
                                   figsize=(6 * plot_ncols, 5 * plot_nrows),
                                   sharey=True, squeeze=False)
    axes_pr_flat = axes_pr.flatten()
    
    for idx, key in enumerate(plot_datasets):
        if key in results:
            ax = axes_pr_flat[idx]
            plot_pr_curves(results[key], key, ax, active_model_names, model_colors)
    
    for i in range(len(plot_datasets), len(axes_pr_flat)):
        fig_pr.delaxes(axes_pr_flat[i])
    
    fig_pr.suptitle('Precision-Recall Curves', fontsize=16, y=1.03)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if SAVE_PLOTS:
        save_path = os.path.join(PLOT_SAVE_DIR, f"pr_curves.{PLOT_FORMAT}")
        fig_pr.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig_pr)

# Save predicted probabilities
prob_data = []
for dataset_key in dataset_keys:
    display_names = {'train': 'Train', 'test': 'Test', 'val': 'Validation'}
    dataset_name = display_names.get(dataset_key, dataset_key.capitalize())
    
    y_true = results[dataset_key]['y_true']
    y_proba_map = results[dataset_key]['y_proba']
    
    for i in range(len(y_true)):
        row = {
            'Dataset': dataset_name,
            'Sample_Index': i,
            'True_Label': y_true.iloc[i] if hasattr(y_true, 'iloc') else y_true[i]
        }
        
        for model_name in active_model_names:
            y_proba = y_proba_map.get(model_name)
            if y_proba is not None and i < len(y_proba):
                row[f'{model_name}_Probability'] = y_proba[i]
            else:
                row[f'{model_name}_Probability'] = np.nan
        
        prob_data.append(row)

prob_df = pd.DataFrame(prob_data)
prob_df.to_csv(PREDICTED_PROBABILITIES_CSV_FILENAME, index=False)
print(f"预测概率已保存至 {PREDICTED_PROBABILITIES_CSV_FILENAME}")

# 输出评估指标表
print("\n--- 模型性能指标及95%置信区间 ---")
metrics_to_display = ['AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1 Score', 'PPV', 'NPV']

if not metrics_ci_df.empty:
    output_df = metrics_ci_df.set_index(['Dataset', 'Model'])
    
    for metric in metrics_to_display:
        metric_low = f"{metric}_low"
        metric_high = f"{metric}_high"
        if metric in output_df.columns and metric_low in output_df.columns and metric_high in output_df.columns:
            output_df[metric] = output_df.apply(
                lambda row: f"{row[metric]:.3f} ({row[metric_low]:.3f}-{row[metric_high]:.3f})"
                if pd.notna(row[metric]) and pd.notna(row[metric_low]) and pd.notna(row[metric_high])
                else (f"{row[metric]:.3f}" if pd.notna(row[metric]) else "N/A"),
                axis=1
            )
            output_df = output_df.drop(columns=[metric_low, metric_high], errors='ignore')
    
    final_columns = [col for col in metrics_to_display if col in output_df.columns]
    if final_columns:
        output_df_display = output_df[final_columns]
        print(output_df_display)

# 生成性能热力图
print("\n生成性能指标热力图...")
heatmap_metrics = ['AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1 Score']

if not metrics_ci_df.empty:
    try:
        metrics_pivot = metrics_ci_df.pivot_table(
            index='Model', columns='Dataset', values=heatmap_metrics
        )
        
        if not metrics_pivot.empty:
            plt.figure(figsize=(12, 8))
            sns.heatmap(metrics_pivot, annot=True, fmt=".3f", cmap="rainbow",
                       linewidths=.5, linecolor='lightgrey', cbar=True,
                       vmin=0.0, vmax=1.0)
            plt.title('模型性能指标对比', fontsize=16, pad=20)
            plt.tight_layout()
            
            if SAVE_PLOTS:
                save_path = os.path.join(PLOT_SAVE_DIR, f"metrics_heatmap.{PLOT_FORMAT}")
                plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
            plt.close()
    except:
        pass

print("\n--- 分析完成 ---")