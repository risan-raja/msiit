import optuna
from optuna.integration import CatBoostPruningCallback
import catboost
import warnings
import os
from catboost import CatBoostClassifier, Pool, MetricVisualizer
import copy
from sklearnex import patch_sklearn
import numpy as np
import pandas as pd
from category_encoders import (
    BackwardDifferenceEncoder,
    BaseNEncoder,
    BinaryEncoder,
    CatBoostEncoder,
    CountEncoder,
    GLMMEncoder,
    HelmertEncoder,
    JamesSteinEncoder,
    LeaveOneOutEncoder,
    MEstimateEncoder,
    SummaryEncoder,
    TargetEncoder,
    WOEEncoder,
)

warnings.filterwarnings("ignore")

import uuid
from sklearn.experimental import enable_iterative_imputer
from sklearn import set_config
from sklearn.base import clone as model_clone
from sklearn.cluster import *
from sklearn.impute import *
from sklearn.compose import *
from sklearn.cross_decomposition import *
from sklearn.decomposition import *
from sklearn.ensemble import *
from sklearn.feature_selection import *
from sklearn.gaussian_process import *
from sklearn.linear_model import *
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.multioutput import *
from sklearn.multiclass import *
from sklearn.naive_bayes import *
from sklearn.neighbors import *
from sklearn.neural_network import *
from sklearn.pipeline import *
from sklearn.preprocessing import *
from sklearn.kernel_approximation import *
from sklearn.svm import *
from sklearn.tree import *
from sklearn.utils import *
from sklearn.dummy import *
from sklearn.semi_supervised import *
from sklearn.discriminant_analysis import *
from sklearn.covariance import *
from collections import Counter
import sklearn
from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.calibration import *
import joblib
from pprint import pprint as pp

pd.options.compute.use_numba = True
pd.options.compute.use_numexpr = True
pd.options.compute.use_bottleneck = True
pd.options.display.max_columns = 90
set_config(display="diagram")
warnings.filterwarnings("ignore")
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from imblearn.pipeline import make_pipeline as make_imb_pipeline
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.under_sampling import InstanceHardnessThreshold
from imblearn.over_sampling import (
    ADASYN,
    SMOTE,
    RandomOverSampler,
    SVMSMOTE,
    SMOTENC,
    SMOTEN,
    BorderlineSMOTE,
    KMeansSMOTE,
)

plt.style.use("fivethirtyeight")

import seaborn as sns

sns.set()
from joblib import parallel_backend
from joblib.memory import Memory

# patch_sklearn()
KAGGLE_ENV = 1
DATA_INPUT = "/kaggle/input/marketing-strategy-personalised-offer/"
DATA_OUTPUT = "/kaggle/working/"
cwd = os.path.abspath(os.getcwd())
if "mlop3n/Pycharm" in cwd or "u170690" in cwd:
    KAGGLE_ENV = 0
    DATA_INPUT = "kaggle/input/marketing-strategy-personalised-offer/"
    DATA_OUTPUT = "kaggle/working/"
elif 'content' in cwd:
    KAGGLE_ENV = 0
    DATA_INPUT = "/content/drive/MyDrive/projects/msiit/kaggle/input/marketing-strategy-personalised-offer/"
    DATA_OUTPUT = "/content/drive/MyDrive/projects/msiit/kaggle/working/"

CACHE = Memory(DATA_OUTPUT + "joblib", verbose=0)
patch_sklearn()

data = pd.read_csv(DATA_INPUT + "train_data.csv")
eval_data = pd.read_csv(DATA_INPUT + "test_data.csv")


def gen_train_test(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=10
    )
    return X_train, X_test, y_train, y_test


def save_data():
    global data, eval_data
    data.to_parquet(DATA_OUTPUT + "data.parquet")
    eval_data.to_parquet(DATA_OUTPUT + "eval_data.parquet")


def quick_test(X):
    clfs = [
        RandomForestClassifier(class_weight="balanced_subsample", random_state=42),
        DecisionTreeClassifier(class_weight="balanced", random_state=42),
        HistGradientBoostingClassifier(random_state=42),
        LogisticRegressionCV(max_iter=10000, class_weight="balanced", random_state=42),
    ]
    y = data.target
    X_train, X_test, y_train, y_test = gen_train_test(X, y, test_size=0.5)
    for clf in clfs:
        y_pred = clf.fit(X_train, y_train).predict(X_test)
        score = f1_score(y_test, y_pred, average="macro")
        print(f"{clf.__class__.__name__} :: {score}")


def check_RF_perf(X, y):
    clf = RandomForestClassifier(
        class_weight="balanced", n_jobs=24, max_features=None, max_depth=8
    )
    with parallel_backend("threading"):
        scores = cross_validate(
            clf,
            X,
            y,
            cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42),
            n_jobs=24,
            return_train_score=True,
            scoring="f1_macro",
        )
    _ = plt.plot(scores["test_score"], label="TEST")
    _ = plt.plot(scores["train_score"], label="TRAIN")
    _ = plt.legend()


def check_catNB_perf(X, y):
    min_c = X.nunique().astype("int").to_numpy() + 1
    # clf = RandomForestClassifier(class_weight='balanced',
    #                              n_jobs=24,
    # #                              max_features=None,
    #                              )
    class_prior = (y.value_counts() / X.shape[0]).sort_index().to_numpy()
    clf = CategoricalNB(
        fit_prior=True,
        alpha=0.0000003,
        min_categories=min_c,
        #                     class_prior=class_prior
    )
    categories_ = []
    for c in X.columns:
        categories_.append(sorted(list(X[c].unique())))

    work = make_pipeline(OrdinalEncoder(categories=categories_), clf)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.5,
        random_state=42,
        shuffle=True,
        stratify=y,
    )
    with parallel_backend("threading"):
        #     scores = cross_validate(clf,X,y,cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42),n_jobs=24,return_train_score=True,scoring='f1_macro')
        y_pred = work.fit(X_train, y_train).predict(X_test)
        print(classification_report(y_pred, y_test))

data = pd.read_parquet('kaggle/input/gen-data/data.parquet')
eval_data = pd.read_parquet('kaggle/input/gen-data/eval_data.parquet')
nr_data = pd.read_parquet('kaggle/input/gen-data/nr_data.parquet')
nr_eval_data = pd.read_parquet('kaggle/input/gen-data/nr_eval_data.parquet')



X = data[eval_data.columns]
y = data.target
all_data = Pool(
    data = X,
    label = y,
    cat_features=list(eval_data.columns),
    thread_count=-1,
    feature_names=list(eval_data.columns),
)
X_train, X_test, y_train, y_test = gen_train_test(data[eval_data.columns],data.target,0.2)
train_data = Pool(
    data = X_train,
    label = y_train,
    cat_features=list(eval_data.columns),
    thread_count=-1,
    feature_names=list(eval_data.columns),

)
test_data = Pool(
    data = X_test,
    label = y_test,
    cat_features=list(eval_data.columns),
    thread_count=-1,
    feature_names=list(eval_data.columns),
)


def objective(trial: optuna.trial.Trial):
    global train_data, test_data, X_test, y_test
    ##### Classifier Parameters
    acc_metric = catboost.metrics.Accuracy(use_weights=True)
    f1_metric = catboost.metrics.F1(use_weights=True)

    params = {
        "learning_rate":trial.suggest_float('learning_rate',0.0001,0.1),
        "max_ctr_complexity" : trial.suggest_int('max_ctr_complexity',2,15),
        "loss_function": trial.suggest_categorical('loss_function',["CrossEntropy","Logloss"]),
        "bootstrap_type":trial.suggest_categorical('bootstrap_type',["Bayesian","MVS","Bernoulli"]),
        "subsample":trial.suggest_float('subsample',0.5,0.999),
        "depth":trial.suggest_int('depth',2,7),
        "leaf_estimation_backtracking":trial.suggest_categorical('leaf_estimation_backtracking',["AnyImprovement","No"]),
        "grow_policy" :  trial.suggest_categorical("grow_policy" ,["SymmetricTree","Lossguide","Depthwise"]),
        "simple_ctr" : trial.suggest_categorical("simple_ctr",['BinarizedTargetMeanValue','Counter','Borders','Buckets']) ,
        "combinations_ctr": trial.suggest_categorical("combinations_ctr",['BinarizedTargetMeanValue','Counter','Borders','Buckets']) ,
        "l2_leaf_reg": trial.suggest_int('l2_leaf_reg',2,15),
        'leaf_estimation_method': trial.suggest_categorical('leaf_estimation_method',['Newton','Gradient']),
        "random_strength": trial.suggest_int('random_strength',2,15),
        "model_shrink_mode":trial.suggest_categorical('model_shrink_mode',['Constant','Decreasing']),
        'model_shrink_rate':trial.suggest_float('model_shrink_rate',0.1,0.9999),
        'langevin': trial.suggest_categorical('langevin',[True,False]),
        #### Common Parameters
        'thread_count': -1,
        'random_seed': 42,
        'train_dir': 'kaggle/working/joblib',
        'depth': trial.suggest_int('depth',2,10),
        'custom_metric': [f1_metric,acc_metric],
        #'custom_metric': [f1_metric,acc_metric],catboost.metrics.AUC(use_weights=True)],
        'eval_metric': acc_metric,
        'name': 'msiit',
        'iterations': 1000,
        'logging_level': 'Silent',
        'task_type': 'CPU',
        'use_best_model': True,
#         'one_hot_max_size': 256,
#         'early_stopping_rounds': 300,
        'boost_from_average': True
    }
    if params['langevin']:
        params['posterior_sampling'] = trial.suggest_categorical('posterior_sampling',[True,False])
        if params['posterior_sampling']:
            params['model_shrink_mode'] = 'Constant'

    # if params['grow_policy'] != 'Lossguide':
    #     params["score_function"]= trial.suggest_categorical('score_function',["L2","NewtonL2","Cosine","NewtonCosine"]),
    # else:
    #     params["score_function"]= trial.suggest_categorical('score_function',["L2","NewtonL2"])
    if params['bootstrap_type'] =='Bayesian':
        del params['subsample']
    pruning_callback = CatBoostPruningCallback(trial, "Accuracy")
    clf: CatBoostClassifier = CatBoostClassifier(**params)
    clf.fit(train_data,eval_set=test_data,callbacks=[pruning_callback])
    pruning_callback.check_pruned()
    y_pred = clf.predict(X_test)
    f1_score = sklearn.metrics.f1_score(y_test,y_pred,average='macro')
    return f1_score

if __name__=='__main__':
    ### Init Neptune
    # run = neptune.init(project=project, api_token=my_api_token)
    # neptune_callback = optuna_utils.NeptuneCallback(run,log_plot_param_importances=False)
    RDB = "redis://opla:ZnnfZ6EssaXW7Yq$@redis-19917.c240.us-east-1-3.ec2.cloud.redislabs.com:19917"
    storage = optuna.storages.RedisStorage(url=RDB)
    ### init optuna
    warnings.filterwarnings('ignore')
#     sampler = optuna.samplers.TPESampler(constant_liar=True,n_startup_trials=250,n_ei_candidates=24,warn_independent_sampling=False)
    sampler = optuna.samplers.RandomSampler(seed=42)
    study = optuna.create_study(direction="maximize",load_if_exists=True,study_name='TPE_01',
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
            sampler=sampler,
            storage=storage)
    study.optimize(objective,n_jobs=1, n_trials=5,show_progress_bar=False)
