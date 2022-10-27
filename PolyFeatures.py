import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import *
from sklearn.pipeline import *
from sklearn.compose import *
from joblib import parallel_backend
from msiit_db import MyDB
from sklearn.naive_bayes import *
from sklearn.linear_model import *
from sklearn.feature_selection import *
from sklearn.model_selection import *
from sklearn.metrics import *
db = MyDB()
def gen_train_test(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=10
    )
    return X_train, X_test, y_train, y_test

nominal = [x for x in db.eval_data.columns if "nominal_" in x]
ordinal = [x for x in db.eval_data.columns if "ordinal_" in x]
binary = [x for x in db.eval_data.columns if "binary_" in x]
interval = [x for x in db.eval_data.columns if "interval_" in x]
data_groups = [nominal,ordinal,binary,interval]

def gen_col_data(df):
    idx = df.index.to_flat_index().to_list()
    col_names = [f'col_{i}' for i in range(7)]
    df = pd.DataFrame(columns=col_names)
    for j,i_ in enumerate(idx):
        for k,c in zip(i_,col_names):
            df.loc[j,c] =k
    return df
orig_columns = ['binary__offer_expiration',
 'ordinal__income_range',
 'ordinal__no_visited_cold_drinks',
 'binary__travelled_more_than_15mins_for_offer',
 'ordinal__restaur_spend_less_than20',
 'nominal__marital_status',
 'nominal__restaurant_type',
 'ordinal__age',
 'binary__prefer_western_over_chinese',
 'binary__travelled_more_than_25mins_for_offer',
 'ordinal__no_visited_bars',
 'binary__gender',
 'nominal__car',
 'binary__restuarant_same_direction_house',
 'binary__cooks_regularly',
 'nominal__customer_type',
 'ordinal__qualif',
 'binary__is_foodie',
 'ordinal__no_take_aways',
 'nominal__job_industry',
 'binary__restuarant_opposite_direction_house',
 'binary__has_children',
 'ordinal__type_of_rest_rating',
 'interval__temperature',
 'ordinal__restaur_spend_greater_than20',
 'interval__travel_time',
 'interval__season',
 'ordinal__dest_distance',
 'binary__prefer_home_food',
 ]

master_X = pd.concat([db.data[db.eval_data.columns],db.eval_data],ignore_index=True,axis=0).astype(np.uint16)
enc = OrdinalEncoder()
enc.fit(master_X.loc[:,ordinal])
db.data[ordinal] = enc.transform(db.data[ordinal]).astype(np.uint16)
db.eval_data[ordinal] = enc.transform(db.eval_data.loc[:,ordinal]).astype(np.uint16)
db.nr_data[db.nr_eval_data.columns] = db.data[db.nr_eval_data.columns].astype(np.uint16)
db.nr_eval_data = db.eval_data[db.nr_eval_data.columns].astype(np.uint16)
master_X = pd.concat([db.data[db.eval_data.columns],db.eval_data],ignore_index=True,axis=0).astype(np.uint16)

# ct = make_column_transformer(
#     (OneHotEncoder(sparse=True),make_column_selector(pattern='nominal_*|interval__season|ordinal_*')),
#     remainder='passthrough',n_jobs=-1
# )
# enc = make_pipeline(ct,PolynomialFeatures(include_bias=False,interaction_only=True,degree=3,order='F'),VarianceThreshold())

ct = make_column_transformer(
    (Normalizer(),make_column_selector(pattern='interval__*|ordinal_*')),
    (OneHotEncoder(sparse=True),make_column_selector(pattern='nominal_*|interval__*')),
    remainder='passthrough',
    sparse_threshold=50,
    n_jobs=-1
)
enc = make_pipeline(ct,PolynomialFeatures(include_bias=False,interaction_only=True,degree=3,order='F'),VarianceThreshold())

# print(master_X.head())
with parallel_backend('loky',):
    enc.fit(master_X[db.orig_columns])
    X = db.data[db.orig_columns]
    X_enc = enc.transform(X)
X_enc = pd.DataFrame.sparse.from_spmatrix(X_enc,columns=list(enc.get_feature_names_out()))


with open('kaggle/working/poly_features.pkl','wb') as fp:
    pickle.dump(X_enc,fp,protocol=5)
    
clf = RidgeClassifierCV(scoring='f1_macro',fit_intercept=False,class_weight='balanced',cv=RepeatedStratifiedKFold())
y = db.y
X_train, X_test, y_train, y_test = gen_train_test(X_enc,y,test_size=0.2)
with parallel_backend('loky',n_jobs=-1):
    y_pred=clf.fit(X_train,y_train).predict(X_test)
print(clf.__class__.__name__)
print(classification_report(y_test,y_pred))

clfs = [
    BernoulliNB(binarize=False,alpha=1e-09,fit_prior=True,),
    ComplementNB(alpha=1e-09,fit_prior=True,),
#     GaussianNB(),
    MultinomialNB(alpha=1e-09,fit_prior=True,),
    LogisticRegressionCV(max_iter=1000000,
                         fit_intercept=False,
                         scoring='f1_macro',n_jobs=-1,             
                         class_weight='balanced',
                         random_state=42,
                         solver='saga',
                         penalty='elasticnet',
                         l1_ratios=np.linspace(0,1,num=10)
                        )

       ]
for clf in clfs:
    with parallel_backend('loky',n_jobs=-1):
        y_pred=clf.fit(X_train,y_train).predict(X_test)
    print(clf.__class__.__name__)
    print(classification_report(y_test,y_pred))

# clf = BernoulliNB(binarize=False,alpha=1e-09,fit_prior=True,)
