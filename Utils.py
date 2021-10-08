import os
import sys
import tqdm
import uuid
import time
import json
import signal
import pickle
import logging
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
import pretty_errors

from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator
from pyod.models.xgbod import XGBOD
from pyod.models.knn import KNN
from pyod.models.hbos import HBOS
from pyod.models.rod import ROD
from pyod.models.pca import PCA
from pyod.models.lmdd import LMDD
from pyod.models.copod import COPOD
# from pyod.models.mo_gaal import MO_GAAL
from sklearn.svm import SVC
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from catboost import CatBoostRegressor
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTENC
from imblearn.combine import SMOTETomek
from imblearn.combine import SMOTEENN
from imblearn.ensemble import RUSBoostClassifier
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from lightgbm.sklearn import LGBMClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import SpectralClustering
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.feature_selection import VarianceThreshold


# import sklearn
# import xgboost
# import pandas
# import imblearn
# import pyod
# import lightgbm
# import catboost
# logging.info('sklearn\t\t' + sklearn.__version__)
# logging.info('pandas\t\t' + pandas.__version__)
# logging.info('imblearn\t\t' + imblearn.__version__)
# logging.info('xgboost\t\t' + xgboost.__version__)
# logging.info('catboost\t\t' + catboost.__version__)
# logging.info('lightgbm\t\t' + lightgbm.__version__)


def pretty(d):
    logging.info('pretty')
    return json.dumps(d, indent=4, ensure_ascii=False)


def exit(signum, frame):
    logging.info('func Exit')
    print(signum, frame)
    print('EXIT!')
    exit()


def csv(X, f):
    logging.info('to csv:\t' + f)
    X.to_csv(f + '.csv', index=False, encoding='gbk')


def df(x: DataFrame):
    return pd.DataFrame(x)


def na_inf_err(d: DataFrame):
    d[np.isinf(d)] = pd.NA
    return d


def na_table(d: DataFrame):
    na = pd.DataFrame(d.isna().sum()).reset_index()
    nna = pd.DataFrame(d.notna().sum()).reset_index()
    na = na[na[0] > 0]
    na = na.merge(nna, on='index')
    na.columns = ('col', 'isNA', 'notNA')
    return na.sort_values(by='isNA')


def na_plot(d: DataFrame, label=None):
    n = na_table(d)
    x = [i for i in range(0, d.shape[0]+100, 100)]
    y = [n[n.isNA > i].shape[0] for i in x]
    sns.lineplot(x, y)
    plt.legend(label)


def fillna(d: DataFrame):
    logging.info('fillna')
    formula = (
        ('N_CF_OPA_R', 'N_CF_OPERATE_A', '/', 'REVENUE'),
        ('CFSGS_R', 'C_FR_SALE_G_S', '/', 'REVENUE'),
        ('N_CF_OPA_OPAP', 'N_CF_OPERATE_A', '/', 'OPA_PROFIT'),
        # ('N_CF_OPA_OP', 'N_CF_OPERATE_A', '/', 'OPERATE_PROFIT'),
        ('AR_R', 'AR', '/', 'REVENUE'),
        ('ADV_R_R', 'ADVANCE_RECEIPTS', '/', 'REVENUE'),
        ('CASH_CL', 'CASH_C_EQUIV', '/', 'T_CL'),
        ('N_CF_OPA_LIAB', 'N_CF_OPERATE_A', '/', 'T_LIAB'),
        ('TEAP_TL', 'T_EQUITY_ATTR_P', '/', 'T_LIAB'),
        ('TL_TEAP', 'T_LIAB', '/', 'T_EQUITY_ATTR_P'),
        ('OP_TL', 'OPERATE_PROFIT', '/', 'T_LIAB'),
        ('N_CF_OPA_CL', 'N_CF_OPERATE_A', '/', 'T_CL'),
        ('N_CF_OPA_NCL', 'N_CF_OPERATE_A', '/', 'T_NCL'),
        ('OP_CL', 'OPERATE_PROFIT', '/', 'T_CL'),
        ('TSE_TA', 'T_SH_EQUITY', '/', 'T_ASSETS'),
        ('C_TA', 'CASH_C_EQUIV', '/', 'T_ASSETS'),
        ('LT_AMOR_EXP_TA', 'LT_AMOR_EXP', '/', 'T_ASSETS'),
        ('NCA_TA', 'T_NCA', '/', 'T_ASSETS'),
        ('ST_BORR_TA', 'ST_BORR', '/', 'T_ASSETS'),
        ('NCL_TA', 'T_NCL', '/', 'T_LIAB'),
        ('REPAY_TA', 'PREPAYMENT', '/', 'T_ASSETS'),
        # ('AR_TA', 'AR', '/', 'T_ASSETS'),
        ('INVEN_TA', 'INVENTORIES', '/', 'T_ASSETS'),
        ('CL_TA', 'T_CL', '/', 'T_LIAB'),
        ('ADV_R_TA', 'ADVANCE_RECEIPTS', '/', 'T_ASSETS'),
        ('AR_TA', 'AR', '/', 'T_ASSETS'),
        ('TEAP_TA', 'T_EQUITY_ATTR_P', '/', 'T_ASSETS'),
        ('FIXED_A_TA', 'FIXED_ASSETS', '/', 'T_ASSETS'),
        ('CA_TA', 'T_CA', '/', 'T_ASSETS'),
        ('INTAN_A_TA', 'INTAN_ASSETS', '/', 'T_ASSETS'),
        ('AIL_TR', 'ASSETS_IMPAIR_LOSS', '/', 'T_REVENUE'),
        ('NOPG_TR', 'NOPERATE_INCOME', '/', 'T_ASSETS'),
        ('NI_TR', 'N_INCOME', '/', 'T_ASSETS'),
        ('TCOGS_TR', 'T_COGS', '/', 'T_ASSETS'),
        ('TP_TR', 'T_PROFIT', '/', 'T_ASSETS'),
        ('NOPL_TR', 'NOPERATE_EXP', '/', 'T_ASSETS'),
        ('ADMIN_EXP_TR', 'ADMIN_EXP', '/', 'T_ASSETS'),
        ('BTAX_SURCHG_TR', 'BIZ_TAX_SURCHG', '/', 'T_ASSETS'),
        ('IT_TR', 'INCOME_TAX', '/', 'T_ASSETS'),
        ('OP_TP', 'OPERATE_PROFIT', '/', 'T_PROFIT'),
    )
    for i, j, k, l in tqdm.tqdm(formula):
        rows = d.loc[d[i].isna() & d[j].notna() & d[l].notna(), (i, j, l)]
        if k == '/':
            rows.loc[:, i] = rows[j] / rows[l]
        elif k == '+':
            rows.loc[:, i] = rows[j] + rows[l]
        elif k == '-':
            rows.loc[:, i] = rows[j] - rows[l]
        elif k == '*':
            rows.loc[:, i] = rows[j] * rows[l]
        d.loc[d[i].isna() & d[j].notna() & d[l].notna(), i] = rows[i]

    # d.loc[:, 'N_NOPI_TP'] = (d['NOPERATE_INCOME'] - d['NOPERATE_EXP']) / d['T_PROFIT']
    # d.loc[:, 'NCL_WC'] = d['T_NCL'] / (d['T_CA'] - d['T_CL'])
    # d.loc[:, 'EQU_MULTIPLIER'] = 1 / (1 - d['T_LIAB'] / d['T_ASSETS'])
    # d.loc[:, 'CAP_FIX_RATIO'] = (d['T_ASSETS'] - d['T_CA']) / d['T_EQUITY_ATTR_P']

    return d


class NA():

    def __init__(self, estmator=RandomForestRegressor(n_jobs=-1)):
        logging.info('NA init')
        self.estmator = estmator

    def _fit(self, X, y):
        self.model = self.estmator.fit(X, y)

    def _predict(self, X):
        return self.model.predict(X)

    def predict(self, X_):
        logging.info('NA predict' + str(X_.shape))
        X = X_.copy()
        na = na_table(X)
        for i in range(na.shape[0]-1):
            t = time.time()
            col = na.col[i]
            _X = X[na.col.loc[i+1:]].fillna(0)
            _y = X[col]
            X_train = _X.loc[_y.notna(), :]
            y_train = _y.loc[_y.notna()]
            self._fit(X_train, y_train)
            X.loc[_y.isna(), col] = self._predict(_X.loc[_y.isna(), :])
            if i % 10 == 0:
                logging.info('predict\t' + str(i) + '\t' + str(time.time()-t))
        return X

    def fit(self, X):
        logging.info('NA fit')
        return self

    def transform(self, X):
        logging.info('NA transform')
        return self.predict(X)

    def fit_transform(self, X):
        logging.info('NA fit_transform')
        return self.predict(X)
