from Config import *


class BaseModel():

    def __init__(self, estimators: dict, params=params, n_jobs=N_JOBS, cv=CV, verbose=True) -> None:
        warnings.filterwarnings("ignore")
        logging.info('BaseModel init')
        self.estimators = estimators
        self.params_gs = {k: params[k]
                          for k in self.estimators.keys() if k in params}
        self.gs = {k: GridSearchCV(self.estimators[k], self.params_gs[k], n_jobs=n_jobs, cv=cv, verbose=4) for k in self.estimators.keys() if k in params}
        self.cv_results = {k: 0 for k in self.estimators.keys()}
        self.best_params = {k: 0 for k in self.estimators.keys()}
        self.verbose = verbose

    def __len__(self):
        return self.estimators.__len__()
    
    def __iter__(self):
        for i in self.estimators.keys():
            yield (i, self.estimators[i])

    def search(self, X, y, which=None):
        logging.info('BaseModel search')

        def s(k):
            self.estimators[k] = self.gs[k].fit(X, y)
            self.cv_results[k] = self.estimators[k].cv_results_
            self.best_params[k] = self.estimators[k].best_params_
            self.estimators[k] = self.estimators[k].best_estimator_

        if which:
            s(which)
        else:
            for k in self.estimators.keys():
                t = time.time()
                if k in self.params_gs.keys() and np.unique(y).__len__() > 1:
                    logging.info('GridSearch\t' + k)
                    try:
                        s(k)
                    except Exception as e:
                        logging.error(e)
                    logging.info('time\t' + str(time.time()-t))
                else:
                    if np.unique(y).__len__() == 1:
                        logging.warn('np.unique(y).__len__() == 1')
                    logging.info('fit\t' + k)
                    try:
                        self.estimators[k].fit(X, y)
                    except Exception as e:
                        logging.error(e)
                    logging.info('time\t' + str(time.time()-t))
        return self

    def get_best_params(self):
        logging.info('BaseModel get_best_params')
        return self.best_params

    def set_params(self, params: dict):
        logging.info('BaseModel set_params')
        self.params = params
        return self

    def fit(self, X, y):
        logging.info('BaseModel fit')
        for k in self.estimators.keys():
            logging.info('fit\t' + k)
            t = time.time()
            try:
                self.estimators[k].fit(X, y)
            except Exception as e:
                logging.error(e)
            logging.info('time\t' + str(time.time()-t))
        return self

    def predict(self, X):
        logging.info('BaseModel predict')
        pred = {}
        for i in self.estimators.keys():
            try:
                pred[i] = self.estimators[i].predict(X)
            except Exception as e:
                logging.error(e)
        return df(pred)

    def predict_proba(self, X):
        logging.info('BaseModel predict_proba')
        pred = pd.DataFrame([0] * X.shape[0], columns=['a'])
        for i in self.estimators.keys():
            t = time.time()
            if hasattr(self.estimators[i], 'predict_proba'):
                logging.info('predict_proba\t' + i)
                a = self.estimators[i].predict_proba(X)
                logging.debug('predict_proba\t' + str(a.shape))
                try:
                    pred[i] = a[:, 1]
                except IndexError as e:
                    logging.error(e)
                    pred[i] = a
                logging.info('time\t' + str(time.time()-t))
            else:
                logging.info('predict\t' + i)
                pred[i] = self.estimators[i].predict(X)
                logging.info('time\t' + str(time.time()-t))
        return pred.drop(columns=['a'])

    def get(self, estimator=None):
        logging.info('BaseModel get \t' + estimator)
        if estimator:
            return self.estimators[estimator]
        else:
            return self.estimators.values()

    def _get_features_importance(self, estimator):
        logging.info('_get_features_importance\t' + estimator)
        if hasattr(self.estimators[estimator], 'ceof_'):
            return self.estimators[estimator].ceof_
        elif hasattr(self.estimators[estimator], 'feature_importances_'):
            return self.estimators[estimator].feature_importances_
        elif hasattr(self.estimators[estimator], 'features_importance_'):
            return self.estimators[estimator].features_importance_
        else:
            return None

    def get_importance(self, features):
        logging.info('BaseModel get_importance')
        importance = pd.DataFrame(features, columns=['features'])
        for i in self.estimators.keys():
            a = self._get_features_importance(i)
            if a is not None:
                importance[i] = a
        return importance


def features_type(X_):
    logging.info('features_type')
    X = X_.copy()
    cat_features = set(d3[d3.类型 == 1].字段名)
    cat_features.discard('FLAG')
    # for i in na_table(X).col:
    #     X[i+'_isna'] = X[i].isna()
    #     cat_features.add(i+'_isna')
    cat_features = list(cat_features)
    num_features = set(X.columns) - set(cat_features)
    num_features = list(num_features)
    return (X, cat_features, num_features)


def preprocessing(X_, col=set([])):
    logging.info('preprocessing\t' + industry)
    X = X_.copy()
    file = data_dir + industry + '_X.csv'
    if os.path.exists(file):
        logging.info('from csv:\t' + file)
        X = pd.read_csv(file)
    else:
        logging.info('from RF\t')
        X = fillna(X)
        t = na_table(X)
        assert RATE_TO_DROP < RATE_TO_FILL
        X = X.drop(columns=[i for i in t[t.isNA > X.shape[0]/RATE_TO_DROP].col if i not in col])

        X, cat_features, num_features = features_type(X)
        X.loc[:, cat_features] = X[cat_features].fillna('-1')

        Xn = X[num_features]
        Xn = Xn.mask(np.isinf(Xn), pd.NA)
        t = na_table(Xn)
        Xn.loc[:, t[t.isNA <= X.shape[0] / RATE_TO_FILL].col] = NA().predict(Xn[t[t.isNA <= X.shape[0]/RATE_TO_FILL].col])
        for i in t.col:
            Xn[i].fillna(Xn[i].median(), inplace=True)

        X.loc[:, num_features] = Xn
        X.to_csv(file, index=False)
    return X


def feature_engineering(X_):
    logging.info('feature_engineering')
    X = X_.copy()
    #########
    # TO-DO #
    #########
    # X['REVENUE'] = X['REVENUE'].replace(0, 1)
    if 'T_CA' in set(X_.columns):
        X['流动比率'] = X['T_CA'] / X['T_CL']
        X['WORK_CAPTIAL'] = X['T_CA'] - X['T_CL']
        X['流动资产比率'] = X['T_CA'] / X['T_ASSETS']
        X['营运资金比率'] = (X['T_CA'] - X['T_CL']) / X['T_CA']
        X['速动比率'] = (X['T_CA'] - X['INVENTORIES']) / X['T_CL']
        X['营运资金'] = X['T_CA'] - X['T_CL']

    if 'NOTES_RECEIV' in set(X_.columns):
        X['应收类资产比率'] = (X['NOTES_RECEIV'] + X['AR']) / X['T_ASSETS']

    X['总资回报率'] = X['N_INCOME'] / X['T_ASSETS']
    X['ZZZ'] = X['WORK_CAPTIAL'] / X['T_ASSETS']
    X['资产负债率'] = X['T_LIAB'] / X['T_ASSETS']
    X['销售成本利润率'] = X['T_PROFIT'] / X['SELL_EXP']
    # X['利息保障倍数'] = (X['N_INCOME'] + X['INCOME_TAX'] + X['N_INT_EXP']) / X['N_INT_EXP']
    X['产权比率'] = X['T_LIAB'] / X['T_SH_EQUITY']
    X['销售净利率'] = X['N_INCOME'] / X['REVENUE']
    X['净资产'] = X['T_ASSETS'] - X['T_LIAB']
    X['净资产收益率'] = X['N_INCOME'] - X['净资产']
    X['流动负债合计/资产总计'] = X['T_CL'] / X['T_ASSETS']
    X['流动负债比率'] = X['T_CL'] / X['T_LIAB']
    X['留存收益资产比'] = (X['SURPLUS_RESER'] + X['RETAINED_EARNINGS']) / X['T_ASSETS']
    X['总营业成本率'] = X['T_COGS'] / X['T_REVENUE']
    X['成本费用利润率'] = X['T_PROFIT'] / (X['COGS'] + X['SELL_EXP'] + X['ADMIN_EXP'] + X['FINAN_EXP'])
    X['营业外收支净额'] = X['NOPERATE_INCOME'] - X['NOPERATE_EXP']
    X['净利润/利润总额'] = X['N_INCOME'] / X['T_PROFIT']
    X['营业毛利率'] = (X['REVENUE'] - X['COGS']) / X['REVENUE']
    X['营业净利率'] = (X['T_PROFIT'] + X['FINAN_EXP']) / X['T_ASSETS']
    X['应收账款周转率'] = X['AR'] / X['T_REVENUE']
    X['AP_TA'] = X['AP'] / X['T_ASSETS']
    X['COGS_TR'] = X['COGS']/X['T_REVENUE']
    X['SELL_EXP_TR'] = X['SELL_EXP']/X['T_REVENUE']
    X['INV_INC_TR'] = X['INVEST_INCOME']/X['T_REVENUE']
    X['IT_TR'] = X['INCOME_TAX']/X['T_PROFIT']
    X['OP_TR'] = X['OPERATE_PROFIT']/X['T_REVENUE']
    X['FINAN_EXP_TR'] = X['FINAN_EXP']/X['T_REVENUE']
    X['R_TR'] = X['REVENUE']/X['T_REVENUE']
    X['N_CF_OPA_TR'] = X['N_CF_OPERATE_A']/X['T_REVENUE']
    # X = X.fillna(0)
    # X['是否盈利'] = X['REVENUE'] > 0
    # X['zz'] = X['净资产'] > 0
    # a = ['N_CF_OPA_R', 'CFSGS_R', 'AR_R', 'ADV_R_R', 'CASH_CL', 'N_CF_OPA_LIAB', 'TEAP_TL', 'TL_TEAP', 'OP_TL', 'N_CF_OPA_CL', 'N_CF_OPA_NCL', 'OP_CL', 'TSE_TA', 'C_TA', 'LT_AMOR_EXP_TA', 'NCA_TA', 'ST_BORR_TA', 'NCL_TA','REPAY_TA', 'INVEN_TA', 'CL_TA', 'ADV_R_TA', 'AR_TA', 'TEAP_TA', 'FIXED_A_TA', 'CA_TA', 'INTAN_A_TA', 'AIL_TR', 'NOPG_TR', 'NI_TR', 'TCOGS_TR', 'TP_TR', 'NOPL_TR', 'ADMIN_EXP_TR', 'BTAX_SURCHG_TR', 'IT_TR', 'OP_TP', 'N_CF_OPA_TR', 'NCL_WC', 'EQU_MULTIPLIER', 'CAP_FIX_RATIO', 'AP_TA', 'COGS_TR', 'SELL_EXP_TR', 'INV_INC_TR', 'IT_TR', 'OP_TR', 'FINAN_EXP_TR', 'N_NOPI_TP', 'R_TR']
    # a = ['N_CF_OPA_R', 'CFSGS_R', 'AR_R', 'ADV_R_R', 'CASH_CL', 'N_CF_OPA_LIAB', 'TEAP_TL', 'TL_TEAP', 'OP_TL', 'N_CF_OPA_CL', 'N_CF_OPA_NCL', 'OP_CL', 'TSE_TA', 'C_TA', 'LT_AMOR_EXP_TA', 'NCA_TA', 'ST_BORR_TA', 'NCL_TA','REPAY_TA', 'INVEN_TA', 'CL_TA', 'ADV_R_TA', 'AR_TA', 'TEAP_TA', 'FIXED_A_TA', 'CA_TA', 'INTAN_A_TA', 'AIL_TR', 'NOPG_TR', 'NI_TR', 'TCOGS_TR', 'TP_TR', 'NOPL_TR', 'ADMIN_EXP_TR', 'BTAX_SURCHG_TR', 'IT_TR', 'OP_TP', 'N_CF_OPA_TR', 'NCL_WC', 'EQU_MULTIPLIER', 'CAP_FIX_RATIO', 'AP_TA', 'COGS_TR', 'SELL_EXP_TR', 'INV_INC_TR', 'IT_TR', 'OP_TR', 'FINAN_EXP_TR', 'N_NOPI_TP', 'R_TR']
    # a.extend(list(X.iloc[:, -25:].columns))
    # return X.loc[:, a]
    # return X.iloc[:,-25:]
    logging.info(X.shape)
    X = X.mask(np.isinf(X.values), np.median(X.values))
    X = X.fillna(0)
    return X


def postprocessing(X, y, features=None):
    logging.info('postprocessing\t' + industry)
    v = VarianceThreshold(VARIANCE_THRESHOLD).fit(X)
    i = v.get_support()
    X = X.loc[:, i]
    i = SelectKBest(mutual_info_classif, k=SELECT_K_BEST).fit(
        X[y.notna().values], y[y.notna().values]).get_support()
    MI = df(mutual_info_classif(X.loc[y.notna().values, i], y[y.notna().values]))
    MI['c'] = list(X.loc[:, i].columns)
    csv(MI, results_dir + file_name + industry + '_MI')
    return X, v


class Model():

    def __init__(self, X, y, base_estimators, final_estimator=final_estimator):
        logging.info('Model init')
        self.X_ = self._process(X, y)
        self.X = self.X_[y.notna().values]
        self.X_pred = self.X_[y.isna().values]
        self.y = y[y.notna().values]
        self._processed = False
        self.clf = BaseModel(deepcopy(base_estimators))
        self.final_estimator = final_estimator

    def _postprocess(self):
        logging.info('Model _postprocess')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state=2021)
        try:
            self.X_train, self.y_train = SMOTE(n_jobs=N_JOBS).fit_resample(self.X_train, self.y_train)
        except Exception as e:
            logging.error(e)

    def _process(self, X, y=None):
        logging.info('Model _process')
        if y is not None:
            X = preprocessing(X)
            X = feature_engineering(X)
            X = postprocessing(X, y)
            self.columns = list(X.columns)
            self.scaler = StandardScaler().fit(X)
            X = self.scaler.transform(X)
            self._processed = True
        else:
            X = preprocessing(X, col=self.columns)
            X = feature_engineering(X)
            X = X[self.columns]
            X = self.scaler.transform(X)
        return X

    def fit(self, optimal=False):
        logging.info('Model fit')
        if not self._processed:
            self._postprocess()
        if optimal:
            self.clf.search(self.X_train, self.y_train)
        else:
            self.clf.fit(self.X_train, self.y_train)
        self.final_estimator.fit(
            self.clf.predict_proba(self.X_train), self.y_train)
        return self

    def predict(self, X=None):
        logging.info('Model predict')
        if X is not None:
            X = self._process(X)
            y_pred = self.clf.predict(X)
            y_pred_proba = self.clf.predict_proba(X)
            y_pred['stacking'] = self.final_estimator.predict(y_pred_proba)
            return (X, y_pred)
        else:
            self.y_pred = self.clf.predict(self.X_pred)
            self.y_pred_proba = self.clf.predict_proba(self.X_pred)
            self.y_pred['stacking'] = self.final_estimator.predict(self.y_pred_proba)
            return self.y_pred
    
    def refit(self):
        self.clf.fit(self.X, self.y)

    def report(self, X=None):
        logging.info('Model report')
        self.y_pred = self.clf.predict(self.X_test)
        self.y_pred_proba = self.clf.predict_proba(self.X_test)
        self.y_pred['stacking'] = self.final_estimator.predict(self.y_pred_proba)
        self.y_pred = self.y_pred.astype(int)
        print('\n', self.final_estimator, '\n')
        print(pretty(self.clf.best_params), '\n')
        for i in self.y_pred.columns:
            if np.unique(self.y_test).__len__() > 1:
                print(i + '\t\tAUC\t', roc_auc_score(self.y_test, self.y_pred[i]), '\n')
                print(classification_report(self.y_test, self.y_pred[i]), '\n')
            else:
                logging.warn('Only one class present in y_true')

    def get_importance(self):
        logging.info('Model get_importance')
        return self.clf.get_importance(self.columns)


def main(s):
    logging.info('main(' + s.__str__() + ')')

    global file_name
    if s.__len__() > 1:
        file_name = s[1]
    else:
        # file_name = uuid.uuid4().__str__()
        file_name = time.strftime('%Y%m%d%H%M%S', time.localtime())
    logging.info(file_name)

    global d2
    d1.columns = ['TICKER_SYMBOL', 'INDUSTRY']
    d2 = d2.merge(d1, on='TICKER_SYMBOL')

    if TYPE == 0:
        d4 = {'所有行业': d2}
    elif TYPE == 1:
        d4 = {'制造业': d2[d2.INDUSTRY == '制造业']}
    elif TYPE == 2:
        d4 = {
            '制造业A': d2[d2.INDUSTRY == '制造业'], 
            '非制造业': d2[d2.INDUSTRY != '制造业']
        }
    elif TYPE == 'n':
        d4 = d2['INDUSTRY'].drop_duplicates()
        # error = set(['建筑业', '交通运输、仓储和邮政业', '房地产业', '电力、热力、燃气及水生产和供应业'])
        error = set([])
        completed = set([])
        d4 = {k: d2[d2.INDUSTRY==k] for k in d4 if k not in error.union(completed)}

    for i in d4.keys():
        global industry
        industry = i
        d = d4[i]
        d = d.drop_duplicates()
        e = ['TICKER_SYMBOL', 'INDUSTRY']
        X = d.loc[:, d3[d3.类型 != -1].字段名]
        X = X.drop(columns=['FLAG'])
        y = d.FLAG

        logging.info('\n\n' + '-' * 100 + '\n')
        logging.info(i + '\t' + str(X.shape))

        try:
            clf = Model(X, y, base_estimators)
            print('\nDefault parameters')
            clf.fit(optimal=False)
            clf.report()
            print('\nParameters tunning')
            clf.fit(optimal=True)
            clf.report()
            clf.refit()
            importance = clf.get_importance()
            y_pred = clf.predict()
            e.extend(y_pred.columns)
            y_pred['TICKER_SYMBOL'] = d.loc[y.isna(), 'TICKER_SYMBOL'].reset_index(drop=True)
            y_pred['INDUSTRY'] = d.loc[y.isna(), 'INDUSTRY'].reset_index(drop=True)
            y_pred = y_pred.loc[:, e]
            csv(importance, results_dir + file_name + i + '_importance')
            csv(y_pred, results_dir + file_name + i + '_pred')
            with open(results_dir + file_name + i + '_params', 'w') as f:
                f.write(pretty(clf.clf.best_params.__str__()))
            with open(model_dir + file_name + i, 'wb') as f:
                pickle.dump(clf, f)
        except Exception as e:
            logging.error(str(e) + '\n')
            continue

def eval(i, d):
    global industry
    industry = i
    global d3
    d3 = d

if __name__ == '__main__':    
    d1 = pd.read_csv(data_dir + '1.csv')
    d2 = pd.read_csv(data_dir + '2.csv')
    d3 = pd.read_csv(data_dir + '3.csv')
    main(sys.argv)
