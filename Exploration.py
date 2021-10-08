# %%
from Model import *

d1 = pd.read_csv(data_dir + '1.csv')
d2 = pd.read_csv(data_dir + '2.csv')
d3 = pd.read_csv(data_dir + '3.csv')
d1.columns = ['TICKER_SYMBOL', 'INDUSTRY']
d2 = d2.merge(d1, on='TICKER_SYMBOL')
d2 = d2[d2.INDUSTRY == '制造业']
d2 = d2.loc[:, d3[d3.类型 != -1].字段名]
d2 = d2.drop_duplicates()

X = d2.drop(columns=['FLAG'])
y = d2.FLAG

X = preprocessing(X, '制造业')
X = feature_engineering(X)
X, _ = postprocessing(X, y)
X_ = X.copy()
y_ = y.copy()

# %%
X = X_.copy()
y = y_.copy()
X = X[y.notna().values]
y = y[y.notna()].astype(int)
print(X.shape)
i = VarianceThreshold(0.1).fit(X).get_support()
X = X.loc[:, i]
print(X.shape)
X_ = X.copy()
y_ = y.copy()

#%%
X = X_.copy()
y = y_.copy()
i = SelectKBest(mutual_info_classif, k=20).fit(X, y).get_support()
X = X.loc[:, i]
MI = df(mutual_info_classif(X,y))
MI['c'] = list(X.columns)
csv(MI, 'MI')

#%%
X = X_.copy()
y = y_.copy()
a = X.columns
X = StandardScaler().fit_transform(X)
X = pd.DataFrame(X, columns=a)
# X = PCA(20).fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2021)
X_train, y_train = SMOTEENN(n_jobs=-1).fit_resample(X_train, y_train)

# %%
def train(clf, name, attr):
    clf = clf
    clf.fit(X_train, y_train)
    a = df(getattr(clf, attr))
    a['c'] = X.columns
    a = a.sort_values(by=[0])[-20:]
    csv(a, name)
    y_pred = clf.predict_proba(X_test)[:, 1] > 0.5
    print(classification_report(y_test, y_pred))
    print(roc_auc_score(y_pred, y_test))

train(DecisionTreeClassifier(), 'DecisionTree', 'feature_importances_')
train(CatBoostClassifier(), 'CatBoost', 'feature_importances_')
train(XGBClassifier(), 'XGBoost', 'feature_importances_')
train(LGBMClassifier(), 'LGBM', 'feature_importances_')

#%%
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
a = df(getattr(clf, 'coef_')).T
a['c'] = X.columns
a = a.sort_values(by=[0])[-20:]
csv(a, 'SVM')
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print(roc_auc_score(y_pred, y_test))

# %%
gs = GridSearchCV(clf, params['dt'], cv=3).fit(X_train, y_train)
y_pred = gs.best_estimator_.predict_proba(X_test)[:, 1] > 0.5
print(gs.best_params_)
print(classification_report(y_test, y_pred))

# %%
base_estimators = dict(
    # svm=SVC(),
    # lr=LogisticRegression(),
    # xbod = XGBOD(n_jobs=-1),
    # knn=KNN(),
    # dt=DecisionTreeClassifier(),
    # rf=RandomForestClassifier(),
    # mlp=MLPClassifier(),
    # ada=AdaBoostClassifier(),
    # bag=BaggingClassifier(),
    # cat=CatBoostClassifier(verbose=True),
    rbc=RUSBoostClassifier(),
    # eec=EasyEnsembleClassifier(),
    # bbc=BalancedBaggingClassifier(),
    # brc=BalancedRandomForestClassifier(),
    # xgb=XGBClassifier(),
    # lgbm=LGBMClassifier(),
)
clf = BaseModel(base_estimators)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# print(classification_report(y_test, y_pred))

# %%
# y_pred = clf.predict(X_test)
for i in y_pred:
    print()
    print(i)
    print(classification_report(y_test, y_pred[i]))
    print(roc_auc_score(y_pred[i], y_test))

# %%
from Model import *

d1 = pd.read_csv(data_dir + '1.csv')
d2 = pd.read_csv(data_dir + '2.csv')
d3 = pd.read_csv(data_dir + '3.csv')
d1.columns = ['TICKER_SYMBOL', 'INDUSTRY']
d2 = d2.merge(d1, on='TICKER_SYMBOL')
d4 = d2['INDUSTRY'].drop_duplicates()
d4 = ['制造业']
i = d4[0]
d = d2[d2.INDUSTRY == i]
d = d.drop_duplicates()
d = d.loc[:, d3[d3.类型 != -1].字段名]
X = d.drop(columns=['FLAG'])
y = d.FLAG

print(i, '\t'*8, X.shape)

clf = Model(X, y, base_estimators, i).fit()

#%%
clf.report()
importance = clf.get_importance()
importance['c'] = list(clf.columns)
importance.to_csv(data_dir +  i + '_importance.csv', index=False, encoding='gbk')
# y_pred = clf.predict()
# csv(y_pred, i + '_pred')
