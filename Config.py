from Utils import *

FORMAT = '%(asctime)s\t\t%(filename)s[:%(lineno)d]\t%(funcName)20s\t%(levelname)s: %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)
warnings.filterwarnings("ignore")
pretty_errors.replace_stderr()

sns.set_style("darkgrid")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

signal.signal(signal.SIGINT, exit)
signal.signal(signal.SIGTERM, exit)

data_dir = 'data/'
results_dir = 'results/'
model_dir = 'model/'

# pretrain_model = model_dir + '20210505222308'
pretrain_model = None

TYPE = 'n'
N_JOBS = -1
CV = None
RATE_TO_DROP = 2
RATE_TO_FILL = 4
SELECT_K_BEST = 20
VARIANCE_THRESHOLD = 0.1

final_estimator = LGBMClassifier()

params = dict(
    lr=dict(
        penalty=['l1', 'l2', 'elasticnet', 'none'],
        C=np.arange(0.5, 1.5, 0.1),
        # solver=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        # multi_class=['auto', 'ovr', 'multinomial'],
    ),
    dt=dict(
        # splitter=['best', 'random'],
        criterion=['gini', 'entropy'],
    ),
    svm=dict(
        kernel=['rfb', 'poly', 'sigmod', 'linear'],
    ),
    rf=dict(
        criterion=['gini', 'entropy'],
    ),
    mlp=dict(
        activation=['identity', 'logistic', 'tanh', 'relu'],
        solver=['adam'],
        learning_rate=['constant', 'invscaling', 'adaptive'],
    ),
    sc=dict(
    ),
    cat=dict(
        loss_function=['AUC', 'F1', 'Logloss', 'CrossEntropy']
    ),
    ada=dict(
        algorithm=['SAMME', 'SAMME.R'],
        # random_state=[2021],
        # n_estimatorsint=range(25, 50, 1),
        # learning_rate=np.arange(0, 1, 0.1),
    ),
    bag=dict(
    ),
    xgb=dict(
        booster=['gbtree', 'gblinear', 'dart'],
        # tree_method=['auto', 'exact', 'approx', 'hist'],
        grow_policy=['depthwise', 'lossguide']
    ),
    lgbm=dict(
        boosting_type=['gbdt', 'dart', 'goss', 'rf'],
        # objective=['multiclass', 'regression', 'binary', 'lambdarank'],
    ),
    hbos=dict(
        contamination=[0.01],
    ),
    mo=dict(
        contamination=[0.01],
    ),
)

base_estimators = dict(
    svm=SVC(probability=True),
    lr=LogisticRegression(),
    dt=DecisionTreeClassifier(),
    mlp=MLPClassifier(),

    # # knn=KNN(),
    # # hbos=HBOS(),
    # # lmdd=LMDD(),
    # # copod=COPOD(),
    rbc=RUSBoostClassifier(),
    eec=EasyEnsembleClassifier(),
    bbc=BalancedBaggingClassifier(),
    brc=BalancedRandomForestClassifier(),
    # xbod = XGBOD(n_jobs=-1),
    # # mo=MO_GAAL(contamination=0.01),
    iso=IsolationForest(),

    rf=RandomForestClassifier(),
    ada=AdaBoostClassifier(),
    bag=BaggingClassifier(),
    cat=CatBoostClassifier(devices='gpu',verbose=False),
    xgb=XGBClassifier(),
    lgbm=LGBMClassifier(),
)
