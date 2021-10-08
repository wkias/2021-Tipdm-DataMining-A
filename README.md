## 介绍

以下内容可能存在

```
*.pdf           A题题目或相关资料
setup.sh        创建conda环境
start.sh        训练模型，导出日志
Model.py        模型主体
Exploration.py  数据探索
Config.py       设置
Utils.py        杂项
log/            日志
model/          预训练模型
results/        预测结果及必要文件
zips/           代码、模型、日志、结果 备份
data/           数据集及相关文件
    *.xlsx      原始数据集
    *.csv       原始数据集
    *_X.csv     行业预处理数据
```

## 运行

附件转换为csv UTF-8 文件放入 `data/` 

```
data/
    1.csv
    2.csv
    3.csv
```
使用 `setup.sh` 或手动创建conda环境

```bash
conda create -n w
conda activate w
pip install sklearn pandas seaborn pyod imblearn catboost xgboost lightgbm pretty_errors
```

或根据 `requirements.txt` 安装依赖

```bash
pip install -r requirements.txt
```

运行 `start.sh` 

```bash
./start.sh
```

或手动运行（不会生成日志及备份）

```bash
python Model.py
```