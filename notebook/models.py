import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import lightgbm as lgb

def best_knn_clf(X : pd.DataFrame,
             y : pd.DataFrame, 
             scoring='accuracy',
             metrics = 'binary'):
    '''
    ！！！！有点问题，暂时先不用管
    K近邻算法参数寻优\n
    X：输入模型的特征\n
    y：输入模型的标签\n
    scoring：模型的评价标准，可取的值为 ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']\n 
    metrics:
        numeric: 数值型数据的距离
        binary: 二值型数据的距离
        text: 文本或高维稀疏数据的距离
        
    return: 输出最佳的模型
    '''
    # 定义参数网格
    if metrics == 'numeric':
        metric = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
    elif metrics == 'binary':
        metric = ['hamming', 'jaccard']
    elif metrics == 'text':
        metric = ['cosine', 'correlation']
        
    param_grid = {'n_neighbors' : [3, 5, 7, 9],
                  'weights' : ['uniform', 'distance'],
                  'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
                  'leaf_size' : [10, 20, 30, 40, 50],
                  'p' : [1, 2],
                  'metric' : metric}
    model = KNeighborsClassifier()

    # 使用GridSearchCV进行超参数调优
    grid_search = GridSearchCV(model, param_grid, scoring=scoring, 
                               n_jobs=-1, cv=5)
    grid_search.fit(X, y)
    print(f'KNN Best Params: ',grid_search.best_params_)
    print(f'KNN Best Score: ', grid_search.best_score_)
    return grid_search.best_estimator_


def best_bayes_clf(X : pd.DataFrame,
                   y : pd.DataFrame, 
                   scoring = 'accuracy',
                   model_name = 'GaussianNB'):
    '''
    ***朴素贝叶斯*** 分类器模型寻优\n
    X：输入模型的特征\n
    y：输入模型的标签\n
    scoring：模型的评价标准，可取的值为 ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']\n
    model: 
        GaussianNB：假设特征服从高斯分布（正态分布）。
        MultinomialNB：适用于离散特征（如文本分类中的词频或 TF-IDF 值）。
        BernoulliNB：适用于二值特征（如文本分类中的是否出现某个词）。
        
    return: 输出最佳的随机森林模型
    '''
    # 模型
    if model_name == 'GaussianNB':
        model = GaussianNB()
        param_grid = {'var_smoothing':[1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}
        
    elif model_name == 'MultinomialNB':
        model = MultinomialNB()
        param_grid = {'alpha': [0.1, 0.5, 1.0, 1.5, 2.0],  # 平滑参数
                      'fit_prior': [True, False]}  # 是否学习类别的先验概率
        
    elif model_name == 'BernoulliNB':
        model = BernoulliNB()
        param_grid = {'alpha': [0.1, 0.5, 1.0, 1.5, 2.0],  # 平滑参数
                      'fit_prior': [True, False],          # 是否学习类别的先验概率
                      'binarize': [0.0, 0.5, 1.0]}          # 特征二值化的阈值
    
    # 使用GridSearchCV进行超参数调优
    grid_search = GridSearchCV(model, param_grid, scoring=scoring, 
                               n_jobs=-1, cv=5, verbose=2)
    grid_search.fit(X, y)   
    # 打印最佳参数和得分
    print(f'{model_name} Best Params: ',grid_search.best_params_)
    print(f'{model_name} Best Score: ', grid_search.best_score_)      
    return grid_search.best_estimator_


def best_svm_clf(X : pd.DataFrame,
             y : pd.DataFrame, 
             scoring='accuracy',
             kernel = 'rbf'):
    '''
    支持向量机参数寻优\n
    X：输入模型的特征\n
    y：输入模型的标签\n
    kernel：选择使用哪种和函数，取值为 ['rbf', 'linear', 'poly', 'sigmoid']\n
    scoring：模型的评价标准，可取的值为 ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    return: 输出最佳的模型
    '''
    # 定义参数网格
    param_grid = {'C': [0.025, 0.05, 0.1, 0.5, 1, 10],
                'kernel': [kernel],
                'degree': [2, 3, 4],
                'gamma': ['scale', 'auto'],
                'coef0': [0, 0.1, 0.5, 1]}
    model = SVC()

    # 使用GridSearchCV进行超参数调优
    grid_search = GridSearchCV(model, param_grid, scoring=scoring, 
                               n_jobs=-1, cv=5)
    grid_search.fit(X, y)
    print(f'svc_{kernel} Best Params: ',grid_search.best_params_)
    print(f'svc_{kernel} Best Score: ', grid_search.best_score_)
    return grid_search.best_estimator_


def best_decisiontree_clf(X : pd.DataFrame,
                          y : pd.DataFrame, 
                          scoring='accuracy'):
    '''
    ***决策树*** 分类器模型寻优\n
    X：输入模型的特征\n
    y：输入模型的标签\n
    scoring：模型的评价标准，可取的值为 ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']\n
    return: 输出最佳的决策树模型
    '''
    # 参数网格
    param_grid = {"criterion": ["gini", "entropy", 'log_loss'],  # 分裂标准
                  "splitter": ["best", "random"],    # 分裂策略
                  "max_depth": [None, 3, 5, 10],     # 树的最大深度
                  "min_samples_split": [2, 5, 10],   # 节点分裂的最小样本数
                  "min_samples_leaf": [1, 2, 5],     # 叶子节点的最小样本数
                  "min_weight_fraction_leaf": [0.0],  # 叶子节点的最小权重比例 (样本不平衡时用)
                  "max_features": [None, "sqrt", "log2", 0.5],  # 分裂时考虑的最大特征数
                  "max_leaf_nodes": [None, 10, 20],  # 最大叶子节点数
                  "min_impurity_decrease": [0.0, 0.01, 0.1],  # 最小不纯度减少量
                  "class_weight": [None, "balanced"],  # 类别权重
                  "ccp_alpha": [0.0, 0.01, 0.1]}     # 代价复杂度剪枝参数
    # 模型
    model = DecisionTreeClassifier(random_state=42)
    
    # 使用GridSearchCV进行超参数调优
    grid_search = GridSearchCV(model, param_grid, scoring=scoring, 
                               n_jobs=-1, cv=5, verbose=2)
    grid_search.fit(X, y)
    
    # 打印最佳参数和得分
    print(f'DecisionTree Best Params: ',grid_search.best_params_)
    print(f'DecisionTree Best Score: ', grid_search.best_score_)
    
    return grid_search.best_estimator_

def best_randomforest_clf(X : pd.DataFrame,
                          y : pd.DataFrame, 
                          scoring = 'accuracy',
                          n_estimators = 100,  # 树不断增加，一般认为就不会过拟合，尽量增加树的棵树
                          min_weight_fraction_leaf = 0,
                          min_impurity_decrease=[0.0],
                          bootstrap = True,
                          oob_score = False,
                          class_weight = None):
    '''
    ***随机森林*** 分类器模型寻优\n
    X：输入模型的特征\n
    y：输入模型的标签\n
    scoring：模型的评价标准，可取的值为 ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']\n
    min_weight_fraction_leaf: 叶子节点的最小样本数\n
    min_impurity_decrease: 最小不纯度减少量\n
    bootstrap：是否使用有放回抽样\n
    oob_score: 袋外样本评估模型\n
    class_weight: 类别权重默认为None，可选 ['balanced']\n
    return: 输出最佳的随机森林模型
    '''
    # 参数网格
    param_grid = {"criterion": ["gini", "entropy", 'log_loss'],
                  'max_depth': [None, 10, 20, 30],
                  'min_samples_split': [2, 5, 10],
                  'min_samples_leaf': [1, 2, 4],
                  'max_features': [None, "sqrt", "log2", 0.5],
                  "max_leaf_nodes": [None, 10, 20],
                  "min_impurity_decrease": min_impurity_decrease,}
    # 模型
    model = RandomForestClassifier(n_estimators=n_estimators,
                                   min_weight_fraction_leaf=min_weight_fraction_leaf,
                                   bootstrap=bootstrap,
                                   oob_score=oob_score,
                                   n_jobs=-1,
                                   random_state=42,
                                   class_weight=class_weight)
    
    # 使用GridSearchCV进行超参数调优
    grid_search = GridSearchCV(model, param_grid, scoring=scoring, 
                               n_jobs=-1, cv=5, verbose=2)
    grid_search.fit(X, y)
    
    # 打印最佳参数和得分
    print(f'RandomForest Best Params: ',grid_search.best_params_)
    print(f'RandomForest Best Score: ', grid_search.best_score_)
    
    return grid_search.best_estimator_

def best_adaboost_clf(X : pd.DataFrame,
                      y : pd.DataFrame, 
                      scoring='accuracy',
                      base_estimator = 'deprecated',
                      algorithm = 'SAMME.R'):
    '''
    Adaboost参数寻优\n
    X：输入模型的特征\n
    y：输入模型的标签\n
    scoring：模型的评价标准，可取的值为 ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']\n
    base_estimator: 
        'SAMME': 输出类型为离散的类别标签，权重更新依据为分类错误率，要求弱分类器输出类别标签
        'SAMME.R': 输出类型为离散的类别的概率分布，权重更新依据为估计概率，要求弱分类器输出概率（实现 predict_proba 方法）
    
    return: 输出最佳的模型
    '''
    # 定义参数网格
    param_grid = {'n_estimators': [1000],}  # 对于集成算法，这个越大越好
    model = AdaBoostClassifier(algorithm=algorithm, 
                               base_estimator=base_estimator)

    # 使用GridSearchCV进行超参数调优
    grid_search = GridSearchCV(model, param_grid, scoring=scoring, 
                               n_jobs=-1, cv=5)
    grid_search.fit(X, y)
    print(f'AdaBoost Best Params: ',grid_search.best_params_)
    print(f'AdaBoost Best Score: ', grid_search.best_score_)
    return grid_search.best_estimator_

def best_lightgbm_clf(X : pd.DataFrame,
                      y : pd.DataFrame, 
                      scoring='accuracy',
                      boosting_type = 'gbdt'):
    '''
    lightgbm参数寻优\n
    X：输入模型的特征\n
    y：输入模型的标签\n
    scoring：模型的评价标准，可取的值为 ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']\n
    boosting_type:
        gbdt: 训练速度快，适用于大多数场景
        dart: 数据量小且容易过拟合的场景
        rf: 高维数据或需要强正则化的场景，防止过拟合的能力强
        
    return: 输出最佳的模型
    '''
    # 定义参数网格
    param_grid = {'num_leaves': [31, 63],  # [31, 63]
                  'max_depth': [3, 7, -1],  # [3, 5, 7, -1]
                  'min_split_gain': [0, 0.1, 0.2],  # [0, 0.1, 0.2]
                  'min_child_weight': [0.001, 0.1, 1],  # [0.001, 0.01, 0.1, 1]
                  'min_child_samples': [10, 20, 50],  # [10, 20, 50]
                  'subsample_freq': [0],  # [5, 10]
                  'colsample_bytree' : [0.8, 1],  # [0.8, 1]
                  'reg_alpha': [0, 0.1],  # [0, 0.1]
                  'reg_lambda': [0, 0.1]}  # [0, 0.1]
    
    model = lgb.LGBMClassifier(boosting_type=boosting_type, 
                               learning_rate=0.01,
                               n_estimators=200,  # 1000
                               n_jobs=-1)
    
    # 使用GridSearchCV进行超参数调优
    grid_search = GridSearchCV(model, param_grid, scoring=scoring, 
                               n_jobs=-1, cv=5)
    grid_search.fit(X, y)
    print(f'LightGBM Best Params: ',grid_search.best_params_)
    print(f'LightGBM Best Score: ', grid_search.best_score_)
    return grid_search.best_estimator_