import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB


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
    print(f'GaussianNB Best Params: ',grid_search.best_params_)
    print(f'GaussianNB Best Score: ', grid_search.best_score_)      
    return grid_search.best_estimator_

def best_svm_clf(X : pd.DataFrame,
             y : pd.DataFrame, 
             scoring='accuracy',
             kernel = 'rbf'):
    '''
    SVC参数寻优\n
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
