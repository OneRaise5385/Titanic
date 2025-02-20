import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC



def best_svc(X : pd.DataFrame,
             y : pd.DataFrame, 
             kernel = 'rbf',
             scoring='accuracy'):
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

