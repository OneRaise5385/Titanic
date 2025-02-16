import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def data_overview(df, head=True):
    '''数据详情'''
    # 基本信息
    if head:
        display(df.head())
    # 缺失值
    missing_values = df.isnull().sum()
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    overview = pd.DataFrame({
        'Missing Count': missing_values,
        'Missing(%)': missing_percentage.round(1)
    })
    # 各变量取值数量
    unique_counts = df.nunique()
    overview['Unique_counts'] = df.nunique()
    # 各变量取值范围
    unique_values = {col: df[col].unique() for col in df.columns}
    unique_value = []
    for col in unique_counts.index:
        if unique_counts[col] <= 20:
            unique_value.append(unique_values[col])
        else:
            unique_value.append('Much')
    overview['Unique_values'] = unique_value
    # 数据类型
    overview['Dtype'] = df.dtypes
    display(overview)
    # 重复值
    duplicate_count = df.duplicated().sum()
    print(f"Duplicate Rows Count: {duplicate_count} duplicate rows found" 
          if duplicate_count > 0 else "No duplicate rows found")
    
def preprocess_num(df: pd.DataFrame, columns_num: list):
    '''
    连续变量的处理\n
    df: 包含需要处理的数值型数据\n
    columns_cat: 数值型变量的列名
    return: 包含处理前和处理后的数据框
    '''
    scaler = StandardScaler()
    df_standardized = pd.DataFrame(scaler.fit_transform(df[columns_num]), 
                                   columns=[item + 'Standard' for item in df[columns_num].columns])
    df = pd.concat([df, df_standardized], axis=1)
    print([item + 'Standard' for item in df[columns_num].columns])
    return df

def preprocess_cat(df: pd.DataFrame, columns_cat: list):
    '''
    分类变量的处理\n
    df: 需要处理的包含分类变量的数据\n
    columns_cat: 分类变量的列名\n
    return: 包含处理前和处理后的数据框
    '''
    encoder = OneHotEncoder(sparse=False)
    df_cat = pd.DataFrame(encoder.fit_transform(df[columns_cat]),
                         columns=encoder.get_feature_names_out(columns_cat))
    df = pd.concat([df, df_cat], axis=1)
    columns_cat_processed = encoder.get_feature_names_out(columns_cat)
    print(list(columns_cat_processed))
    return df

def impute_num(df, columns_X, columns_y, model):
    '''
    使用模型填充的方式填补**数值型**缺失值\n
    df: 含有缺失值的数据框\n
    columns_X: 填补缺失值时使用的特征X的列名，要求没有缺失值\n
    columns_y: 需要填补的列\n
    model: 填补时使用的sklearn中的模型，可选的模型包括`RandomForestRegressor`, `LinearRegression`, `SVR`\n
    return: 返回填充好的数据框
    '''
    df_X_y = df[columns_X + [columns_y]].dropna().reset_index(drop=True)
    # 划分训练集与测试集
    X_train, X_test, y_train, y_test = train_test_split(df_X_y[columns_X], df_X_y[columns_y], test_size=0.3, random_state=2025)
    # 训练模型
    model.fit(X_train, y_train)
    # 模型效果
    y_pred = model.predict(X_test)
    print(model)
    print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
    if df[columns_y].isna().sum():
        df.loc[df[columns_y].isna(), columns_y] = model.predict(df[columns_X][df[columns_y].isna()])
    return df