"""
输入: 
    openml的id
    test_size: 测试集比例
    val_size: 验证集比例
    random_state: 随机种子

输出: 
    train_loader: 训练集
    val_loader: 验证集
    test_loader: 测试集

处理流程：
    1. 根据openml的id, 加载数据集
    2. 自动处理分类特征和连续特征
    3. 自动划分训练集、验证集、测试集
    4. 返回训练集、验证集、测试集、y值缩放器

备选数据:
    1. student_performance: 学生成绩  id=46589
    2. abalone: 鲍鱼年龄  id=44956
    3. white_wine: 白葡萄酒质量  id=44971
    4. red_wine: 红葡萄酒质量  id=44972
    5. energy_efficiency: 能源效率  id=43918
    6. naval_propulsion: 海军推进  id=44969
    7. Estimation_of_Obesity_Level: 估计肥胖水平  id=46840
"""

import openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
from scipy.io import arff

def load_data(openml_id, test_size=0.2, val_size=0.2, random_state=42):
    if openml_id == 42571:
        # 本地加载 ARFF 数据集
        dpath = "/home/zhangjunyu/zjywork/zjycode/DKRR_version/datasets/42571/dataset.arff"
        raw_data, meta = arff.loadarff(dpath)
        df = pd.DataFrame(raw_data)
        # 将字节类型的分类特征解码为字符串，便于后续 OneHot 处理
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].apply(lambda v: v.decode('utf-8') if isinstance(v, bytes) else v)

        # 识别目标列：优先使用常见命名，否则使用最后一列
        candidate_targets = ['class', 'target', 'y', 'label']
        target_col = next((c for c in candidate_targets if c in df.columns), df.columns[-1])

        X = df.drop(columns=[target_col])
        y = df[target_col]
        data_name = f"local_arff_{openml_id}"
    else:
        dataset = openml.datasets.get_dataset(openml_id)
        data_name = dataset.name
        # 处理数据集
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

    # 打印数据集信息
    print("-"*100)
    print(f"数据集名称: {data_name}, id: {openml_id}")

    # 预处理与划分
    X_processed, y_processed, y_scaler = preprocess_data(X, y)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X_processed, y_processed, test_size, val_size, random_state
    )

    # 打印数据形状信息
    print(f"处理前 X.shape: {X.shape}, y.shape: {y.shape}")
    print(
        f"处理后 X_train.shape: {X_train.shape}, X_val.shape: {X_val.shape}, X_test.shape: {X_test.shape}"
    )
    print(
        f"处理后 y_train.shape: {y_train.shape}, y_val.shape: {y_val.shape}, y_test.shape: {y_test.shape}"
    )
    print("-"*100)
    return X_train, X_val, X_test, y_train, y_val, y_test, y_scaler

def preprocess_data(X, y):
    # 处理分类特征, 使用one-hot编码, 
    # 将数据都转换为numpy类型
    X_processed = X.copy()
    # 数值与类别列划分
    categorical_features_all = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=['number']).columns.tolist()

    # 基于基数拆分类别列
    low_cardinality_cats = []
    high_cardinality_cats = []
    for col in categorical_features_all:
        try:
            n_unique = X[col].nunique(dropna=True)
        except Exception:
            n_unique = pd.Series(X[col]).nunique(dropna=True)
        if n_unique > 10:
            high_cardinality_cats.append(col)
        else:
            low_cardinality_cats.append(col)

    transformers = []
    if len(numerical_features) > 0:
        transformers.append(('num', StandardScaler(), numerical_features))
    if len(low_cardinality_cats) > 0:
        transformers.append(('cat_low', OneHotEncoder(handle_unknown='ignore'), low_cardinality_cats))
    if len(high_cardinality_cats) > 0:
        transformers.append(('cat_high', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), high_cardinality_cats))

    preprocessor = ColumnTransformer(transformers=transformers)
    X_processed = preprocessor.fit_transform(X)
    # 将稀疏矩阵转换为稠密矩阵，避免后续 torch.tensor 构造时报错
    if hasattr(X_processed, 'toarray'):
        X_processed = X_processed.toarray()

    # 类别型转换为0-N的数值
    if y.dtype == 'object' or pd.api.types.is_categorical_dtype(y):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        #打印label_encoder的classes和number的映射
        print('-'*90)
        print(f"类别型y的映射关系 {label_encoder.classes_} -> {label_encoder.transform(label_encoder.classes_)}")
        print('-'*90)
    else:
        y = y.to_numpy()
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    y_scaler = StandardScaler()
    y_processed = y_scaler.fit_transform(y)
    return X_processed, y_processed, y_scaler

def split_data(X, y, test_size=0.2, val_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    # 回归数据集
    # reg_id_list = [43440, 43918, 46840, 44956, 46880, 44969, 42712, 42821, 42225, 45048, 41540, 42571, 42705]
    reg_id_list = [43440, 43918, 46840, 44956, 46880, 44969, 42712, 42821, 42225, 45048, 41540, 42571]
    # 分类数据集
    # cls_id_list = [1464, 31, 1487, 3, 1471, 1046, 151, 23512, 143, 1597, 1219, 351, 1169]
    cls_id_list = [1464, 31, 1487, 3, 1471, 1046, 151, 143]
    # total数据集
    id_list = reg_id_list + cls_id_list
    for id in id_list:
        X_train, X_val, X_test, y_train, y_val, y_test, y_scaler = load_data(openml_id=id, test_size=0.15, val_size=0.15, random_state=42)
    # X_train, X_val, X_test, y_train, y_val, y_test, y_scaler = load_data(openml_id=46840, test_size=0.2, val_size=0.2, random_state=42)
    
