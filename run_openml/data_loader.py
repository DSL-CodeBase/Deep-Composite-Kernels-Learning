"""
Data loader for OpenML datasets

Input:
    openml_id: OpenML dataset ID
    test_size: Test set ratio
    val_size: Validation set ratio
    random_state: Random seed

Output:
    X_train, X_val, X_test: Training, validation, and test features
    y_train, y_val, y_test: Training, validation, and test labels
    y_scaler: StandardScaler for target values

Processing steps:
    1. Load dataset from OpenML by ID
    2. Automatically process categorical and numerical features
    3. Split into train/val/test sets
    4. Return processed data and scaler
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
        dpath = "/home/zhangjunyu/zjywork/zjycode/DKRR_version/datasets/42571/dataset.arff"
        raw_data, meta = arff.loadarff(dpath)
        df = pd.DataFrame(raw_data)
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].apply(lambda v: v.decode('utf-8') if isinstance(v, bytes) else v)

        candidate_targets = ['class', 'target', 'y', 'label']
        target_col = next((c for c in candidate_targets if c in df.columns), df.columns[-1])

        X = df.drop(columns=[target_col])
        y = df[target_col]
        data_name = f"local_arff_{openml_id}"
    else:
        dataset = openml.datasets.get_dataset(openml_id)
        data_name = dataset.name
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

    print("-"*100)
    print(f"Dataset name: {data_name}, id: {openml_id}")

    X_processed, y_processed, y_scaler = preprocess_data(X, y)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X_processed, y_processed, test_size, val_size, random_state
    )

    print(f"Original X.shape: {X.shape}, y.shape: {y.shape}")
    print(f"Processed X_train.shape: {X_train.shape}, X_val.shape: {X_val.shape}, X_test.shape: {X_test.shape}")
    print(f"Processed y_train.shape: {y_train.shape}, y_val.shape: {y_val.shape}, y_test.shape: {y_test.shape}")
    print("-"*100)
    return X_train, X_val, X_test, y_train, y_val, y_test, y_scaler

def preprocess_data(X, y):
    X_processed = X.copy()
    categorical_features_all = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=['number']).columns.tolist()

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
    if hasattr(X_processed, 'toarray'):
        X_processed = X_processed.toarray()

    if y.dtype == 'object' or pd.api.types.is_categorical_dtype(y):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        print('-'*90)
        print(f"Label encoding: {label_encoder.classes_} -> {label_encoder.transform(label_encoder.classes_)}")
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
    reg_id_list = [43440, 43918, 46840, 44956, 46880, 44969, 42712, 42821, 42225, 45048, 41540, 42571]
    cls_id_list = [1464, 31, 1487, 3, 1471, 1046, 151, 143]
    id_list = reg_id_list + cls_id_list
    for id in id_list:
        X_train, X_val, X_test, y_train, y_val, y_test, y_scaler = load_data(openml_id=id, test_size=0.15, val_size=0.15, random_state=42)
    
