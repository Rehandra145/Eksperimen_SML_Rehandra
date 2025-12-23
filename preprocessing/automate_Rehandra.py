"""
Automated Data Preprocessing Module
Eksperimen SML - Rehandra

Module ini berisi fungsi-fungsi untuk melakukan preprocessing data
secara otomatis dan mengembalikan data yang siap dilatih.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os


def load_data(file_path: str) -> pd.DataFrame:
    """
    Memuat dataset dari file CSV.
    
    Args:
        file_path: Path ke file CSV
        
    Returns:
        DataFrame yang berisi dataset
    """
    df = pd.read_csv(file_path)
    print(f"[LOAD] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menangani missing values dalam dataset.
    - Kolom numerik: diisi dengan median
    - Kolom kategorikal: diisi dengan modus
    
    Args:
        df: DataFrame input
        
    Returns:
        DataFrame tanpa missing values
    """
    df = df.copy()
    
    # Kolom numerik -> isi dengan median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"[MISSING] {col}: filled with median ({median_val})")
    
    # Kolom kategorikal -> isi dengan modus
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != 'Heart Disease' and df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"[MISSING] {col}: filled with mode ({mode_val})")
    
    total_missing = df.isnull().sum().sum()
    print(f"[MISSING] Total missing values after handling: {total_missing}")
    
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghapus baris duplikat dari dataset.
    
    Args:
        df: DataFrame input
        
    Returns:
        DataFrame tanpa duplikat
    """
    df = df.copy()
    duplicates_count = df.duplicated().sum()
    
    if duplicates_count > 0:
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        print(f"[DUPLICATE] Removed {duplicates_count} duplicate rows")
    else:
        print("[DUPLICATE] No duplicates found")
    
    return df


def handle_outliers(df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
    """
    Mendeteksi dan menangani outlier menggunakan IQR capping.
    
    Args:
        df: DataFrame input
        method: Metode deteksi outlier ('iqr')
        
    Returns:
        DataFrame dengan outlier yang sudah ditangani
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        
        if outliers > 0:
            # Capping (winsorization)
            df[col] = np.clip(df[col], lower_bound, upper_bound)
            print(f"[OUTLIER] {col}: {outliers} outliers capped")
    
    return df


def encode_target(df: pd.DataFrame, target_col: str = 'Heart Disease') -> tuple:
    """
    Melakukan encoding pada kolom target.
    
    Args:
        df: DataFrame input
        target_col: Nama kolom target
        
    Returns:
        Tuple (DataFrame, LabelEncoder)
    """
    df = df.copy()
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col])
    
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"[ENCODE] {target_col} encoded: {mapping}")
    
    return df, le


def normalize_features(X: pd.DataFrame) -> tuple:
    """
    Melakukan normalisasi fitur menggunakan StandardScaler.
    
    Args:
        X: DataFrame fitur
        
    Returns:
        Tuple (DataFrame yang dinormalisasi, StandardScaler)
    """
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X), 
        columns=X.columns
    )
    print(f"[NORMALIZE] Features normalized using StandardScaler")
    
    return X_scaled, scaler


def split_data(X: pd.DataFrame, y: pd.Series, 
               test_size: float = 0.2, 
               random_state: int = 42) -> tuple:
    """
    Membagi data menjadi training dan test set.
    
    Args:
        X: DataFrame fitur
        y: Series target
        test_size: Proporsi data test
        random_state: Random seed
        
    Returns:
        Tuple (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    print(f"[SPLIT] Train: {X_train.shape[0]} samples ({100-test_size*100:.0f}%)")
    print(f"[SPLIT] Test: {X_test.shape[0]} samples ({test_size*100:.0f}%)")
    
    return X_train, X_test, y_train, y_test


def preprocess_data(file_path: str, 
                    target_col: str = 'Heart Disease',
                    test_size: float = 0.2,
                    random_state: int = 42,
                    save_output: bool = True,
                    output_dir: str = None) -> tuple:
    """
    Fungsi utama untuk melakukan preprocessing data secara otomatis.
    
    Pipeline preprocessing:
    1. Load data
    2. Handle missing values
    3. Remove duplicates
    4. Handle outliers (IQR capping)
    5. Encode target variable
    6. Normalize features (StandardScaler)
    7. Split data (train/test)
    
    Args:
        file_path: Path ke file dataset CSV
        target_col: Nama kolom target
        test_size: Proporsi data test (default 0.2)
        random_state: Random seed (default 42)
        save_output: Apakah menyimpan output ke CSV
        output_dir: Direktori untuk menyimpan output
        
    Returns:
        Tuple (X_train, X_test, y_train, y_test)
    """
    print("=" * 60)
    print("AUTOMATED DATA PREPROCESSING")
    print("=" * 60)
    
    # 1. Load data
    df = load_data(file_path)
    
    # 2. Handle missing values
    df = handle_missing_values(df)
    
    # 3. Remove duplicates
    df = remove_duplicates(df)
    
    # 4. Handle outliers
    df = handle_outliers(df)
    
    # 5. Encode target
    df, label_encoder = encode_target(df, target_col)
    
    # 6. Split features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # 7. Normalize features
    X_scaled, scaler = normalize_features(X)
    
    # 8. Split data
    X_train, X_test, y_train, y_test = split_data(
        X_scaled, y, test_size, random_state
    )
    
    # 9. Save output if requested
    if save_output:
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 
                'dataset_preprocessing'
            )
        os.makedirs(output_dir, exist_ok=True)
        
        # Save train data
        train_data = X_train.copy()
        train_data[target_col] = y_train.values
        train_data.to_csv(os.path.join(output_dir, 'train_data.csv'), index=False)
        
        # Save test data
        test_data = X_test.copy()
        test_data[target_col] = y_test.values
        test_data.to_csv(os.path.join(output_dir, 'test_data.csv'), index=False)
        
        print(f"\n[SAVE] Output saved to: {output_dir}")
        print(f"  - train_data.csv ({X_train.shape[0]} rows)")
        print(f"  - test_data.csv ({X_test.shape[0]} rows)")
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED!")
    print("=" * 60)
    
    return X_train, X_test, y_train, y_test


# Entry point untuk eksekusi langsung
if __name__ == "__main__":
    # Path ke dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, '..', 'dataset_raw', 'Heart_Disease_Prediction.csv')
    
    # Jalankan preprocessing
    X_train, X_test, y_train, y_test = preprocess_data(
        file_path=dataset_path,
        target_col='Heart Disease',
        test_size=0.2,
        random_state=42,
        save_output=True
    )
    
    # Tampilkan ringkasan
    print(f"\nData siap dilatih:")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  y_test shape: {y_test.shape}")
