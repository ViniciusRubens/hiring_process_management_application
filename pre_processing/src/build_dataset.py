import sklearn
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Config ---
RAW_DATA_PATH = '../../data_analysis/data/dataset.csv'
X_TRAIN_PATH = '../artifacts/X_train.parquet'
X_TEST_PATH = '../artifacts/X_test.parquet'
Y_TRAIN_PATH = '../artifacts/y_train.parquet'
Y_TEST_PATH = '../artifacts/y_test.parquet'
SCALER_PATH = '../artifacts/scaler.pkl'

TEST_SIZE = 0.2
RANDOM_STATE = 42
# --------------------

def load_data(path):

    print(f"Loading processed data from {path}...")
    df = pd.read_csv(path)

    print("\nDataset info: \n")
    df.info()

    return df

def split_data(df, target_column, test_size, random_state = RANDOM_STATE):

    X = df.drop(target_column, axis = 1)

    y = df[target_column]

    print(f"Splitting data... Test size: {TEST_SIZE}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = test_size, random_state = random_state
    )

    print(f"X Train set shape: {X_train.shape}")
    print(f"X Test set shape: {X_test.shape}")
    print(f"Y Train set shape: {y_train.shape}")
    print(f"Y Test set shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test

def standard_scaler(train_set, test_set):

    numeric_columns = train_set.select_dtypes(include = ['int64', 'float64']).columns
    print(f"Numeric columns: {numeric_columns}")

    scaler = StandardScaler()

    # Applying StandardScaler to numerical variables
    train_set[numeric_columns] = scaler.fit_transform(train_set[numeric_columns])
    test_set[numeric_columns] = scaler.transform(test_set[numeric_columns])

    return scaler, train_set, test_set

def pre_processing(X_train, X_test):

    print("\nPre-processing data...")

    print("\nStarting scaling...")
    scaler, X_train_preprocessed, X_test_preprocessed = standard_scaler(X_train, X_test)

    print("\nX_train_preprocessed head: \n")
    print(X_train_preprocessed.head())
    print("\nX_test_preprocessed head: \n")
    print(X_test_preprocessed.head())

    return scaler, X_train_preprocessed, X_test_preprocessed

def save_artifacts(
    X_train, X_test, y_train, y_test, 
    scaler_path, scaler, 
    x_train_path, x_test_path, y_train_path, y_test_path,
    target_column
):
    # Turning back the np.array to df
    X_train = pd.DataFrame(X_train, columns = X_train.columns)
    X_test = pd.DataFrame(X_test, columns = X_test.columns)

    print(f"Saving scaler to {scaler_path}...")
    pickle.dump(scaler, open(scaler_path,'wb'))

    print(f"Saving splits to 'artifacts' folder...")
    X_train.to_parquet(x_train_path, index = False)
    X_test.to_parquet(x_test_path, index = False)

    # Save target as DataFrame to keep column name
    y_train.to_frame(name = target_column).to_parquet(y_train_path, index=False)
    y_test.to_frame(name = target_column).to_parquet(y_test_path, index=False)

    return

if __name__ == "__main__":
    hiring_process_df = load_data(RAW_DATA_PATH)

    target_column = 'status'
    X_train, X_test, y_train, y_test = split_data(hiring_process_df, target_column, TEST_SIZE)
    
    # Preprocessing, especially the application of encoding techniques and data normalization, 
    # should ideally be done after dividing the dataset into training and test sets. This prevents 
    # information leakage from the test set to the training set, which can happen if preprocessing 
    # is done before the division.
    
    scaler, X_train_preprocessed, X_test_preprocessed = pre_processing(X_train, X_test)

    save_artifacts(
        X_train_preprocessed, X_test_preprocessed, y_train, y_test,
        SCALER_PATH, scaler, 
        X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH,
        target_column
    )

    print("\nBuild dataset to modelling complete.")
