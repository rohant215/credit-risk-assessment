import pandas as pd
import numpy as np

def load_data():
    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",
        sep=" ",
        header=None
    )

    df.columns = [
        "Status",
        "Duration",
        "CreditHistory",
        "Purpose",
        "Amount",
        "Savings",
        "Employment",
        "InstallRate",
        "PersonalStatus",
        "Debtors",
        "Residence",
        "Property",
        "Age",
        "InstallPlans",
        "Housing",
        "ExistingCredits",
        "Job",
        "Liable",
        "Telephone",
        "Foreign",
        "Target"
    ]

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category').cat.codes

    df['Target'] = df['Target'].map({1:1, 2:0})

    X = df.drop("Target", axis=1).values
    y = df["Target"].values

    np.random.seed(42)
    indices = np.random.permutation(len(X))
    train_size = int(0.8 * len(X))
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    return X_train, y_train, X_test, y_test, df

