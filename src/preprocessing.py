import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess():
    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",
        sep=" ",
        header=None
    )

    df.columns = [
        "Status_of_existing_checking_account",
        "Duration_in_month",
        "Credit_history",
        "Purpose",
        "Credit_amount",
        "Savings_account/bonds",
        "Present_employment_since",
        "Installment_rate_in_percentage_of_disposable_income",
        "Personal_status_and_sex",
        "Other_debtors/guarantors",
        "Present_residence_since",
        "Property",
        "Age_in_years",
        "Other_installment_plans",
        "Housing",
        "Number_of_existing_credits_at_this_bank",
        "Job",
        "Number_of_people_being_liable_to_provide_maintenance_for",
        "Telephone",
        "Foreign_worker",
        "Target"
    ]

    df['Risk'] = df['Target'].map({1: "Good", 2: "Bad"})
    df.drop(columns=['Target'], inplace=True)

    # Encode
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])

    X = df.drop('Risk', axis=1)
    y = df['Risk']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_test, y_train, y_test
