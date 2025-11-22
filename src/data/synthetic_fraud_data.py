import numpy as np
import pandas as pd
import random
import uuid
from datetime import datetime, timedelta

def generate_user_profiles(num_users, cities):
    profiles = []

    for user_id in range(num_users):
        profiles.append({
            "user_id": user_id,
            "home_city": random.choice(cities),
            "avg_amount": np.random.uniform(200, 2000),
            "std_amount": np.random.uniform(50, 500),
            "active_hours": random.choice([(6,22), (8,20), (10,23)])
        })

    return pd.DataFrame(profiles)


def generate_normal_transaction(user, num_merchants):
    amount = np.random.normal(user["avg_amount"], user["std_amount"])
    amount = max(1, amount)

    start, end = user["active_hours"]
    hour = random.randint(start, end)

    timestamp = datetime.now() - timedelta(days=random.randint(0, 180))
    timestamp = timestamp.replace(hour=hour, minute=random.randint(0,59))

    return {
        "user_id": user["user_id"],
        "merchant_id": random.randint(0, num_merchants - 1),
        "device_id": random.choice(["iOS", "Android", "Web", "POS", "USSD"]),
        "location": user["home_city"],
        "amount": round(amount, 2),
        "timestamp": timestamp,
        "is_fraud": 0,
        "fraud_reason": None
    }


def inject_fraud(tx, cities):
    fraud_type = random.choice([
        "high_amount_spike",
        "device_change",
        "geo_mismatch",
        "rapid_fire",
        "odd_hour"
    ])

    if fraud_type == "high_amount_spike":
        tx["amount"] *= random.uniform(5, 12)

    elif fraud_type == "device_change":
        tx["device_id"] = random.choice(["UnknownDevice", uuid.uuid4().hex[:8]])

    elif fraud_type == "geo_mismatch":
        tx["location"] = random.choice([c for c in cities if c != tx["location"]])

    elif fraud_type == "rapid_fire":
        tx["timestamp"] = tx["timestamp"].replace(minute=random.randint(0, 2))

    elif fraud_type == "odd_hour":
        hour = random.randint(1, 4)
        tx["timestamp"] = tx["timestamp"].replace(hour=hour)

    tx["is_fraud"] = 1
    tx["fraud_reason"] = fraud_type

    return tx


def generate_fraud_dataset(
    num_users=500,
    num_merchants=25,
    num_transactions=10000,
    fraud_rate=0.05,
    cities=None,
    seed=None
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if cities is None:
        cities = ["Delhi", "Mumbai", "Bangalore", "Pune", "Hyderabad", "Chennai"]

    users = generate_user_profiles(num_users, cities)

    transactions = []

    for _ in range(num_transactions):
        user = users.sample(1).iloc[0]
        tx = generate_normal_transaction(user, num_merchants)

        if random.random() < fraud_rate:
            tx = inject_fraud(tx, cities)

        transactions.append(tx)

    df = pd.DataFrame(transactions)
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df