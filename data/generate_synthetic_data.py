import numpy as np
import pandas as pd
from pathlib import Path


def generate_synthetic_customers(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)

    genders = np.random.choice(["Male", "Female"], size=n_samples, p=[0.5, 0.5])
    geographies = np.random.choice(["India", "Germany", "France", "Spain"], size=n_samples,
                                   p=[0.55, 0.15, 0.15, 0.15])
    ages = np.random.randint(18, 75, size=n_samples)
    tenures = np.random.randint(0, 11, size=n_samples)  # years with company
    balances = np.random.normal(loc=70000, scale=30000, size=n_samples)
    balances = np.clip(balances, 0, None)
    num_products = np.random.randint(1, 5, size=n_samples)
    has_credit_card = np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7])
    is_active_member = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])
    estimated_salary = np.random.normal(loc=80000, scale=40000, size=n_samples)
    estimated_salary = np.clip(estimated_salary, 10000, None)

    # Simple churn probability logic
    base_prob = 0.2
    prob = (
        base_prob
        + (ages - 40) * 0.004
        + (balances - 70000) / 250000
        + (1 - is_active_member) * 0.20
        + (num_products == 1) * 0.08
        + (tenures < 2) * 0.10
    )

    prob = 1 / (1 + np.exp(-prob))
    prob = np.clip(prob, 0.05, 0.95)

    churn = np.random.binomial(1, prob)

    df = pd.DataFrame({
        "CustomerID": np.arange(1, n_samples + 1),
        "Gender": genders,
        "Geography": geographies,
        "Age": ages,
        "Tenure": tenures,
        "Balance": balances.astype(int),
        "NumProducts": num_products,
        "HasCreditCard": has_credit_card,
        "IsActiveMember": is_active_member,
        "EstimatedSalary": estimated_salary.astype(int),
        "Churn": churn
    })

    return df


if __name__ == "__main__":
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    df = generate_synthetic_customers()
    out_path = data_dir / "raw_customers.csv"
    df.to_csv(out_path, index=False)
    print(f"Synthetic dataset saved to {out_path.resolve()}")
