import pandas as pd
import numpy as np


def load_data(path: str = "data/online_retail.xlsx") -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Drop nulls in critical columns
    df = df.dropna(subset=["CustomerID", "Description"])

    # Remove cancellations (InvoiceNo starting with 'C')
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    # Remove negative/zero quantities and prices
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    # Parse dates
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Revenue column
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    # CustomerID as int
    df["CustomerID"] = df["CustomerID"].astype(int)

    return df


def get_snapshot_date(df: pd.DataFrame) -> pd.Timestamp:
    """Use 1 day after last invoice as analysis date."""
    return df["InvoiceDate"].max() + pd.Timedelta(days=1)


if __name__ == "__main__":
    df_raw = load_data()
    print(f"Raw shape: {df_raw.shape}")
    df_clean = clean_data(df_raw)
    print(f"Clean shape: {df_clean.shape}")
    print(df_clean.head(3))