import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def create_spark_session():
    spark = SparkSession.builder \
        .appName("CLV_Feature_Engineering") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def compute_rfm_spark(df_pandas: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    """
    Use PySpark for RFM + behavioral feature computation.
    Returns a pandas DataFrame (collected after Spark computation).
    """
    spark = create_spark_session()

    # Convert to Spark DF
    df_spark = spark.createDataFrame(df_pandas)

    snapshot_ts = snapshot_date.timestamp()

    # Cast InvoiceDate to timestamp
    df_spark = df_spark.withColumn(
        "InvoiceTS", F.unix_timestamp("InvoiceDate")
    )

    # RFM aggregation
    rfm = df_spark.groupBy("CustomerID").agg(
        # Recency: days since last purchase
        F.round((snapshot_ts - F.max("InvoiceTS")) / 86400, 2).alias("Recency"),
        # Frequency: number of unique invoices
        F.countDistinct("InvoiceNo").alias("Frequency"),
        # Monetary: total revenue
        F.round(F.sum("Revenue"), 2).alias("Monetary"),
        # Advanced behavioral features
        F.round(F.avg("Revenue"), 2).alias("AvgOrderValue"),
        F.round(F.stddev("Revenue"), 2).alias("StdOrderValue"),
        F.count("InvoiceNo").alias("TotalTransactions"),
        F.countDistinct("StockCode").alias("UniqueProducts"),
        F.round(F.sum("Quantity"), 0).alias("TotalQuantity"),
        F.countDistinct(F.date_format("InvoiceDate", "yyyy-MM")).alias("ActiveMonths"),
    )

    # Customer lifespan in days
    lifespan = df_spark.groupBy("CustomerID").agg(
        F.round(
            (F.max("InvoiceTS") - F.min("InvoiceTS")) / 86400, 2
        ).alias("LifespanDays")
    )

    rfm = rfm.join(lifespan, on="CustomerID", how="left")

    # Purchase frequency per month
    rfm = rfm.withColumn(
        "FreqPerMonth",
        F.when(F.col("ActiveMonths") > 0,
               F.round(F.col("Frequency") / F.col("ActiveMonths"), 2)
               ).otherwise(F.col("Frequency"))
    )

    # Average inter-purchase time (days between orders)
    rfm = rfm.withColumn(
        "AvgInterPurchaseDays",
        F.when(F.col("Frequency") > 1,
               F.round(F.col("LifespanDays") / (F.col("Frequency") - 1), 2)
               ).otherwise(F.lit(None))
    )

    result = rfm.toPandas()
    result["StdOrderValue"] = result["StdOrderValue"].fillna(0)
    result["AvgInterPurchaseDays"] = result["AvgInterPurchaseDays"].fillna(
        result["AvgInterPurchaseDays"].median()
    )

    spark.stop()
    return result


def add_rfm_scores(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Quintile-based RFM scoring (1–5)."""
    df = rfm_df.copy()

    # Recency: lower is better → reverse rank
    df["R_Score"] = pd.qcut(df["Recency"], q=5, labels=[5, 4, 3, 2, 1])
    df["F_Score"] = pd.qcut(df["Frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5])
    df["M_Score"] = pd.qcut(df["Monetary"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5])

    df["R_Score"] = df["R_Score"].astype(int)
    df["F_Score"] = df["F_Score"].astype(int)
    df["M_Score"] = df["M_Score"].astype(int)
    df["RFM_Score"] = df["R_Score"] + df["F_Score"] + df["M_Score"]

    return df


if __name__ == "__main__":
    from src.data_processing import load_data, clean_data, get_snapshot_date

    df = clean_data(load_data())
    snapshot = get_snapshot_date(df)
    rfm = compute_rfm_spark(df, snapshot)
    rfm = add_rfm_scores(rfm)
    print(f"RFM shape: {rfm.shape}")
    print(rfm.describe())
    rfm.to_csv("data/rfm_features.csv", index=False)
    print("Saved to data/rfm_features.csv")