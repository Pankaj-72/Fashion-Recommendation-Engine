"""Optional Spark aggregations; Pandas remains the default local pipeline."""

from __future__ import annotations


def aggregate_transactions_spark(transactions_path: str):
    """Return Spark purchase counts without collecting the full table locally."""
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F
    except ImportError as exc:
        raise RuntimeError("Install pyspark to use the optional distributed pipeline") from exc
    spark = SparkSession.builder.appName("fashion-recommender-features").getOrCreate()
    transactions = spark.read.option("header", True).csv(transactions_path)
    return transactions.groupBy("customer_id").agg(F.count("article_id").alias("purchase_count"))
