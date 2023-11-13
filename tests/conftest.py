import os

import pytest
from pyspark.sql import SparkSession


@pytest.fixture
def spark() -> SparkSession:
    spark = (
        SparkSession.builder.master("local[1]")
        .appName("spark-session-for-testing")
        .config("spark.sql.shuffle.partitions", 1)
        .config("spark.default.parallelism", os.cpu_count())
        .config("spark.rdd.compress", False)
        .config("spark.shuffle.compress", False)
        .config("spark.dynamicAllocation.enabled", False)
        .config("spark.executor.cores", 1)
        .config("spark.executor.instances", 1)
        .config("spark.ui.enabled", False)
        .config("spark.ui.showConsoleProgress", False)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()
