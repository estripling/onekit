import os

import pytest
from pyspark.sql import DataFrame as SparkDF
from pyspark.sql import SparkSession

from onekit import sparktlz


@pytest.mark.slow
class TestSparkToolz:
    def test_union(self, spark: SparkSession):
        df1 = spark.createDataFrame([dict(x=1, y=2), dict(x=3, y=4)])
        df2 = spark.createDataFrame([dict(x=5, y=6), dict(x=7, y=8)])
        df3 = spark.createDataFrame([dict(x=0, y=1), dict(x=2, y=3)])

        actual = sparktlz.union(df1, df2, df3)
        expected = df1.unionByName(df2).unionByName(df3)
        self.assert_dataframe_equal(actual, expected)

    def test_spark_session(self, spark: SparkSession):
        assert isinstance(spark, SparkSession)
        assert spark.sparkContext.appName == "spark-session-for-testing"
        assert spark.version == "3.1.1"

    @staticmethod
    def assert_dataframe_equal(lft_df: SparkDF, rgt_df: SparkDF) -> None:
        """Assert that the left and right data frames are equal."""
        lft_schema = lft_df.schema.simpleString()
        rgt_schema = rgt_df.schema.simpleString()
        assert lft_schema == rgt_schema, "schema mismatch"

        assert lft_df.count() == rgt_df.count(), "row count mismatch"

        lft_rows = lft_df.subtract(rgt_df)
        rgt_rows = rgt_df.subtract(lft_df)
        assert (lft_rows.count() == 0) and (rgt_rows.count() == 0), "row mismatch"

    @pytest.fixture(scope="class")
    def spark(self) -> SparkSession:
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
