import os
from typing import Union

import pytest
from pyspark.sql import Column as SparkCol
from pyspark.sql import DataFrame as SparkDF
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

import onekit.sparkkit as sk


@pytest.mark.slow
class TestSparkKit:
    def test_peek(self, spark: SparkSession):
        df = spark.createDataFrame(
            [
                dict(x=1, y="a"),
                dict(x=3, y=None),
                dict(x=None, y="c"),
            ]
        )
        actual = (
            df.transform(sk.peek(n=20, shape=True, cache=True, schema=True, index=True))
            .where(F.col("x").isNotNull())
            .transform(sk.peek())
        )
        expected = df.where(F.col("x").isNotNull())
        self.assert_dataframe_equal(actual, expected)

    @pytest.mark.parametrize("x", ["c1", F.col("c2")])
    def test_str_to_col(self, x: Union[str, SparkCol]):
        actual = sk.str_to_col(x)
        assert isinstance(actual, SparkCol)

    def test_union(self, spark: SparkSession):
        df1 = spark.createDataFrame([dict(x=1, y=2), dict(x=3, y=4)])
        df2 = spark.createDataFrame([dict(x=5, y=6), dict(x=7, y=8)])
        df3 = spark.createDataFrame([dict(x=0, y=1), dict(x=2, y=3)])

        actual = sk.union(df1, df2, df3)
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
