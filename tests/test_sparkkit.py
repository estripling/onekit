import datetime as dt
import os

import pytest
from pyspark.sql import Column as SparkCol
from pyspark.sql import DataFrame as SparkDF
from pyspark.sql import (
    Row,
    SparkSession,
)
from pyspark.sql import functions as F
from pyspark.sql import types as T

import onekit.sparkkit as sk


@pytest.mark.slow
class TestSparkKit:
    def test_add_prefix(self, spark: SparkSession):
        df = spark.createDataFrame([Row(a=1, b=2)])

        # all columns
        actual = df.transform(sk.add_prefix("pfx_"))
        expected = spark.createDataFrame([Row(pfx_a=1, pfx_b=2)])
        self.assert_dataframe_equal(actual, expected)

        # with column selection
        actual = df.transform(sk.add_prefix("pfx_", subset=["a"]))
        expected = spark.createDataFrame([Row(pfx_a=1, b=2)])
        self.assert_dataframe_equal(actual, expected)

        # used as transformation function
        actual = df.transform(sk.add_prefix("pfx_"))
        expected = spark.createDataFrame([Row(pfx_a=1, pfx_b=2)])
        self.assert_dataframe_equal(actual, expected)

        # used as transformation function with column selection
        add_prefix__pfx = sk.add_prefix("pfx_", subset=["b"])
        actual = df.transform(add_prefix__pfx)
        expected = spark.createDataFrame([Row(a=1, pfx_b=2)])
        self.assert_dataframe_equal(actual, expected)

    def test_add_suffix(self, spark: SparkSession):
        df = spark.createDataFrame([Row(a=1, b=2)])

        # all columns
        actual = df.transform(sk.add_suffix("_sfx"))
        expected = spark.createDataFrame([Row(a_sfx=1, b_sfx=2)])
        self.assert_dataframe_equal(actual, expected)

        # with column selection
        actual = df.transform(sk.add_suffix("_sfx", subset=["a"]))
        expected = spark.createDataFrame([Row(a_sfx=1, b=2)])
        self.assert_dataframe_equal(actual, expected)

        # used as transformation function
        actual = df.transform(sk.add_suffix("_sfx"))
        expected = spark.createDataFrame([Row(a_sfx=1, b_sfx=2)])
        self.assert_dataframe_equal(actual, expected)

        # used as transformation function with column selection
        add_suffix__sfx = sk.add_suffix("_sfx", subset=["b"])
        actual = df.transform(add_suffix__sfx)
        expected = spark.createDataFrame([Row(a=1, b_sfx=2)])
        self.assert_dataframe_equal(actual, expected)

    def test_count_nulls(self, spark: SparkSession):
        df = spark.createDataFrame(
            [
                Row(x=1, y=2, z=None),
                Row(x=4, y=None, z=6),
                Row(x=7, y=8, z=None),
                Row(x=10, y=None, z=None),
            ]
        )

        actual = sk.count_nulls(df)
        expected = spark.createDataFrame([Row(x=0, y=2, z=3)])
        self.assert_dataframe_equal(actual, expected)

        actual = df.transform(sk.count_nulls)
        self.assert_dataframe_equal(actual, expected)

        actual = sk.count_nulls(df, subset=["x", "z"])
        expected = spark.createDataFrame([Row(x=0, z=3)])
        self.assert_dataframe_equal(actual, expected)

        actual = df.transform(sk.count_nulls(subset=["x", "z"]))
        self.assert_dataframe_equal(actual, expected)

    def test_cvf(self, spark: SparkSession):
        # single column
        counts = {"a": 3, "b": 1, "c": 1, "g": 2, "h": 1}
        df = spark.createDataFrame(
            [dict(x=v) for v, c in counts.items() for _ in range(c)]
        )

        expected_rows = [
            Row(x="a", count=3, percent=37.5, cumul_count=3, cumul_percent=37.5),
            Row(x="g", count=2, percent=25.0, cumul_count=5, cumul_percent=62.5),
            Row(x="b", count=1, percent=12.5, cumul_count=6, cumul_percent=75.0),
            Row(x="c", count=1, percent=12.5, cumul_count=7, cumul_percent=87.5),
            Row(x="h", count=1, percent=12.5, cumul_count=8, cumul_percent=100.0),
        ]
        expected = spark.createDataFrame(expected_rows)

        for cols in ["x", ["x"], F.col("x")]:
            actual = df.transform(sk.cvf(cols))
            self.assert_dataframe_equal(actual, expected)

        # multiple columns
        df = spark.createDataFrame(
            [
                Row(x="a", y=1),
                Row(x="c", y=1),
                Row(x="b", y=1),
                Row(x="g", y=2),
                Row(x="h", y=1),
                Row(x="a", y=1),
                Row(x="g", y=2),
                Row(x="a", y=2),
            ]
        )
        actual = df.transform(sk.cvf("x"))  # check single column check first
        self.assert_dataframe_equal(actual, expected)

        actual = df.transform(sk.cvf("x", "y"))

        expected_rows = [
            Row(x="a", y=1, count=2, percent=25.0, cumul_count=2, cumul_percent=25.0),
            Row(x="g", y=2, count=2, percent=25.0, cumul_count=4, cumul_percent=50.0),
            Row(x="a", y=2, count=1, percent=12.5, cumul_count=5, cumul_percent=62.5),
            Row(x="b", y=1, count=1, percent=12.5, cumul_count=6, cumul_percent=75.0),
            Row(x="c", y=1, count=1, percent=12.5, cumul_count=7, cumul_percent=87.5),
            Row(x="h", y=1, count=1, percent=12.5, cumul_count=8, cumul_percent=100.0),
        ]
        expected = spark.createDataFrame(expected_rows)
        self.assert_dataframe_equal(actual, expected)

        actual = df.transform(sk.cvf(["x", "y"]))
        self.assert_dataframe_equal(actual, expected)

        actual = df.transform(sk.cvf("x", F.col("y")))
        self.assert_dataframe_equal(actual, expected)

        actual = df.transform(sk.cvf(F.col("x"), F.col("y")))
        self.assert_dataframe_equal(actual, expected)

        actual = df.transform(sk.cvf([F.col("x"), F.col("y")]))
        self.assert_dataframe_equal(actual, expected)

    def test_daterange(self, spark: SparkSession):
        df = spark.createDataFrame(
            [Row(id=1), Row(id=3), Row(id=2), Row(id=2), Row(id=3)]
        )
        expected = spark.createDataFrame(
            [Row(id=i, day=dt.date(2023, 5, d)) for i in [1, 2, 3] for d in range(1, 8)]
        )

        actual = sk.daterange(df, "2023-05-01", "2023-05-07", "id", "day")
        self.assert_dataframe_equal(actual, expected)

        actual = sk.daterange(df, dt.date(2023, 5, 1), dt.date(2023, 5, 7), "id", "day")
        self.assert_dataframe_equal(actual, expected)

    def test_join(self, spark: SparkSession):
        df1 = spark.createDataFrame([dict(id=1, x="a"), dict(id=2, x="b")])
        df2 = spark.createDataFrame([dict(id=1, y="c"), dict(id=2, y="d")])
        df3 = spark.createDataFrame([dict(id=1, z="e"), dict(id=2, z="f")])

        actual = sk.join(df1, df2, df3, on="id")
        expected = df1.join(df2, "id").join(df3, "id")
        self.assert_dataframe_equal(actual, expected)

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

    def test_str_to_col(self):
        actual = sk.str_to_col("x")
        assert isinstance(actual, SparkCol)

        actual = sk.str_to_col(F.col("x"))
        assert isinstance(actual, SparkCol)

    def test_union(self, spark: SparkSession):
        df1 = spark.createDataFrame([dict(x=1, y=2), dict(x=3, y=4)])
        df2 = spark.createDataFrame([dict(x=5, y=6), dict(x=7, y=8)])
        df3 = spark.createDataFrame([dict(x=0, y=1), dict(x=2, y=3)])

        actual = sk.union(df1, df2, df3)
        expected = df1.unionByName(df2).unionByName(df3)
        self.assert_dataframe_equal(actual, expected)

    def test_with_endofweek_date(self, spark: SparkSession):
        df = spark.createDataFrame(
            [
                Row(day="2023-04-30"),
                Row(day="2023-05-01"),
                Row(day="2023-05-02"),
                Row(day="2023-05-03"),
                Row(day="2023-05-04"),
                Row(day="2023-05-05"),
                Row(day="2023-05-06"),
                Row(day="2023-05-07"),
                Row(day="2023-05-08"),
                Row(day=None),
            ]
        )
        actual = df.transform(sk.with_endofweek_date("day", "endofweek"))
        expected = spark.createDataFrame(
            [
                Row(day="2023-04-30", endofweek="2023-04-30"),
                Row(day="2023-05-01", endofweek="2023-05-07"),
                Row(day="2023-05-02", endofweek="2023-05-07"),
                Row(day="2023-05-03", endofweek="2023-05-07"),
                Row(day="2023-05-04", endofweek="2023-05-07"),
                Row(day="2023-05-05", endofweek="2023-05-07"),
                Row(day="2023-05-06", endofweek="2023-05-07"),
                Row(day="2023-05-07", endofweek="2023-05-07"),
                Row(day="2023-05-08", endofweek="2023-05-14"),
                Row(day=None, endofweek=None),
            ]
        )
        self.assert_dataframe_equal(actual, expected)

        actual = df.transform(sk.with_endofweek_date("day", "endofweek", "Sat"))
        expected = spark.createDataFrame(
            [
                Row(day="2023-04-30", endofweek="2023-05-06"),
                Row(day="2023-05-01", endofweek="2023-05-06"),
                Row(day="2023-05-02", endofweek="2023-05-06"),
                Row(day="2023-05-03", endofweek="2023-05-06"),
                Row(day="2023-05-04", endofweek="2023-05-06"),
                Row(day="2023-05-05", endofweek="2023-05-06"),
                Row(day="2023-05-06", endofweek="2023-05-06"),
                Row(day="2023-05-07", endofweek="2023-05-13"),
                Row(day="2023-05-08", endofweek="2023-05-13"),
                Row(day=None, endofweek=None),
            ]
        )
        self.assert_dataframe_equal(actual, expected)

        df = spark.createDataFrame(
            [
                Row(day=dt.date(2023, 4, 30)),
                Row(day=dt.date(2023, 5, 1)),
                Row(day=dt.date(2023, 5, 2)),
                Row(day=dt.date(2023, 5, 3)),
                Row(day=dt.date(2023, 5, 4)),
                Row(day=dt.date(2023, 5, 5)),
                Row(day=dt.date(2023, 5, 6)),
                Row(day=dt.date(2023, 5, 7)),
                Row(day=dt.date(2023, 5, 8)),
                Row(day=None),
            ]
        )
        actual = df.transform(sk.with_endofweek_date("day", "endofweek"))
        expected = spark.createDataFrame(
            [
                Row(day=dt.date(2023, 4, 30), endofweek=dt.date(2023, 4, 30)),
                Row(day=dt.date(2023, 5, 1), endofweek=dt.date(2023, 5, 7)),
                Row(day=dt.date(2023, 5, 2), endofweek=dt.date(2023, 5, 7)),
                Row(day=dt.date(2023, 5, 3), endofweek=dt.date(2023, 5, 7)),
                Row(day=dt.date(2023, 5, 4), endofweek=dt.date(2023, 5, 7)),
                Row(day=dt.date(2023, 5, 5), endofweek=dt.date(2023, 5, 7)),
                Row(day=dt.date(2023, 5, 6), endofweek=dt.date(2023, 5, 7)),
                Row(day=dt.date(2023, 5, 7), endofweek=dt.date(2023, 5, 7)),
                Row(day=dt.date(2023, 5, 8), endofweek=dt.date(2023, 5, 14)),
                Row(day=None, endofweek=None),
            ]
        )
        self.assert_dataframe_equal(actual, expected)

    def test_with_index(self, spark: SparkSession):
        df = spark.createDataFrame(
            [
                Row(x="a"),
                Row(x="b"),
                Row(x="c"),
                Row(x="d"),
                Row(x="e"),
                Row(x="f"),
                Row(x="g"),
                Row(x="h"),
            ],
            schema=T.StructType([T.StructField("x", T.StringType(), True)]),
        )

        actual = df.transform(sk.with_index("idx"))
        expected = spark.createDataFrame(
            [
                Row(x="a", idx=1),
                Row(x="b", idx=2),
                Row(x="c", idx=3),
                Row(x="d", idx=4),
                Row(x="e", idx=5),
                Row(x="f", idx=6),
                Row(x="g", idx=7),
                Row(x="h", idx=8),
            ],
            schema=T.StructType(
                [
                    T.StructField("x", T.StringType(), True),
                    T.StructField("idx", T.IntegerType(), True),
                ]
            ),
        )
        self.assert_dataframe_equal(actual, expected)

    def test_with_startofweek_date(self, spark: SparkSession):
        df = spark.createDataFrame(
            [
                Row(day="2023-04-30"),
                Row(day="2023-05-01"),
                Row(day="2023-05-02"),
                Row(day="2023-05-03"),
                Row(day="2023-05-04"),
                Row(day="2023-05-05"),
                Row(day="2023-05-06"),
                Row(day="2023-05-07"),
                Row(day="2023-05-08"),
                Row(day=None),
            ]
        )
        actual = df.transform(sk.with_startofweek_date("day", "startofweek"))
        expected = spark.createDataFrame(
            [
                Row(day="2023-04-30", startofweek=dt.date(2023, 4, 24)),
                Row(day="2023-05-01", startofweek=dt.date(2023, 5, 1)),
                Row(day="2023-05-02", startofweek=dt.date(2023, 5, 1)),
                Row(day="2023-05-03", startofweek=dt.date(2023, 5, 1)),
                Row(day="2023-05-04", startofweek=dt.date(2023, 5, 1)),
                Row(day="2023-05-05", startofweek=dt.date(2023, 5, 1)),
                Row(day="2023-05-06", startofweek=dt.date(2023, 5, 1)),
                Row(day="2023-05-07", startofweek=dt.date(2023, 5, 1)),
                Row(day="2023-05-08", startofweek=dt.date(2023, 5, 8)),
                Row(day=None, startofweek=None),
            ]
        )
        self.assert_dataframe_equal(actual, expected)

        actual = df.transform(sk.with_startofweek_date("day", "startofweek", "Sat"))
        expected = spark.createDataFrame(
            [
                Row(day="2023-04-30", startofweek=dt.date(2023, 4, 30)),
                Row(day="2023-05-01", startofweek=dt.date(2023, 4, 30)),
                Row(day="2023-05-02", startofweek=dt.date(2023, 4, 30)),
                Row(day="2023-05-03", startofweek=dt.date(2023, 4, 30)),
                Row(day="2023-05-04", startofweek=dt.date(2023, 4, 30)),
                Row(day="2023-05-05", startofweek=dt.date(2023, 4, 30)),
                Row(day="2023-05-06", startofweek=dt.date(2023, 4, 30)),
                Row(day="2023-05-07", startofweek=dt.date(2023, 5, 7)),
                Row(day="2023-05-08", startofweek=dt.date(2023, 5, 7)),
                Row(day=None, startofweek=None),
            ]
        )
        self.assert_dataframe_equal(actual, expected)

        df = spark.createDataFrame(
            [
                Row(day=dt.date(2023, 4, 30)),
                Row(day=dt.date(2023, 5, 1)),
                Row(day=dt.date(2023, 5, 2)),
                Row(day=dt.date(2023, 5, 3)),
                Row(day=dt.date(2023, 5, 4)),
                Row(day=dt.date(2023, 5, 5)),
                Row(day=dt.date(2023, 5, 6)),
                Row(day=dt.date(2023, 5, 7)),
                Row(day=dt.date(2023, 5, 8)),
                Row(day=None),
            ]
        )
        actual = df.transform(sk.with_startofweek_date("day", "startofweek"))
        expected = spark.createDataFrame(
            [
                Row(day=dt.date(2023, 4, 30), startofweek=dt.date(2023, 4, 24)),
                Row(day=dt.date(2023, 5, 1), startofweek=dt.date(2023, 5, 1)),
                Row(day=dt.date(2023, 5, 2), startofweek=dt.date(2023, 5, 1)),
                Row(day=dt.date(2023, 5, 3), startofweek=dt.date(2023, 5, 1)),
                Row(day=dt.date(2023, 5, 4), startofweek=dt.date(2023, 5, 1)),
                Row(day=dt.date(2023, 5, 5), startofweek=dt.date(2023, 5, 1)),
                Row(day=dt.date(2023, 5, 6), startofweek=dt.date(2023, 5, 1)),
                Row(day=dt.date(2023, 5, 7), startofweek=dt.date(2023, 5, 1)),
                Row(day=dt.date(2023, 5, 8), startofweek=dt.date(2023, 5, 8)),
                Row(day=None, startofweek=None),
            ]
        )
        self.assert_dataframe_equal(actual, expected)

    def test_with_weekday(self, spark: SparkSession):
        df = spark.createDataFrame(
            [
                Row(day="2023-05-01"),
                Row(day="2023-05-02"),
                Row(day="2023-05-03"),
                Row(day="2023-05-04"),
                Row(day="2023-05-05"),
                Row(day="2023-05-06"),
                Row(day="2023-05-07"),
                Row(day=None),
            ]
        )
        actual = df.transform(sk.with_weekday("day", "weekday"))
        expected = spark.createDataFrame(
            [
                Row(day="2023-05-01", weekday="Mon"),
                Row(day="2023-05-02", weekday="Tue"),
                Row(day="2023-05-03", weekday="Wed"),
                Row(day="2023-05-04", weekday="Thu"),
                Row(day="2023-05-05", weekday="Fri"),
                Row(day="2023-05-06", weekday="Sat"),
                Row(day="2023-05-07", weekday="Sun"),
                Row(day=None, weekday=None),
            ]
        )
        self.assert_dataframe_equal(actual, expected)

        actual = df.withColumn("day", F.to_date("day", "yyyy-MM-dd")).transform(
            sk.with_weekday("day", "weekday")
        )
        self.assert_dataframe_equal(
            actual,
            expected.withColumn("day", F.to_date("day", "yyyy-MM-dd")),
        )

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
