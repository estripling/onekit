import datetime as dt
import math
from typing import (
    Callable,
    List,
)

import pytest
import toolz
from pyspark.sql import Column as SparkCol
from pyspark.sql import DataFrame as SparkDF
from pyspark.sql import (
    Row,
    SparkSession,
)
from pyspark.sql import functions as F
from pyspark.sql import types as T

import onekit.pythonkit as pk
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

    def test_all_col(self, spark: SparkSession):
        df = spark.createDataFrame(
            [
                Row(i=1, a=False, b=False, expect=False),
                Row(i=2, a=False, b=True, expect=False),
                Row(i=3, a=True, b=False, expect=False),
                Row(i=4, a=True, b=True, expect=True),
                Row(i=5, a=False, b=None, expect=False),
                Row(i=6, a=True, b=None, expect=False),
                Row(i=7, a=None, b=False, expect=False),
                Row(i=8, a=None, b=True, expect=False),
                Row(i=9, a=None, b=None, expect=False),
            ]
        )
        actual = df.withColumn("fx", sk.all_col("a", "b")).select("i", "fx")
        expected = df.select("i", F.col("expect").alias("fx"))
        self.assert_dataframe_equal(actual, expected)

    def test_any_col(self, spark: SparkSession):
        df = spark.createDataFrame(
            [
                Row(i=1, a=False, b=False, expect=False),
                Row(i=2, a=False, b=True, expect=True),
                Row(i=3, a=True, b=False, expect=True),
                Row(i=4, a=True, b=True, expect=True),
                Row(i=5, a=False, b=None, expect=False),
                Row(i=6, a=True, b=None, expect=True),
                Row(i=7, a=None, b=False, expect=False),
                Row(i=8, a=None, b=True, expect=True),
                Row(i=9, a=None, b=None, expect=False),
            ]
        )
        actual = df.withColumn("fx", sk.any_col("a", "b")).select("i", "fx")
        expected = df.select("i", F.col("expect").alias("fx"))
        self.assert_dataframe_equal(actual, expected)

    def test_assert_dataframe_equal(self, spark: SparkSession):
        lft_df = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4)])
        rgt_df = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4)])

        assert sk.assert_dataframe_equal(lft_df, rgt_df) is None

    def test_assert_row_count_equal(self, spark: SparkSession):
        lft_df = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4)])

        rgt_df__equal = spark.createDataFrame([Row(x=1), Row(x=3)])
        rgt_df__different = spark.createDataFrame([Row(x=1)])

        assert sk.assert_row_count_equal(lft_df, rgt_df__equal) is None

        with pytest.raises(sk.RowCountMismatchError):
            sk.assert_row_count_equal(lft_df, rgt_df__different)

    def test_assert_row_equal(self, spark: SparkSession):
        lft_df = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4)])

        rgt_df__equal = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4)])
        rgt_df__different = spark.createDataFrame([Row(x=1, y=7), Row(x=3, y=9)])

        assert sk.assert_row_equal(lft_df, rgt_df__equal) is None

        with pytest.raises(sk.RowMismatchError):
            sk.assert_row_equal(lft_df, rgt_df__different)

    def test_assert_schema_equal(self, spark: SparkSession):
        lft_df = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4)])

        rgt_df__equal = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4)])
        rgt_df__different_type = spark.createDataFrame(
            [Row(x=1, y="2"), Row(x=3, y="4")]
        )
        rgt_df__different_size = spark.createDataFrame([Row(x=1), Row(x=3)])

        assert sk.assert_schema_equal(lft_df, rgt_df__equal) is None

        with pytest.raises(sk.SchemaMismatchError):
            sk.assert_schema_equal(lft_df, rgt_df__different_type)

        with pytest.raises(sk.SchemaMismatchError):
            sk.assert_schema_equal(lft_df, rgt_df__different_size)

    def test_bool_to_int(self, spark: SparkSession):
        df = spark.createDataFrame(
            [
                Row(i=1, x=True, expect=1),
                Row(i=2, x=False, expect=0),
            ],
            schema=T.StructType(
                [
                    T.StructField("i", T.IntegerType(), True),
                    T.StructField("x", T.BooleanType(), True),
                    T.StructField("expect", T.IntegerType(), True),
                ]
            ),
        )

        actual = df.transform(sk.bool_to_int()).select("i", F.col("x").alias("fx"))
        expected = df.select("i", F.col("expect").alias("fx"))
        self.assert_dataframe_equal(actual, expected)

    def test_bool_to_str(self, spark: SparkSession):
        df = spark.createDataFrame(
            [
                Row(i=1, x=True, expect="true"),
                Row(i=2, x=False, expect="false"),
            ],
            schema=T.StructType(
                [
                    T.StructField("i", T.IntegerType(), True),
                    T.StructField("x", T.BooleanType(), True),
                    T.StructField("expect", T.StringType(), True),
                ]
            ),
        )

        actual = df.transform(sk.bool_to_str()).select("i", F.col("x").alias("fx"))
        expected = df.select("i", F.col("expect").alias("fx"))
        self.assert_dataframe_equal(actual, expected)

        assert sk.select_col_types(actual, T.StringType) == ["fx"]

    def test_check_column_present(self, spark: SparkSession):
        df = spark.createDataFrame([Row(x=1, y=2)])
        actual = df.transform(sk.check_column_present("x"))
        assert actual is df

        actual = df.transform(sk.check_column_present("x", "y"))
        assert actual is df

        with pytest.raises(sk.ColumnNotFoundError):
            df.transform(sk.check_column_present("z"))

        with pytest.raises(sk.ColumnNotFoundError):
            df.transform(sk.check_column_present("x", "y", "z"))

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

    def test_date_range(self, spark: SparkSession):
        df = spark.createDataFrame(
            [Row(id=1), Row(id=3), Row(id=2), Row(id=2), Row(id=3)]
        )
        expected = spark.createDataFrame(
            [Row(id=i, d=dt.date(2023, 5, d)) for i in [1, 2, 3] for d in range(1, 8)]
        )

        actual = sk.date_range(df, "2023-05-01", "2023-05-07", "id", "d")
        self.assert_dataframe_equal(actual, expected)

        actual = sk.date_range(df, dt.date(2023, 5, 1), dt.date(2023, 5, 7), "id", "d")
        self.assert_dataframe_equal(actual, expected)

    @pytest.mark.parametrize("func", [toolz.identity, pk.str_to_date])
    def test_filter_date(self, spark: SparkSession, func: Callable):
        d0 = func("2024-01-01")
        df = spark.createDataFrame(
            [
                Row(d=func("2023-11-30"), n1=False, n2=False, n30=False, n_inf=True),
                Row(d=func("2023-12-02"), n1=False, n2=False, n30=False, n_inf=True),
                Row(d=func("2023-12-03"), n1=False, n2=False, n30=True, n_inf=True),
                Row(d=func("2023-12-15"), n1=False, n2=False, n30=True, n_inf=True),
                Row(d=func("2023-12-20"), n1=False, n2=False, n30=True, n_inf=True),
                Row(d=func("2023-12-29"), n1=False, n2=False, n30=True, n_inf=True),
                Row(d=func("2023-12-30"), n1=False, n2=False, n30=True, n_inf=True),
                Row(d=func("2023-12-31"), n1=False, n2=True, n30=True, n_inf=True),
                Row(d=d0, n1=True, n2=True, n30=True, n_inf=True),
                Row(d=func("2024-01-02"), n1=False, n2=False, n30=False, n_inf=False),
                Row(d=func("2024-01-08"), n1=False, n2=False, n30=False, n_inf=False),
                Row(d=func("2024-01-10"), n1=False, n2=False, n30=False, n_inf=False),
            ],
        )

        for n, col in [
            (1, "n1"),
            (2, "n2"),
            (30, "n30"),
            (float("inf"), "n_inf"),
            (math.inf, "n_inf"),
        ]:
            actual = df.transform(sk.filter_date("d", d0=d0, n=n)).select("d")
            expected = df.where(col).select("d")
            self.assert_dataframe_equal(actual, expected)

        for n in [None, "0", "a_string"]:
            with pytest.raises(TypeError):
                df.transform(sk.filter_date("d", d0=d0, n=n))

        for n in [0, -1, 1.0, 1.5]:
            with pytest.raises(ValueError):
                df.transform(sk.filter_date("d", d0=d0, n=n))

    def test_has_column(self, spark: SparkSession):
        df = spark.createDataFrame([Row(x=1, y=2)])
        assert sk.has_column(df, cols=["x"])
        assert sk.has_column(df, cols=["x", "y"])
        assert not sk.has_column(df, cols=["x", "y", "z"])
        assert not sk.has_column(df, cols=["z"])
        assert not sk.has_column(df, cols=["x", "z"])

    def test_is_dataframe_equal(self, spark: SparkSession):
        lft_df = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4)])

        rgt_df__equal = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4)])
        rgt_df__different = spark.createDataFrame([Row(x=1)])

        assert sk.is_dataframe_equal(lft_df, rgt_df__equal)
        assert not sk.is_dataframe_equal(lft_df, rgt_df__different)

    def test_is_row_count_equal(self, spark: SparkSession):
        lft_df = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4)])

        rgt_df__equal = spark.createDataFrame([Row(x=1), Row(x=3)])
        rgt_df__different = spark.createDataFrame([Row(x=1)])

        assert sk.is_row_count_equal(lft_df, rgt_df__equal)
        assert not sk.is_row_count_equal(lft_df, rgt_df__different)

    def test_is_row_equal(self, spark: SparkSession):
        lft_df = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4)])

        rgt_df__equal = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4)])
        rgt_df__different = spark.createDataFrame([Row(x=1, y=7), Row(x=3, y=9)])

        assert sk.is_row_equal(lft_df, rgt_df__equal)
        assert not sk.is_row_equal(lft_df, rgt_df__different)

    def test_is_schema_equal(self, spark: SparkSession):
        lft_df = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4)])

        rgt_df__equal = spark.createDataFrame([Row(x=1, y=2), Row(x=3, y=4)])
        rgt_df__different_type = spark.createDataFrame(
            [Row(x=1, y="2"), Row(x=3, y="4")]
        )
        rgt_df__different_size = spark.createDataFrame([Row(x=1), Row(x=3)])

        assert sk.is_schema_equal(lft_df, rgt_df__equal)
        assert not sk.is_schema_equal(lft_df, rgt_df__different_type)
        assert not sk.is_schema_equal(lft_df, rgt_df__different_size)

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
                dict(x=1, y="a", z=True),
                dict(x=3, y=None, z=False),
                dict(x=None, y="c", z=True),
            ]
        )
        actual = (
            df.transform(sk.peek(n=20, shape=True, cache=True, schema=True, index=True))
            .where(F.col("x").isNotNull())
            .transform(sk.peek())
        )
        expected = df.where(F.col("x").isNotNull())
        self.assert_dataframe_equal(actual, expected)

    @pytest.mark.parametrize(
        "col_type, expected",
        [
            (T.BooleanType, ["bool"]),
            (T.DoubleType, ["double"]),
            (T.FloatType, ["float"]),
            (T.IntegerType, ["int"]),
            (T.LongType, ["long"]),
            (T.StringType, ["str"]),
            ([T.DoubleType, T.FloatType], ["double", "float"]),
            ([T.IntegerType, T.LongType], ["int", "long"]),
            (
                [T.DoubleType, T.FloatType, T.IntegerType, T.LongType],
                ["double", "float", "int", "long"],
            ),
            (None, None),
            ("BooleanType", None),
            (1, None),
            (2.0, None),
            ([None, "BooleanType"], None),
        ],
    )
    def test_select_col_types(
        self,
        spark: SparkSession,
        col_type: T.DataType,
        expected: List[str],
    ):
        df = spark.createDataFrame(
            [Row(bool=True, double=1.0, float=2.0, int=3, long=4, str="string")],
            schema=T.StructType(
                [
                    T.StructField("bool", T.BooleanType(), nullable=True),
                    T.StructField("double", T.DoubleType(), nullable=True),
                    T.StructField("float", T.FloatType(), nullable=True),
                    T.StructField("int", T.IntegerType(), nullable=True),
                    T.StructField("long", T.LongType(), nullable=True),
                    T.StructField("str", T.StringType(), nullable=True),
                ]
            ),
        )
        if expected is not None:
            actual = sk.select_col_types(df, col_type)
            assert actual == expected
        else:
            with pytest.raises(TypeError):
                sk.select_col_types(df, col_type)

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

    @pytest.mark.parametrize("func", [toolz.identity, pk.str_to_date])
    def test_with_date_diff_ago(self, spark: SparkSession, func: Callable):
        d0 = func("2024-01-01")
        df = spark.createDataFrame(
            [
                Row(i=1, d=func("2023-11-30"), expect=32),
                Row(i=2, d=func("2023-12-15"), expect=17),
                Row(i=3, d=func("2023-12-20"), expect=12),
                Row(i=4, d=func("2023-12-29"), expect=3),
                Row(i=5, d=func("2023-12-30"), expect=2),
                Row(i=6, d=func("2023-12-31"), expect=1),
                Row(i=7, d=func("2024-01-01"), expect=0),
                Row(i=8, d=func("2024-01-02"), expect=-1),
                Row(i=9, d=func("2024-01-08"), expect=-7),
                Row(i=10, d=func("2024-01-10"), expect=-9),
            ]
        )

        actual = df.transform(sk.with_date_diff_ago("d", d0, "fx")).select("i", "fx")
        expected = df.select("i", F.col("expect").cast(T.IntegerType()).alias("fx"))
        self.assert_dataframe_equal(actual, expected)

    @pytest.mark.parametrize("func", [toolz.identity, pk.str_to_date])
    def test_with_date_diff_ahead(self, spark: SparkSession, func: Callable):
        d0 = func("2024-01-01")
        df = spark.createDataFrame(
            [
                Row(i=1, d=func("2023-11-30"), expect=-32),
                Row(i=2, d=func("2023-12-15"), expect=-17),
                Row(i=3, d=func("2023-12-20"), expect=-12),
                Row(i=4, d=func("2023-12-29"), expect=-3),
                Row(i=5, d=func("2023-12-30"), expect=-2),
                Row(i=6, d=func("2023-12-31"), expect=-1),
                Row(i=7, d=func("2024-01-01"), expect=0),
                Row(i=8, d=func("2024-01-02"), expect=1),
                Row(i=9, d=func("2024-01-08"), expect=7),
                Row(i=10, d=func("2024-01-10"), expect=9),
            ]
        )

        actual = df.transform(sk.with_date_diff_ahead("d", d0, "fx")).select("i", "fx")
        expected = df.select("i", F.col("expect").cast(T.IntegerType()).alias("fx"))
        self.assert_dataframe_equal(actual, expected)

    def test_with_digitscale(self, spark: SparkSession):
        df = spark.createDataFrame(
            [
                Row(i=1, x=0.0, expect=0.0),
                Row(i=2, x=0.1, expect=0.0),
                Row(i=3, x=1.0, expect=1.0),
                Row(i=4, x=10.0, expect=2.0),
                Row(i=5, x=100.0, expect=3.0),
                Row(i=6, x=1_000.0, expect=4.0),
                Row(i=7, x=10_000.0, expect=5.0),
                Row(i=8, x=100_000.0, expect=6.0),
                Row(i=9, x=1_000_000.0, expect=7.0),
                Row(i=10, x=0.2, expect=0.30102999566398125),
                Row(i=11, x=2.0, expect=1.3010299956639813),
                Row(i=12, x=20.0, expect=2.3010299956639813),
                Row(i=13, x=-0.5, expect=0.6989700043360187),
                Row(i=14, x=-5.0, expect=1.6989700043360187),
                Row(i=15, x=-50.0, expect=2.6989700043360187),
                Row(i=16, x=None, expect=None),
            ],
        )
        actual = df.transform(sk.with_digitscale("x", "fx")).select("i", "fx")
        expected = df.select("i", F.col("expect").alias("fx"))
        self.assert_dataframe_equal(actual, expected)

    def test_with_digitscale_int(self, spark: SparkSession):
        df = spark.createDataFrame(
            [
                Row(i=1, x=0.0, expect=0),
                Row(i=2, x=0.1, expect=0),
                Row(i=3, x=1.0, expect=1),
                Row(i=4, x=10.0, expect=2),
                Row(i=5, x=100.0, expect=3),
                Row(i=6, x=1_000.0, expect=4),
                Row(i=7, x=10_000.0, expect=5),
                Row(i=8, x=100_000.0, expect=6),
                Row(i=9, x=1_000_000.0, expect=7),
                Row(i=10, x=0.2, expect=0),
                Row(i=11, x=2.0, expect=1),
                Row(i=12, x=20.0, expect=2),
                Row(i=13, x=-0.5, expect=0),
                Row(i=14, x=-5.0, expect=1),
                Row(i=15, x=-50.0, expect=2),
                Row(i=16, x=None, expect=None),
            ],
            schema=T.StructType(
                [
                    T.StructField("i", T.IntegerType(), True),
                    T.StructField("x", T.FloatType(), True),
                    T.StructField("expect", T.IntegerType(), True),
                ]
            ),
        )
        actual = df.transform(sk.with_digitscale("x", "fx", kind="int")).select(
            "i", "fx"
        )
        expected = df.select("i", F.col("expect").alias("fx"))
        self.assert_dataframe_equal(actual, expected)

    def test_with_digitscale_linear(self, spark: SparkSession):
        df = spark.createDataFrame(
            [
                Row(i=1, x=0.0, expect=0.0),
                Row(i=2, x=0.1, expect=0.0),
                Row(i=3, x=1.0, expect=1.0),
                Row(i=4, x=10.0, expect=2.0),
                Row(i=5, x=100.0, expect=3.0),
                Row(i=6, x=1_000.0, expect=4.0),
                Row(i=7, x=10_000.0, expect=5.0),
                Row(i=8, x=100_000.0, expect=6.0),
                Row(i=9, x=1_000_000.0, expect=7.0),
                Row(i=10, x=0.2, expect=0.11111111111111112),
                Row(i=11, x=2.0, expect=1.1111111111111112),
                Row(i=12, x=20.0, expect=2.111111111111111),
                Row(i=13, x=-0.5, expect=0.4444444444444445),
                Row(i=14, x=-5.0, expect=1.4444444444444444),
                Row(i=15, x=-50.0, expect=2.4444444444444446),
                Row(i=16, x=None, expect=None),
            ],
        )
        actual = df.transform(sk.with_digitscale("x", "fx", kind="linear")).select(
            "i", "fx"
        )
        expected = df.select("i", F.col("expect").alias("fx"))
        self.assert_dataframe_equal(actual, expected)

    @pytest.mark.parametrize("kind", [None, "LOG", 1, 2.0, "invalid_kind"])
    def test_with_digitscale_invalid_kind(self, kind: str):
        with pytest.raises(ValueError):
            sk.with_digitscale("x", "fx", kind=kind)

    @pytest.mark.parametrize("f", [toolz.identity, pk.str_to_date])
    def test_with_endofweek_date(self, spark: SparkSession, f: Callable):
        field_type = T.StringType() if f == toolz.identity else T.DateType()
        df = spark.createDataFrame(
            [
                Row(i=1, d=f("2023-04-30"), e1=f("2023-04-30"), e2=f("2023-05-06")),
                Row(i=2, d=f("2023-05-01"), e1=f("2023-05-07"), e2=f("2023-05-06")),
                Row(i=3, d=f("2023-05-02"), e1=f("2023-05-07"), e2=f("2023-05-06")),
                Row(i=4, d=f("2023-05-03"), e1=f("2023-05-07"), e2=f("2023-05-06")),
                Row(i=5, d=f("2023-05-04"), e1=f("2023-05-07"), e2=f("2023-05-06")),
                Row(i=6, d=f("2023-05-05"), e1=f("2023-05-07"), e2=f("2023-05-06")),
                Row(i=7, d=f("2023-05-06"), e1=f("2023-05-07"), e2=f("2023-05-06")),
                Row(i=8, d=f("2023-05-07"), e1=f("2023-05-07"), e2=f("2023-05-13")),
                Row(i=9, d=f("2023-05-08"), e1=f("2023-05-14"), e2=f("2023-05-13")),
                Row(i=10, d=None, e1=None, e2=None),
            ],
            schema=T.StructType(
                [
                    T.StructField("i", T.IntegerType(), True),
                    T.StructField("d", field_type, True),
                    T.StructField("expect_sun", field_type, True),
                    T.StructField("expect_sat", field_type, True),
                ]
            ),
        )

        actual = df.transform(sk.with_endofweek_date("d", "fx")).select("i", "fx")
        expected = df.select("i", F.col("expect_sun").alias("fx"))
        self.assert_dataframe_equal(actual, expected)

        actual = df.transform(sk.with_endofweek_date("d", "fx", "Sat")).select(
            "i", "fx"
        )
        expected = df.select("i", F.col("expect_sat").alias("fx"))
        self.assert_dataframe_equal(actual, expected)

    @pytest.mark.skip(reason="nondeterministic output")
    def test_with_increasing_id(self, spark: SparkSession):
        df = spark.createDataFrame(
            [
                Row(i=1, x="a", expect=0),
                Row(i=2, x="b", expect=8589934592),
                Row(i=3, x="c", expect=17179869184),
                Row(i=4, x="d", expect=25769803776),
                Row(i=5, x="e", expect=34359738368),
                Row(i=6, x="f", expect=42949672960),
                Row(i=7, x="g", expect=51539607552),
                Row(i=8, x="h", expect=60129542144),
            ],
            schema=T.StructType(
                [
                    T.StructField("i", T.IntegerType(), True),
                    T.StructField("x", T.StringType(), True),
                    T.StructField("expect", T.LongType(), True),
                ]
            ),
        )

        actual = df.transform(sk.with_increasing_id("fx")).select("i", "fx")
        expected = df.select("i", F.col("expect").alias("fx"))
        self.assert_dataframe_equal(actual, expected)

    def test_with_index(self, spark: SparkSession):
        df = spark.createDataFrame(
            [
                Row(i=1, x="a", expect=1),
                Row(i=2, x="b", expect=2),
                Row(i=3, x="c", expect=3),
                Row(i=4, x="d", expect=4),
                Row(i=5, x="e", expect=5),
                Row(i=6, x="f", expect=6),
                Row(i=7, x="g", expect=7),
                Row(i=8, x="h", expect=8),
            ],
            schema=T.StructType(
                [
                    T.StructField("i", T.IntegerType(), True),
                    T.StructField("x", T.StringType(), True),
                    T.StructField("expect", T.IntegerType(), True),
                ]
            ),
        )

        actual = df.transform(sk.with_index("fx")).select("i", "fx")
        expected = df.select("i", F.col("expect").alias("fx"))
        self.assert_dataframe_equal(actual, expected)

    @pytest.mark.parametrize("f", [toolz.identity, pk.str_to_date])
    def test_with_startofweek_date(self, spark: SparkSession, f: Callable):
        field_type = T.StringType() if f == toolz.identity else T.DateType()
        s2d = pk.str_to_date
        df = spark.createDataFrame(
            [
                Row(i=1, d=f("2023-04-30"), e1=s2d("2023-04-24"), e2=s2d("2023-04-30")),
                Row(i=2, d=f("2023-05-01"), e1=s2d("2023-05-01"), e2=s2d("2023-04-30")),
                Row(i=3, d=f("2023-05-02"), e1=s2d("2023-05-01"), e2=s2d("2023-04-30")),
                Row(i=4, d=f("2023-05-03"), e1=s2d("2023-05-01"), e2=s2d("2023-04-30")),
                Row(i=5, d=f("2023-05-04"), e1=s2d("2023-05-01"), e2=s2d("2023-04-30")),
                Row(i=6, d=f("2023-05-05"), e1=s2d("2023-05-01"), e2=s2d("2023-04-30")),
                Row(i=7, d=f("2023-05-06"), e1=s2d("2023-05-01"), e2=s2d("2023-04-30")),
                Row(i=8, d=f("2023-05-07"), e1=s2d("2023-05-01"), e2=s2d("2023-05-07")),
                Row(i=9, d=f("2023-05-08"), e1=s2d("2023-05-08"), e2=s2d("2023-05-07")),
                Row(i=10, d=None, e1=None, e2=None),
            ],
            schema=T.StructType(
                [
                    T.StructField("i", T.IntegerType(), True),
                    T.StructField("d", field_type, True),
                    T.StructField("expect_sun", T.DateType(), True),
                    T.StructField("expect_sat", T.DateType(), True),
                ],
            ),
        )

        actual = df.transform(sk.with_startofweek_date("d", "fx")).select("i", "fx")
        expected = df.select("i", F.col("expect_sun").alias("fx"))
        self.assert_dataframe_equal(actual, expected)

        actual = df.transform(sk.with_startofweek_date("d", "fx", "Sat")).select(
            "i", "fx"
        )
        expected = df.select("i", F.col("expect_sat").alias("fx"))
        self.assert_dataframe_equal(actual, expected)

    @pytest.mark.parametrize("func", [toolz.identity, pk.str_to_date])
    def test_with_weekday(self, spark: SparkSession, func: Callable):
        df = spark.createDataFrame(
            [
                Row(i=1, d=func("2023-05-01"), expect="Mon"),
                Row(i=2, d=func("2023-05-02"), expect="Tue"),
                Row(i=3, d=func("2023-05-03"), expect="Wed"),
                Row(i=4, d=func("2023-05-04"), expect="Thu"),
                Row(i=5, d=func("2023-05-05"), expect="Fri"),
                Row(i=6, d=func("2023-05-06"), expect="Sat"),
                Row(i=7, d=func("2023-05-07"), expect="Sun"),
                Row(i=8, d=None, expect=None),
            ]
        )
        actual = df.transform(sk.with_weekday("d", "fx")).select("i", "fx")
        expected = df.select("i", F.col("expect").alias("fx"))
        self.assert_dataframe_equal(actual, expected)

    def test_spark_session(self, spark: SparkSession):
        assert isinstance(spark, SparkSession)
        assert spark.sparkContext.appName == "spark-session-for-testing"
        assert spark.version == "3.1.1"

    @staticmethod
    def assert_dataframe_equal(lft_df: SparkDF, rgt_df: SparkDF) -> None:
        """Assert that the left and right data frames are equal."""
        sk.assert_dataframe_equal(lft_df, rgt_df)

    @pytest.fixture(scope="class")
    def spark(self, request: pytest.FixtureRequest) -> SparkSession:
        spark = (
            SparkSession.builder.master("local[*]")
            .appName("spark-session-for-testing")
            .config("spark.executor.instances", 1)
            .config("spark.executor.cores", 4)
            .config("spark.default.parallelism", 4)
            .config("spark.sql.shuffle.partitions", 4)
            .config("spark.rdd.compress", False)
            .config("spark.shuffle.compress", False)
            .config("spark.dynamicAllocation.enabled", False)
            .config("spark.speculation", False)
            .config("spark.ui.enabled", False)
            .config("spark.ui.showConsoleProgress", False)
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("OFF")
        request.addfinalizer(lambda: spark.stop())
        return spark
