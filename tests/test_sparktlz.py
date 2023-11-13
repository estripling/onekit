from pyspark.sql import SparkSession


def test_spark_session(spark: SparkSession):
    assert isinstance(spark, SparkSession)
    assert spark.sparkContext.appName == "spark-session-for-testing"
    assert spark.version == "3.1.1"
