"""
PySpark distributed-processing pipeline for the Airline Passenger Satisfaction dataset.
Corresponds to Report Section 5 (Big Data Analysis with Apache Spark / PySpark).

Run with: spark-submit src/spark_pipeline.py
Requires a local JDK 11 install and JAVA_HOME set (see Report Section 5.4).
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator


def build_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("AirlineSatisfactionBigData")
        .config("spark.driver.memory", "6g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def run(csv_path: str = "data/airline_passenger_satisfaction.csv"):
    spark = build_spark_session()

    sdf = spark.read.csv(csv_path, header=True, inferSchema=True)
    sdf = sdf.drop("_c0", "id")

    # Distributed SQL aggregation
    sdf.createOrReplaceTempView("passengers")
    agg_result = spark.sql("""
        SELECT Class, `Customer Type`,
               COUNT(*) AS total_passengers,
               ROUND(AVG(CASE WHEN satisfaction = 'satisfied' THEN 1 ELSE 0 END), 3) AS satisfaction_rate
        FROM passengers
        GROUP BY Class, `Customer Type`
        ORDER BY satisfaction_rate DESC
    """)
    agg_result.show(truncate=False)

    # Categorical indexing
    indexers = [
        StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep")
        for c in ["Gender", "Customer Type", "Type of Travel", "Class", "satisfaction"]
    ]
    for indexer in indexers:
        sdf = indexer.fit(sdf).transform(sdf)

    feature_cols = [
        "Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes",
        "Inflight wifi service", "Online boarding", "Seat comfort",
        "Class_idx", "Gender_idx", "Customer Type_idx", "Type of Travel_idx",
    ]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
    sdf_vec = assembler.transform(sdf).select("features", F.col("satisfaction_idx").alias("label"))

    train_df, test_df = sdf_vec.randomSplit([0.8, 0.2], seed=42)

    rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=100, maxDepth=8, seed=42)
    rf_model = rf.fit(train_df)
    predictions = rf_model.transform(test_df)

    evaluator_auc = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
    evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
    print("ROC-AUC:", evaluator_auc.evaluate(predictions))
    print("Accuracy:", evaluator_acc.evaluate(predictions))

    spark.stop()


if __name__ == "__main__":
    run()
