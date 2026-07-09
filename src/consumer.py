import sys
import time
import os
import csv
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer, SQLTransformer, IndexToString
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import sparknlp
from sparknlp.base import DocumentAssembler, EmbeddingsFinisher
from sparknlp.annotator import UniversalSentenceEncoder

# --- Configuration ---
TOPIC_NAME = "sentiment-tweets"
BOOTSTRAP_SERVERS = 'localhost:9092'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(BASE_DIR, '..', 'data', 'tweets.csv')

CHECKPOINT_LOCATION = "./checkpoint"
RESULTS_LOG_FILE = "experiment_results.csv"
TRAINING_SIZE = 5000  # Must match producer.py

# Classes present in the dataset
# (positive, negative, litigious, uncertainty)


# --- Global Metrics Tracker ---
class StreamingMetrics:
    total_processed = 0
    total_correct = 0

    @classmethod
    def update(cls, count, correct):
        cls.total_processed += count
        cls.total_correct += correct

    @classmethod
    def get_accuracy(cls):
        return cls.total_correct / cls.total_processed if cls.total_processed > 0 else 0.0


# --- Initialize Spark ---
# Requirement 1: Using Spark with Python
# Note: Use "spark-nlp-silicon" for Apple Silicon Mac.
#       Use "spark-nlp" for Linux/Windows/Intel Mac.
spark = SparkSession.builder \
    .appName("SentimentAnalysis_Project") \
    .config("spark.jars.packages",
            "com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.3,org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# --- Initialize Results Log ---
with open(RESULTS_LOG_FILE, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["BatchID", "Accuracy", "Precision", "Recall", "CumulativeAccuracy"])

print("--- Phase 1: Training Model ---")

# 1. Load Training Data
try:
    raw_data = spark.read.option("header", True).csv(CSV_FILE_PATH)
except Exception as e:
    print(f"Error loading CSV: {e}")
    sys.exit(1)

# Normalize column names — dataset uses "Text" and "Label" directly
if "clean_text" in raw_data.columns:
    raw_data = raw_data.withColumnRenamed("clean_text", "Text")
if "category" in raw_data.columns:
    raw_data = raw_data.withColumnRenamed("category", "Label")

# Filter to known labels and take training split
VALID_LABELS = ['positive', 'negative', 'litigious', 'uncertainty']
train_data = raw_data \
    .filter(F.col("Text").isNotNull() & F.col("Label").isNotNull()) \
    .filter(F.col("Label").isin(VALID_LABELS)) \
    .limit(TRAINING_SIZE)

# 2. Build Spark NLP + MLlib Pipeline
document_assembler = DocumentAssembler() \
    .setInputCol("Text") \
    .setOutputCol("document")

# Universal Sentence Encoder — 512-dim semantic embeddings
use = UniversalSentenceEncoder.pretrained() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence_embeddings")

embeddings_finisher = EmbeddingsFinisher() \
    .setInputCols(["sentence_embeddings"]) \
    .setOutputCols(["finished_embeddings"]) \
    .setOutputAsVector(True)

explode_vectors = SQLTransformer(
    statement="SELECT *, finished_embeddings[0] AS features FROM __THIS__"
)

# handleInvalid="keep" prevents crashes on unseen labels during streaming
label_indexer = StringIndexer(inputCol="Label", outputCol="label_index") \
    .setHandleInvalid("keep") \
    .fit(train_data)

label_converter = IndexToString(
    inputCol="prediction",
    outputCol="predictedLabel",
    labels=label_indexer.labels
)

log_reg = LogisticRegression(featuresCol="features", labelCol="label_index", maxIter=10)

pipeline = Pipeline(stages=[
    document_assembler,
    use,
    embeddings_finisher,
    explode_vectors,
    label_indexer,
    log_reg
])

model = pipeline.fit(train_data)
print("Model trained successfully.")
print(f"Classes: {label_indexer.labels}")

# --- Phase 2: Streaming Inference ---
# Requirement 2: Using Kafka as stream processor
print(f"--- Phase 2: Listening to Kafka Topic '{TOPIC_NAME}' ---")

stream_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", BOOTSTRAP_SERVERS) \
    .option("subscribe", TOPIC_NAME) \
    .option("startingOffsets", "latest") \
    .option("failOnDataLoss", "false") \
    .load()

schema = StructType().add("Text", StringType()).add("Label", StringType())

parsed_stream = stream_df.select(
    F.from_json(F.col("value").cast("string"), schema).alias("data")
).select("data.*")


def process_batch(batch_df, batch_id):
    """Process each micro-batch: predict, evaluate, and log results."""
    batch_df.persist()

    if batch_df.count() > 0:
        try:
            preds = model.transform(batch_df)
            preds = label_converter.transform(preds)

            if "label_index" in preds.columns and "prediction" in preds.columns:

                # Accuracy
                evaluator_acc = MulticlassClassificationEvaluator(
                    labelCol="label_index", predictionCol="prediction", metricName="accuracy")
                accuracy = evaluator_acc.evaluate(preds)

                # Precision
                evaluator_prec = MulticlassClassificationEvaluator(
                    labelCol="label_index", predictionCol="prediction", metricName="weightedPrecision")
                precision = evaluator_prec.evaluate(preds)

                # Recall
                evaluator_rec = MulticlassClassificationEvaluator(
                    labelCol="label_index", predictionCol="prediction", metricName="weightedRecall")
                recall = evaluator_rec.evaluate(preds)

                # Cumulative accuracy
                correct = preds.filter("label_index == prediction").count()
                total = preds.count()
                StreamingMetrics.update(total, correct)
                cum_acc = StreamingMetrics.get_accuracy()

                print(
                    f"Batch {batch_id}: Acc={accuracy:.2%} | Prec={precision:.2%} | "
                    f"Rec={recall:.2%} | Cumulative Acc={cum_acc:.2%}"
                )

                with open(RESULTS_LOG_FILE, "a") as f:
                    writer = csv.writer(f)
                    writer.writerow([batch_id, accuracy, precision, recall, cum_acc])

                preds.select("Text", "Label", "predictedLabel").show(5, truncate=40)

        except Exception as e:
            print(f"Batch {batch_id} Warning: {e}")

    batch_df.unpersist()


query = parsed_stream.writeStream \
    .outputMode("append") \
    .foreachBatch(process_batch) \
    .option("checkpointLocation", CHECKPOINT_LOCATION) \
    .start()

query.awaitTermination()
