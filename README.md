# realtime-tweet-sentiment-using-pyspark-kafka
Real-time tweet sentiment classification pipeline using Kafka, PySpark Structured Streaming, and Spark-NLP embeddings with a Logistic Regression classifier.
## Tech Stack
- **Apache Kafka** — real-time message streaming
- **PySpark (Structured Streaming)** — distributed micro-batch processing
- **Spark-NLP** — Universal Sentence Encoder for text embeddings
- **Spark MLlib** — Logistic Regression classifier
- **Python** (`kafka-python`, `pyspark`)

## Dataset
[Sentiment Dataset with 1M Tweets](https://www.kaggle.com/datasets/tariqsays/sentiment-dataset-with-1-million-tweets) (Kaggle)
- Columns: `Text`, `Language`, `Label`
- Classes: `positive`, `negative`, `litigious`, `uncertainty`
- A small sample (`data/sample_tweets.csv`) is included in this repo for quick testing; 
  the full dataset can be downloaded from the link above.

## How It Works

**Training phase:** The first 5,000 valid rows are reserved and used to train the model — 
tweets are embedded via Spark-NLP's Universal Sentence Encoder, then fed into a Logistic 
Regression classifier via a Spark ML `Pipeline`.

**Streaming phase:** The producer sends all remaining rows to a Kafka topic, simulating live 
tweets. The consumer processes each micro-batch through the trained pipeline, computes accuracy, 
weighted precision, weighted recall, and cumulative accuracy across the whole stream, then 
logs results to `experiment_results.csv`.

## Setup & Usage

### Prerequisites
- Apache Kafka & Zookeeper running locally (`localhost:9092`)
- Python 3.10+
- Java 8/11 (required by Spark)

### Install dependencies
```bash
pip install pyspark spark-nlp kafka-python pandas
```

### Run
1. Start Zookeeper and Kafka locally.
2. Start the consumer (trains model, then listens for stream):
```bash
python src/consumer.py
```
3. In a separate terminal, start the producer (streams tweets):
```bash
python src/producer.py
```
4. Watch live predictions and metrics print to the consumer terminal, and check 
   `experiment_results.csv` for the full log.

## Results

Evaluated over 438 consecutive streaming batches (Batch 1 excluded as a pipeline 
initialization artifact):

| Metric | Value |
|---|---|
| Final Cumulative Accuracy | **69.00%** |
| Peak Cumulative Accuracy | 75.32% (Batch 6) |
| Mean Precision | 72.79% |
| Mean Recall | 69.00% |
| Batches Processed | 438 |

The model handles 4-way sentiment classification (positive, negative, litigious, 
uncertainty) — a harder task than typical binary sentiment analysis. Precision 
consistently exceeds recall, indicating a low false-positive rate across classes. 
Full analysis available in [`Report and Result/Report.pdf`](./Report%20and%20Result/Report.pdf).

## Author
Subhadip Maity — [GitHub](https://github.com/subhadip191) · [LinkedIn](https://linkedin.com/in/subhadipmaity191)
