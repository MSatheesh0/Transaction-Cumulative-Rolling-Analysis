from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from datetime import date

# Initialize Spark session
spark=SparkSession.builder.appName("Customer Transactions Analysis").getOrCreate()

# Updated sample data for customer transactions
data = [(1,date(2024,2,1),150.0),
    (1,date(2024,2,2),300.0),
    (1,date(2024,2,3),100.0),
    (1,date(2024,2,7),200.0),
    (2,date(2024,2,4),500.0),
    (2,date(2024,2,5),250.0),
    (2,date(2024,2,10),400.0)]

# Step 3: Define schema and create DataFrame
schema=["customer_id","transaction_date","amount"]
transactions=spark.createDataFrame(data,schema=schema)

# Show Step 3 Output
print("Initial DataFrame:")
transactions.show()

# Step 4: Define window for cumulative sum per customer ordered by transaction_date
cumulative_window = Window.partitionBy("customer_id").orderBy("transaction_date") \
    .rowsBetween(Window.unboundedPreceding, Window.currentRow)

# Step 5: Calculate cumulative amount for each transaction
transactions = transactions.withColumn("cumulative_amount",F.sum("amount").over(cumulative_window))

# Show Step 5 Output
print("DataFrame after adding cumulative amount column:")
transactions.show()

# Step 6: Convert transaction_date to unix timestamp (number of days since epoch) for the rolling window
transactions = transactions.withColumn(
    "transaction_date_days",
    F.unix_timestamp(F.col("transaction_date"), "yyyy-MM-dd") / (24 * 60 * 60)  # Convert to days)

# Show Step 6 Output
print("DataFrame after converting transaction_date to transaction_date_days:")
transactions.show()

# Step 7: Define 7-day rolling window for each customer using transaction_date_days
rolling_window = Window.partitionBy("customer_id").orderBy("transaction_date_days") \
    .rangeBetween(-6, 0)  # Rolling 7 days

# Step 8: Calculate rolling average transaction amount over past 7 days
transactions = transactions.withColumn(
    "rolling_avg_amount",
    F.avg("amount").over(rolling_window)
)

# Show Step 8 Output
print("DataFrame after calculating 7-day rolling average:")
transactions.show()

# Step 9: Stop Spark session
spark.stop()
