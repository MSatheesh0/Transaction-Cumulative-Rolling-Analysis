# -*- coding: utf-8 -*-
"""Untitled8.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1rgRHXWyrFXyzQ0Chxuug2yJxsUzRUIHR
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from datetime import date

s=SparkSession.builder.appName("Analysis").getOrCreate()

d=[(1,date(2024,2,1),150.0),
    (1,date(2024,2,2),300.0),
    (1,date(2024,2,3),100.0),
    (1,date(2024,2,7),200.0),
    (2,date(2024,2,4),500.0),
    (2,date(2024,2,5),250.0),
    (2,date(2024,2,10),400.0)]

sch=["id","t_date","amt"]
t=s.createDataFrame(d,schema=sch)

# Show Step 3 Output
print("Initial DataFrame:")
t.show()

w=Window.partitionBy("id").orderBy("t_date")\
    .rowsBetween(Window.unboundedPreceding,Window.currentRow)

t=t.withColumn("cum_amt",F.sum("amt").over(w))

# Show Step 5 Output
print("DataFrame after adding cumulative amt column:")
t.show()

t=t.withColumn(
    "t_date_days",
    F.unix_timestamp(F.col("t_date"),"yyyy-MM-dd")/(24*60*60)
)

# Show Step 6 Output
print("DataFrame after converting t_date to t_date_days:")
t.show()

rw=Window.partitionBy("id").orderBy("t_date_days")\
    .rangeBetween(-6,0)

t=t.withColumn(
    "roll_avg_amt",
    F.avg("amt").over(rw)
)

# Show Step 8 Output
print("DataFrame after calculating 7-day rolling average:")
t.show()

s.stop()
