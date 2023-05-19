from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression

import smtplib
import configparser
from email.message import EmailMessage

import functions_training_pipeline_pyspark as f_pyspark

df_path = r"/mnt/volume/AWS_Data/Data/dataframe_model_training/training_data.parquet.gzip"

print("Building Spark session...")
spark = (
    SparkSession.builder.appName("Linear Regression Training")
    .config("spark.ui.reverseProxy", "true")
    .config("spark.executor.cores", "4")
    .config("spark.executor.instances", "8")
    .config("spark.executor.memory", "32g")
    .config("spark.driver.memory", "16g")
    # .config("spark.storage.memoryFraction", "0.4")
    # .config("spark.memory.offHeap.enabled", "true")
    # .config("spark.memory.offHeap.size", "4g")
    .getOrCreate()
)

# StorageLevel.MEMORY_ONLY_SER?

print("Instantiating model...")
lr = f_pyspark.Model(model=LinearRegression(), name="LinearRegression")
hyperparameters_for_grid = {"fitIntercept": [True]}
lr.hyperparameters = lr.create_hyperparameter_grid(hyperparameters_for_grid)

print("Reading data...")
data = spark.read.parquet(df_path)
columns = data.drop("opt_value").columns
print("Starting spatial cross validation...")
lr.spatial_cv(data, columns, target_normalized=True)

print("Saving model...")
f_pyspark.save_object(lr)

spark.stop()


# Read email credentials from config file
config = configparser.ConfigParser()
config.read("config.ini")
sender = config.get("Email", "sender_email")
password = config.get("Email", "sender_password")
recipient = config.get("Email", "recipient_email")

# create email
msg = EmailMessage()
msg["Subject"] = "AWS Finished"
msg["From"] = sender
msg["To"] = recipient
msg.set_content("AWS instance is done.")

# send email
with smtplib.SMTP_SSL("smtp.gmx.com", 465) as smtp:
    smtp.login(sender, password)
    smtp.send_message(msg)
    print("Email sent.")
