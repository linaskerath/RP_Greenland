import pandas as pd
from sklearn.linear_model import ElasticNet
import smtplib
import configparser
from email.message import EmailMessage

pd.options.mode.chained_assignment = None

import functions_training_pipeline as f

import sys


# Define a custom stream writer class
class ConsoleWriter:
    def write(self, message):
        sys.__stdout__.write(message)
        sys.__stdout__.flush()


# Replace sys.stdout with the custom writer
sys.stdout = ConsoleWriter()


df_path = r"/mnt/volume/AWS_Data/Data/dataframe_model_training/training_data_UPPER_HALF.parquet.gzip"

print("Creating model object...")
elasticnet = f.Model(model=ElasticNet, name="ElasticNetRegression")
hyperparameters_for_grid = {"alpha": [0.5, 1, 2, 5, 10], "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]}
elasticnet.hyperparameters = elasticnet.create_hyperparameter_grid(hyperparameters_for_grid)

print("Reading data...")
data = pd.read_parquet(df_path)
columns = data.columns.drop(["opt_value"])
print("Starting training...")
elasticnet.spatial_cv(data, columns, target_normalized=True, tune_hyperparameters=True)

f.save_object(elasticnet)

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
