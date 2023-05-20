import pandas as pd
from sklearn.ensemble import RandomForestRegressor

pd.options.mode.chained_assignment = None

import functions_training_pipeline as f

import smtplib
import configparser
from email.message import EmailMessage

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
rf = f.Model(model=RandomForestRegressor, name="RandomForest")
hyperparameters_for_grid = {
    "max_features": ["sqrt", 9, None],
    # "max_samples": [0.7],  # [0.6, None],
    "min_samples_leaf": [2, 6],  # [2, 5, 10],
    # "n_estimators": [150],  # [300],
    "max_depth": [7, 12],
    "n_jobs": [-1]
    # "warm_start": [True],
}
rf.hyperparameters = rf.create_hyperparameter_grid(hyperparameters_for_grid)

print("Reading data...")
data = pd.read_parquet(df_path)
columns = data.columns.drop(["opt_value"])
print("Starting training...")
rf.spatial_cv(data, columns, target_normalized=True, tune_hyperparameters=False)

f.save_object(rf)


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
