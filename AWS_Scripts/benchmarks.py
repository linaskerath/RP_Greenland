import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

pd.options.mode.chained_assignment = None

import functions_training_pipeline as f

import smtplib
import configparser
from email.message import EmailMessage


df_path = r"/mnt/volume/AWS_Data/Data/dataframe_model_training/training_data_UPPER_HALF.parquet.gzip"
data = pd.read_parquet(df_path)

print("Mean benchmark")
mean_benchmark_model = f.Model(model=None, name="MeanBenchmark")
columns = ["mw_value"]
mean_benchmark_model.spatial_cv_mean_benchmark(data, data.columns, target_normalized=True)

print()

print("Microwave benchmark")
y_predictions_mw = f.model_mwBenchmark(data)
# transform back from log transform
data["opt_value"] = np.exp(data["opt_value"]) - 1
print(f"Microwave benchmark RMSE on test set: {mean_squared_error(data['opt_value'], data['mw_value'])}")
print(f"Microwave benchmark R2 on test set: {r2_score(data['opt_value'], data['mw_value'])}")


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
