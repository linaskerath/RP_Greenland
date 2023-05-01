import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

pd.options.mode.chained_assignment = None

import functions_training_pipeline as f

import smtplib
import configparser
from email.message import EmailMessage

df_path = r"../Data/dataframe_extended/"

date_from = "2017-05-01"
date_to = "2019-07-31"


data = f.import_data(date_from, date_to, df_path)
data = f.remove_data(data, removeMaskedClouds=True, removeNoMelt=True)
data = f.data_normalization(data)

print("Mean benchmark")
mean_benchmark_model = f.Model(model= None, name="MeanBenchmark")
columns = ['mw_value']
mean_benchmark_model.spatial_cv_mean_benchmark(data, columns)


print("Microwave benchmark")
y_predictions_mw = f.model_mwBenchmark(data)
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
