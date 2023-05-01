import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

pd.options.mode.chained_assignment = None

import functions_training_pipeline as f

import smtplib
import configparser
from email.message import EmailMessage

df_path = r"../AWS_Data/Data/dataframe_extended/"

date_from = "2019-06-10"
date_to = "2019-06-11"

data = f.import_data(date_from, date_to, df_path)
data = f.remove_data(data, removeMaskedClouds=True, removeNoMelt=True)
data = f.data_normalization(data)

columns = data.columns.drop(["date", "row", "col", "opt_value"])

xgb = f.Model(model=GradientBoostingRegressor, name="XGBoost")
hyperparameters_for_grid = {"min_samples_split": [3, 5], "learning_rate": [0.1, 0.5]}
xgb.hyperparameters = xgb.create_hyperparameter_grid(hyperparameters_for_grid)

xgb.spatial_cv(data, columns)

f.save_object(xgb)


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
