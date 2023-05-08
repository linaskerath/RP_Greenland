import pandas as pd
from sklearn.linear_model import Ridge
import numpy as np

pd.options.mode.chained_assignment = None

import functions_training_pipeline as f

import smtplib
import configparser
from email.message import EmailMessage

df_path = r"/mnt/volume/AWS_Data/Data/dataframe_model_training/training_data.parquet.gzip"

ridge = f.Model(model=Ridge, name="RidgeRegression_WithoutLog")
# chose only best hyperparameter
hyperparameters_for_grid = {"alpha": [20]}
ridge.hyperparameters = ridge.create_hyperparameter_grid(hyperparameters_for_grid)

data = pd.read_parquet(df_path)
# transform target variable back to original scale
data["opt_value"] = np.exp(data["opt_value"]) - 1
columns = data.columns.drop(["opt_value"])
ridge.spatial_cv(data, columns, target_normalized=False)

f.save_object(ridge)


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
