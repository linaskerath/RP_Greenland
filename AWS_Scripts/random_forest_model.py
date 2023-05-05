import pandas as pd
from sklearn.ensemble import RandomForestRegressor

pd.options.mode.chained_assignment = None

import functions_training_pipeline as f

import smtplib
import configparser
from email.message import EmailMessage

df_path = r"/mnt/volume/AWS_Data/Data/dataframe_model_training/training_data.parquet.gzip"

rf = f.Model(model=RandomForestRegressor, name="RandomForest")
hyperparameters_for_grid = {
    "max_features": ["sqrt", 9, None],
    "max_samples": [0.6, None],
    "min_samples_leaf": [2, 5, 10],
    "n_estimators": [300],
}
rf.hyperparameters = rf.create_hyperparameter_grid(hyperparameters_for_grid)

data = pd.read_parquet(df_path)
columns = data.columns.drop(["opt_value"])
rf.spatial_cv(data, columns)

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
