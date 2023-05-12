import pandas as pd
from sklearn.tree import DecisionTreeRegressor

pd.options.mode.chained_assignment = None

import functions_training_pipeline as f

import smtplib
import configparser
from email.message import EmailMessage

df_path = r"/mnt/volume/AWS_Data/Data/dataframe_model_training/training_data.parquet.gzip"

dt = f.Model(model=DecisionTreeRegressor, name="DecisionTree")
hyperparameters_for_grid = {
    "max_depth": [3, 10, None],
    "min_samples_split": [2, 10],
    "min_samples_leaf": [1, 4],
    "max_features": ["auto", None],
}
dt.hyperparameters = dt.create_hyperparameter_grid(hyperparameters_for_grid)

data = pd.read_parquet(df_path)
columns = data.columns.drop(["opt_value"])
dt.spatial_cv(data, columns, target_normalized=True)

f.save_object(dt)


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