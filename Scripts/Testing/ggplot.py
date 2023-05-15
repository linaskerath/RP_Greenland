import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import probplot

path_optical_values = r"/mnt/volume/AWS_Data/Data/optical_values.csv"


df = pd.read_csv(path_optical_values)

# cap all values of above 5 to 5
df["optical_values"] = df["optical_values"].apply(lambda x: 5 if x > 5 else x)

# Plot the QQ plot
fig, ax = plt.subplots(figsize=(8, 6))
probplot(df["optical_values"], plot=ax)

# Add title and labels
ax.set_title("QQ Plot", fontsize=14)
ax.set_xlabel("Theoretical Quantiles", fontsize=12)
ax.set_ylabel("Sample Quantiles", fontsize=12)


# Save the plot as a PNG file
fig.savefig("qq_plot.png", dpi=300, bbox_inches="tight")
