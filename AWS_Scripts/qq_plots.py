import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import probplot

path_optical_values = r"/mnt/volume/AWS_Data/Plots/optical_values.csv"
# path_optical_values = r"../AWS_Data/Plots/optical_values.csv"

path_out = r"/mnt/volume/AWS_Data/Plots"
# path_out = r"../AWS_Data/Plots"

df = pd.read_csv(path_optical_values)

# cap all values of above 5 to 5
df["optical_values"] = df["optical_values"].apply(lambda x: 5 if x > 5 else x)

for i in ["Raw", "Log-Transformed"]:
    # Plot the QQ plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if i == "Raw":
        res = probplot(df["optical_values"], plot=ax)
    else:
        res = probplot(np.log(1 + df["optical_values"]), plot=ax)

    # Adjust colors
    ax.get_lines()[0].set_markerfacecolor("#D3D3D3")
    ax.get_lines()[0].set_markeredgecolor("#808080")
    # increase marker size#D3D3D3
    ax.get_lines()[0].set_markersize(10)
    # make edge color thicker
    ax.get_lines()[0].set_markeredgewidth(1.5)

    ax.get_lines()[1].set_color("firebrick")
    ax.get_lines()[1].set_linewidth(2.0)

    # Add title and labels
    ax.set_title(f"{i} Data", fontsize=20)
    ax.set_xlabel("Theoretical Quantiles", fontsize=17)
    ax.set_ylabel("Data Quantiles", fontsize=17)
    # increase tick label size
    ax.tick_params(axis="both", which="major", labelsize=15)

    # Save the plot as a PNG file
    fig.savefig(rf"{path_out}/qq_plot_{i}.png", dpi=300, bbox_inches="tight")
