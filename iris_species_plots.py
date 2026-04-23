#!/usr/bin/env python3
"""
Perform and plot linear regression for each Iris species separately.
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def plot_species_regression(dataframe, species_name):
    """
    Perform linear regression and plot for a given species.

    Parameters
    ----------
    dataframe : pandas.DataFrame
    species_name : str
    """

    species_df = dataframe[dataframe.species == species_name]

    x = species_df.petal_length_cm
    y = species_df.sepal_length_cm

    regression = stats.linregress(x, y)
    slope = regression.slope
    intercept = regression.intercept

    plt.figure()  # separate plot for each species

    plt.scatter(x, y, label=f"{species_name} Data")
    plt.plot(x, slope * x + intercept, color="orange", label="Fitted line")

    plt.xlabel("Petal length (cm)")
    plt.ylabel("Sepal length (cm)")
    plt.title(species_name)

    plt.legend()

    filename = f"{species_name}_regression.png"
    plt.savefig(filename)
    print(f"Saved: {filename}")


def main():
    dataframe = pd.read_csv("iris.csv")

    species_list = [
        "Iris_setosa",
        "Iris_virginica",
        "Iris_versicolor"
    ]

    for species in species_list:
        plot_species_regression(dataframe, species)


if __name__ == '__main__':
    main()
