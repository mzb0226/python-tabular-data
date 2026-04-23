#! /usr/bin/env python3

"""
Perform and plot linear regression for all three iris species: Iris_setosa, Iris_virginica, Iris_versicolor"

This script reads iris data from a CSV file, computes linear regression
between petal length and sepal length for each species, and saves a plot.
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def plot_species_regression(dataframe, species_name):
    subset = dataframe[dataframe.species == species_name]
    print(subset)

    x = subset.petal_length_cm
    y = subset.sepal_length_cm

    regression = stats.linregress(x, y)

    plt.scatter(x, y, label=f'{species_name} Data')
    plt.plot(x, regression.slope * x + regression.intercept,
             label=f'{species_name} Fit')

    plt.xlabel(f"{species_name} Petal length (cm)")
    plt.ylabel(f"{species_name} Sepal length (cm)")
    plt.legend()


if __name__ == '__main__':
    dataframe = pd.read_csv("iris.csv")

    species_list = [
        "Iris_setosa",
        "Iris_virginica",
        "Iris_versicolor"
    ]

    for species in species_list:
        plot_species_regression(dataframe, species)

    plt.savefig("species_petal_v_sepal_length_regress.png")
