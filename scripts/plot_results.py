"""
This file contains two classes for aggregating and plotting data. One class aggregates data of the various
json files and ultimately constructs a csv file that contains the all the data. The other uses that data
and has a few methods to select a subset of that data and plot it
"""

import typing

import glob
import json
import collections
from os import path, getenv, sep


import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


import pandas as pd


# the following methods are used for averaging data points. They all take a list, which are values
# given by various optimization runs and return a float.


def mean_of_square_root(values: list) -> float:
    """
    Reduction method to reduce the given values into one float by first applying
    a square root to all values and then averaging them by using "mean"-method"

    Args:
        values: (list)
            A list of values to be reduced.

    Returns:
        (float)
            The reduced value.
    """
    return np.mean([np.sqrt(value) for value in values])


def mean_of_annealing_computing_cost(values: list) -> float:
    """
    Returns the mean of the computing cost of a list of durations of quantum annealing

    This calculates the mean of the argument, but also provides scaling by a linear factor
    that is read from the environment variables because the cost depends on your contract
    In order to get the correct value, you have to export the `costperhour` variable with
    the correct price

    Args:
        values: (list)
            A list of quantum annealing durations to be averaged

    Returns:
        (float) The mean of the duration scaled by the cost per hour
    """
    computing_cost_per_hour = getenv("costperhour", "1000")
    print(f"Using a cost of {computing_cost_per_hour} for an hour of quantum annealing")
    return np.mean([float(computing_cost_per_hour) * value for value in values])


def deviation_of_the_mean(values: list) -> float:
    """
    Reduction method to reduce the given values by calculating the deviation
    of the mean

    Args:
        values: (list)
            A list of values to be reduced.

    Returns:
        (float)
            The reduced value.
    """
    return np.std(values) / np.sqrt(len(values))


def average_of_better_than_median(values: list) -> float:
    """
    Calculates the mean of the all values that are greater than the
    median of the passed list

    Args:
        values: (list)
            A list of values to be reduced.

    Returns:
        (float)
            The reduced value.
    """
    median = np.median(values)
    result = 0
    count = 0
    for val in values:
        if val > median:
            continue
        count += 1
        result = +val
    return float(result) / count


def average_of_best(values: list) -> float:
    """
    A method to calculate the mean of the list and ommitting the highest value

    Args:
        values: (list)
            A list of values to be reduced.

    Returns:
        (float) The reduced value.
    """
    values.sort()
    return np.mean(values[:-1])


def average_of_best_percent(values: list, percentage: float) -> float:
    """
    A method to calculate the mean of the best values of a list

    Args:
        values: (list)
            A list of values to be reduced
        percentage: (float)
            A value between 0.0 and 1.0. It is rounded to determine how
            many of the values are used to calculate a mean

    Returns:
        (float) The reduced value.
    """
    values.sort()
    return np.mean(values[: int(percentage * len(values))])


def cumulative_distribution(values: list) -> list:
    """
    Reduction method to construct a cumulative_distribution for a list of lists of values
    It returns a list of values that a histogramm of that list will look like the cumulative
    distribution of all values

    Args:
        values: (list)
            A list of lists of values for which to construct a cumulative distribution

    Returns:
        (list) A list which's histogramm is the cumulative distribution of the input
    """
    result = []
    max_val = max(max(values, key=max)) + 1
    for value_lists in values:
        for val in value_lists:
            result += list(range(int(val), int(max_val), 1))
    return [result]


class PlottingAgent:
    """
    class for creating plots. on initialization reads and prepares data files. Plots
    based on that data can then be created and written by calling `make_figure`
    """

    def __init__(self, fileformat="png"):
        """
        A constructor for a plotting agent. This should be called by the class method
        which handles the correct initialization of the data_extractor, which contains
        the data the plots are based on. After complete initialization, all plots are
        based on the data in the pandas DataFrame `self.data_extractor.df`

        Args:
            fileformat: (str) The fileformat in which to save the plots
        """
        self.savepath = "plots"
        self.fileformat = fileformat
        self.data_extractor = None

    @classmethod
    def read_csv(cls, csv_file: str, fileformat="png"):
        """
        constructor method to set up the data to be plotted by reading a csv file

        Args:
            csv_file: (str)
                name and location of the file
            fileformat: (str)
                fileformat of the plots that an instance makes

        Returns:
            (PlottingAgent)
                a PlottingAgent initialized with the data in the csv file
        """
        agent = PlottingAgent(fileformat)
        agent.data_extractor = DataExtractor.read_csv(csv_file)
        return agent

    @classmethod
    def extract_from_json(cls, glob_dict, constraints={}, fileformat="png"):
        """
        constructor method to set up the data to be plotted by reading the json files
        of the runs that are to be plotted

        The files that are extracted can be specified in two ways. The `glob_dict`
        allows you to include json files, and the `constraints` argument allows you
        to filter by runs with have specific values (usually configuration values)

        Args:
            glob_dict: (dict)
                a dictionary with solver names as keys and a list of glob expressions as values
                For a given `solver: glob_expr` pair, the directory `results_{solver}_sweep` is
                searched using the `glob_expr`
            constraints: (dict)
                a dictionary with data fields name as keys and admissable values in a list
            fileformat: (str)
                fileformat of the plots that an instance makes

        Returns:
            (PlottingAgent) a PlottingAgent initialized with the data of the run files
        """
        agent = PlottingAgent(fileformat)
        agent.data_extractor = DataExtractor.extract_from_json(
            glob_dict=glob_dict,
            constraints=constraints,
        )
        return agent

    def to_csv(self, filename: str):
        """
        wrapper for writing the data into a csv file

        Args:
            filename: (str) location where to save as csv

        Returns:
            (None) creates a csv file on disk
        """
        self.data_extractor.df.to_csv(filename)

    def get_data_points(
        self,
        x_field: str,
        y_field_list: list,
        splitting_fields: list,
        constraints: dict,
    ):
        """
        returns all stored data points with x-value in x_field and y-values in y_field_list.
        The result is a dictonary with keys being the index of the groupby by the
        chosen splitting_fields. The value is a dictionary over the chosen y_fields with values being a pair or
        lists. The first one consists of x-values and the second entry consists of unsanitized y-values

        Args:
            x_field: (str)
                label of the columns containing x-values
            y_field_list: (str)
                list of labels of the columns containing y-values
            splitting_fields: (list)
                list of labels by which columns to groupby the points
            constraints: (dict)
                rules for restricting which data to access from the stored data frame

        Returns:
            (dict) a dictonary over groupby indices with values being dictionaries over y_field labels
        """
        result = {}
        try:
            data_frame = self.data_extractor.get_data(constraints)[
                [x_field] + y_field_list + splitting_fields
            ]
        except KeyError:
            return result
        if splitting_fields:
            grouped_data_frame = data_frame.groupby(splitting_fields, dropna=False).agg(
                list
            )
            for idx in grouped_data_frame.index:
                # if there is only 1 split_field, the index is not a multiindex
                if len(splitting_fields) == 1:
                    idx_tuple = [idx]
                else:
                    idx_tuple = idx
                key = tuple(
                    [
                        split_field + "=" + str(idx_tuple[split_field_index])
                        for split_field_index, split_field in enumerate(
                            splitting_fields
                        )
                    ]
                )
                result[key] = {
                    y_field: (
                        grouped_data_frame.loc[idx][x_field],
                        grouped_data_frame.loc[idx][y_field],
                    )
                    for y_field in y_field_list
                }
        else:
            result[tuple()] = {
                y_field: (data_frame[x_field], data_frame[y_field])
                for y_field in y_field_list
            }
        return result

    def aggregate_data(self, x_values: list, y_values: list, aggregation_methods: list):
        """
        Aggregates a list of y_values by their x_values.

        The two arguments `x_values` and `y_values` have to have the same length. Each pair
        of entries at the the same index are interpreted as one point, so the input
        x_values=[0,1], y_values=[3,4] is interpreted as the two points (0,3) and (1,4)

        Args:
            x_values: (list)
                list of x_values for points in a data set
            y_values: (list)
                list if y_values for points in a data set
            aggregation_methods: (list)
                list of arguments that are accepted by pandas data frames' groupby as aggregation methods
                They are either strings or functions that are defined at the beginning of this file

        Returns:
            (pd.DataFrame) A pd.DataFrame with x_values as indices of a groupby by the argument aggregation_methods
        """
        if len(x_values) != len(y_values):
            raise ValueError("data lists are of unequal length")
        df = pd.DataFrame({"x_field": x_values, "y_field": y_values})
        y_values_df = df[df["y_field"].notna()]
        na_values_df = df[df["x_field"].isna()]
        return (
            y_values_df.groupby("x_field").agg(aggregation_methods)["y_field"],
            na_values_df["y_field"],
        )

    def make_figure(
        self,
        plotname: str,
        x_field: str = "problem_size",
        y_field_list: list = ["total_cost"],
        split_fields: list = [],
        constraints: dict = {},
        aggregate_method=None,
        error_method=None,
        logscale_x: bool = False,
        logscale_y: bool = False,
        xlabel: str = None,
        ylabel: str = None,
        title: str = None,
        plottype: str = "line",
        regression: bool = True,
        **kwargs,
    ):
        """
        Args:
            plotname: (str)
                The name of the plot file to be saved without extension
            x_field: (str)
                The name of the x-Axis key
            y_field_list: (list)
                The list of name of y-axis keys
            split_fields: (list)
                A list with keys by which the extracted data should be grouped.
            constraints: (dict)
                A dictionary of values that the .json file has to have to be included in the plot.
                The key has to be identical to the key in the .json file. In combination with it a
                list with all acceptable values has to be given. If a json file doesn't have this key
                 or a listed value, it is ignored.
            aggregate_method: (str|callable)
                The method that the pandas.DataFrame uses to aggregate a list of y-values into a value to plot
            error_method: (callable)
                The function to be used to plot error bars.
            logscale_x: bool
                Flag for using logarithimc scale on the x-axis
            logscale_y: bool
                Flag for using logarithimc scale on the y-axis
            xlabel: (str)
                Label of the x-axis
            ylabel: (str)
                Label of the y-axis
            title: (str)
                Title of the plot
            plottype: (str)
                a string indicating which kind of plot to make
                 "line"
                 "scatterplot"
                 "histogramm"
                 "boxplot"
                 "cumulative"
                 "density"

            regression: (bool)
                Who knows that it does

        Returns:
            (None) saves a plot to the `plots` folder
        """
        fig, ax = plt.subplots()

        data = self.get_data_points(
            x_field=x_field,
            y_field_list=y_field_list,
            constraints=constraints,
            splitting_fields=split_fields,
        )

        # Each loop corresponds to one group in the groupby the
        # splitting fields
        for splitfield_key, data_dictionary in data.items():
            for y_field, y_field_values in data_dictionary.items():
                x_coordinate_list = y_field_values[0]
                y_coordinate_list = y_field_values[1]
                if splitfield_key:
                    label = str(splitfield_key) + "_" + y_field
                else:
                    label = y_field
                y_field_list_stripped = ", ".join(y_field_list)

                if plottype == "line":
                    if aggregate_method is None:
                        aggregate_method = "mean"
                    if error_method is None:
                        error_method = deviation_of_the_mean

                    y_values, na_values = self.aggregate_data(
                        x_coordinate_list,
                        y_coordinate_list,
                        [aggregate_method, error_method],
                    )
                    ax.errorbar(
                        y_values.index,
                        y_values.iloc[:, 0],
                        label=label,
                        yerr=y_values.iloc[:, 1],
                        **kwargs,
                    )

                    ax.axhline(np.mean(na_values))

                #                    print(y_field_values[1][y_field_values[0].isna()])
                #                    ax.hline()

                if plottype == "scatterplot":
                    if "s" not in kwargs:
                        kwargs["s"] = 7
                    ax.scatter(
                        x_coordinate_list, y_coordinate_list, **kwargs, label=label
                    )
                    # linear regression
                    if regression:
                        m, b = np.polyfit(x_coordinate_list, y_coordinate_list, 1)
                        ax.plot(
                            x_coordinate_list,
                            [m * z + b for z in x_coordinate_list],
                            color="red",
                            label=label,
                        )

                if plottype == "histogramm":
                    ax.hist(y_coordinate_list, label=label, **kwargs)
                    if xlabel is None:
                        xlabel = y_field_list_stripped
                    if ylabel is None:
                        ylabel = "count"

                if plottype == "boxplot":
                    ax.boxplot(y_coordinate_list)
                    if xlabel is None:
                        xlabel = " "

                if plottype == "cumulative":
                    sorted_data = np.sort(y_coordinate_list)
                    plt.step(sorted_data, np.arange(sorted_data.size), label=label)
                    if xlabel is None:
                        xlabel = y_field_list_stripped
                    if ylabel is None:
                        ylabel = "count"

                if plottype == "density":
                    sns.kdeplot(
                        list(y_coordinate_list),
                        label=label,
                        clip=(0.0, max(y_coordinate_list)),
                    )
                    if xlabel is None:
                        xlabel = y_field_list_stripped
                    if ylabel is None:
                        ylabel = "density"

        if xlabel is None:
            xlabel = x_field
        if ylabel is None:
            ylabel = y_field_list
        if title is None:
            title = plottype

        # adjusting additonal settings
        plt.legend()
        if logscale_x:
            ax.set_xscale("log")
        if logscale_y:
            ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.suptitle(title)
        savepath = self.savepath + sep + plotname + "." + self.fileformat
        fig.savefig(savepath)


class DataExtractor:
    """
    A class for reading, building and accessing optimization data in a pandas data frame
    """

    # a dictonary to look up which solver specific dictionaries are saved in json files
    # of that solver
    solver_keys = {
        "sqa": ["sqa_backend", "ising_interface"],
        "qpu": ["dwave_backend", "ising_interface", "cut_samples"],
        "qpu_read": ["dwave_backend", "ising_interface", "cut_samples"],
        "pypsa_glpk": ["pypsa_backend"],
        "classical": [],
    }
    # list of all keys for data entries, that every result file of a solver has
    general_fields = [
        "file_name",
        "problem_size",
        "scale",
        "timeout",
        "total_cost",
        "marginal_cost",
        "power_imbalance",
        "kirchhoff_cost",
        "optimization_time",
        "postprocessing_time",
    ]

    def __init__(self, prefix: str = "results", suffix: str = "sweep"):
        """
        Constructor for a DataExtractor. Don't call this directly but use a class method
        to initialize the DataFrame that contains the data.

        Data files specified in glob_dict are read, filtered by constraints and then saved
        into a pandas data frame. stored data can be queried by a getter for a given solver,
        the data is read from the folder f"{solver}_results_{suffix}"

        Args:
            prefix: (str)
                A prefix of the folder in which to search for json files
            prefix: (str)
                A suffix of the folder in which to search for json files
        """
        self.path_dictionary = {
            solver: "_".join([prefix, solver, suffix])
            for solver in self.solver_keys.keys()
        }
        self.df = None

    @classmethod
    def read_csv(cls, filename: str):
        """
        Returns a DataExtractor with the underlying data being provided by a csv file

        Args:
            filename: (str)
                The filename in the results folder to be passed to pandas read_csv method

        Returns:
            (DataExtractor)
                A setup DataExtractor with the data of the csv file
        """
        agent = DataExtractor()
        agent.df = pd.read_csv(filename)
        agent.df = agent.df.apply(pd.to_numeric, errors="ignore")
        if agent.df.empty:
            print("Data of optimization runs is empty!")
        return agent

    @classmethod
    def extract_from_json(cls, glob_dict: dict, constraints: dict = {}):
        """
        Returns a DataExtractor with the underlying data being provided json files in the
        result folders

        Args:
            glob_dict: (dict)
                a dictionary with solver names as keys and a list of glob expressions as values
                For a given `solver: glob_expr` pair, the directory `results_{solver}_sweep` is
                searched using the `glob_expr`
            constraints: (dict)
                a dictionary with data fields name as keys and admissable values in a list

        Returns:
            (DataExtractor)
                A setup DataExtractor with the data of the json files
        """
        agent = DataExtractor()
        agent.df = pd.DataFrame()
        for solver, glob_list in glob_dict.items():
            agent.df = agent.df._append(
                agent.extract_data(
                    solver,
                    glob_list,
                )
            )
        agent.df = agent.filter_by_constraint(constraints)
        agent.df = agent.df.apply(pd.to_numeric, errors="ignore")
        if agent.df.empty:
            print("Data of optimization runs is empty!")
        return agent

    def expand_solver_dict(self, dict_key: str, dict_value: str):
        """
        expand a dictionary containing solver dependent information into a list of dictionary
        to seperate information of seperate runs saved in the same file

        Args:
            dict_key: (str)
                dictionary key that was used to accessed solver depdendent data
            dict_value: (str)
                dictionary that contains solver specific data to be expanded

        Returns:
            (list) a list of dictionaries which are to be used in a cross product
        """
        # normal return value if no special case for expansion applies
        result = [dict_value]
        if dict_key == "cut_samples":
            result = [{"sample_id": key, **value} for key, value in dict_value.items()]
        return result

    def filter_by_constraint(self, constraints):
        """
        method to filter data frame by constraints. This class implements the filter rule as a
        dictionary with a key, value pair filtering the data frame for rows that have a non trivial entry
        in the column key by checking if that value is an element in value

        Args:
            constraints: (dict)
                a parameter that describes by which rules to filter the data frame.

        Returns:
            (pd.DataFrame) a data frame filtered accoring to the given constraints
        """
        result = self.df
        for key, value in constraints.items():
            result = result[result[key].isin(value) | result[key].isna()]
        return result

    def get_data(self, constraints: dict = None):
        """
        getter for accessing result data. The results are filtered by constraints.

        Args:
            constraints: (dict)
                a dictionary with data frame columns indices as key and lists of values. The data frame
                is filtered by rows that contain such a value in that columns

        Returns:
            (pd.DataFrame) a data frame of all relevant data filtered by constraints
        """
        return self.filter_by_constraint(constraints)

    def extract_data(
        self,
        solver,
        glob_list,
    ):
        """
        extracts all relevant data to be saved in a pandas DataFrame from the results of a given solver.
        The result files to be read are specified in a list of globs expressions

        Args:
            solver: (str)
                label of the solver for which to extract data
            glob_list: (list)
                list of strings, which are glob expressions to specifiy result files

        Returns:
            (pd.DataFrame) a pandas data frame containing all relevant data of the solver results
        """
        # plotData = collections.defaultdict(collections.defaultdict)
        result = pd.DataFrame()
        for glob_expr in glob_list:
            for file_name in glob.glob(
                path.join(self.path_dictionary[solver], glob_expr)
            ):
                result = result.append(
                    self.extract_data_from_file(
                        solver,
                        file_name,
                    )
                )
        return result

    def extract_data_from_file(
        self,
        solver,
        file_name,
    ):
        """
        reads the result file of a solver and builds a pandas data frame containing a entry for
        each run saved in the file

        Args:
            solver: (str)
                label of the solver for which to extract data
            file_name: (str)
                path of the file that contains solver results

        Returns:
            (pd.DataFrame) a pandas data frame containing a row for every run saved in the file
        """
        with open(file_name, encoding="utf-8") as file:
            file_data = json.load(file)

        result_dict = {}
        self.add_config(result_dict, file_data["config"])
        self.add_result(result_dict, file_data["results"])
        return pd.DataFrame(result_dict)

    def add_config(self, result_dict, config):
        result_dict["backend"] = config["backend"]
        result_dict.update(config["backend_config"])
        for subproblem, subproblem_config in config.get("ising_interface", {}).items():
            try:
                result_dict.update(
                    {
                        subproblem + "__" + key: [val]
                        for key, val in subproblem_config.items()
                    }
                )
            except AttributeError:
                continue

    def add_result(self, result_dict, result):
        result_dict.update({key: [val] for key, val in result.items()})
