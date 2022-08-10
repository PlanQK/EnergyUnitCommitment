import glob
import json
import collections

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from os import path, getenv, sep

import pandas as pd


def mean_of_square_root(values: list) -> float:
    """
    Reduction method to reduce the given values into one float by first applying 
    a square root to all values and then averaging them by using "mean"-method"

    @param values: list
        A list of values to be reduced.
    @return: float
        Reduced value.
    """
    return np.mean([np.sqrt(value) for value in values])


def mean_of_annealing_computing_cost(values: list) -> float:
    computing_cost_per_hour = getenv('costperhour', 1000)
    print(f"Using a cost of {computing_cost_per_hour} for an hour of quantum annealing")
    return np.mean([computing_cost_per_hour * value for value in values])


def deviation_of_the_mean(values: list) -> float:
    """
    Reduction method to reduce the given values into one float using the "deviation of the mean"-method

    @param values: list
        A list of values to be reduced.
    @return: float
        Reduced value.
    """
    return np.std(values) / np.sqrt(len(values))


def average_of_better_than_median(values: list) -> float:
    """
    Reduction method to reduce the given values into one float using the "average of better than median"-method

    @param values: list
        A list of values to be reduced.
    @return: float
        Reduced value.
    """
    median = np.median(values)
    result = 0
    count = 0
    for val in values:
        if val > median:
            continue
        count += 1
        result = + val
    return float(result) / count


def average_of_best(values: list) -> float:
    """
    Reduction method to reduce the given values into one float using the "average of best"-method

    @param values: list
        A list of values to be reduced.
    @return: float
        Reduced value.
    """
    values.sort()
    return np.mean(values[:-1])


def average_of_best_percent(values: list, percentage: float) -> float:
    """
    Reduction method to reduce the given values into one float using the "average of best percent"-method

    @param values: list
        A list of values to be reduced.
    @param percentage: float
        The percentage of values to be considered.
    @return: float
        Reduced value.
    """
    values.sort()
    return np.mean(values[:int(percentage * len(values))])


def cumulative_distribution(values: list) -> list:
    """
    Reduction method to construct a cumulative_distribution for a list of lists of values
    It returns a list of values that a histogramm of that list will look like the cumulative
    distribution of all values

    @param values: list
        A list of lists of values for which to construct a cumulative Distribution
    @param: list
        A list which histogramm is a cumulative distribution
    """
    result = []
    max_val = max(max(values, key=max)) + 1
    for value_lists in values:
        for val in value_lists:
            result += list(range(int(val), int(max_val), 1))
    return [result]


class PlottingAgent:
    """
    class for creating plots. on initialization reads and prepares data files. Plots based on that 
    data can then be created and written by calling make_figure
    """

    def __init__(self, fileformat="png"):
        """
        constructor for a plotting agent. On initialization, constructs a DataExtractionAgent to handle
        data management
        """
        self.savepath = "plots"
        self.fileformat = fileformat
        self.data_extraction_agent = None

    @classmethod
    def read_csv(cls, csv_file: str, fileformat="png"):
        """
        constructor method to set up the data to be plotted by reading a csv file
        
        Args:
            csv_file: (str) name and location of the file
            fileformat: (str) fileformat of the plots that an instance makes
        Returns:
            (PlottingAgent) a PlottingAgent initialized with the data in the csv file
        """
        agent = PlottingAgent(fileformat)
        agent.data_extraction_agent = DataExtractionAgent.read_csv(csv_file)
        return agent

    @classmethod
    def extract_from_json(cls, glob_dict, constraints={}, fileformat="png"):
        """
        constructor method to set up the data to be plotted by reading the json files
        of the runs that are to be plotted
        
        Args:
            glob_dict: (dict) a dictionary with solver names as keys and a list of glob expressions as value
            constraints: (dict) a dictionary with data fields name as keys and admissable values in a list
            fileformat: (str) fileformat of the plots that an instance makes
        Returns:
            (PlottingAgent) a PlottingAgent initialized with the data of the run files
        """
        agent = PlottingAgent(fileformat)
        agent.data_extraction_agent = DataExtractionAgent.extract_from_json(
            glob_dict=glob_dict, constraints=constraints,
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
        self.data_extraction_agent.df.to_csv(filename)

    def get_data_points(self,
                        x_field,
                        y_field_list,
                        splitting_fields,
                        constraints,
                        ):
        """
        returns all stored data points with x-value in x_field and y-values in y_field_list.
        The result is a dictonary with keys being the index of the groupby by the
        chosen splittingFields. The value is a dictionary over the chosen yFields with values being a pair or
        lists. The first one consists of x-values and the second entry consists of unsanitized y-values
        
        @param x_field: str
            label of the columns containing x-values
        @param y_field_list: str
            list of labels of the columns containing y-values
        @param splitting_fields: list
            list of labels by which columns to groupby the points
        @param constraints: dict
            rules for restricting which data to access from the stored data frame
        @return: dict
             a dictonary over groupby indices with values being dictionaries over y_field labels
        """
        data_frame = self.data_extraction_agent.get_data(constraints)[[x_field] + y_field_list + splitting_fields]
        result = {}
        if splitting_fields:
            grouped_data_frame = data_frame.groupby(splitting_fields, dropna=False).agg(list)
            for idx in grouped_data_frame.index:
                # if there is only 1 split_field, the index is not a multiindex
                if len(splitting_fields) == 1:
                    idx_tuple = [idx]
                else:
                    idx_tuple = idx
                key = tuple([
                    split_field + "=" + str(idx_tuple[counter])
                    for counter, split_field in enumerate(splitting_fields)
                ])
                result[key] = {
                    y_field: (grouped_data_frame.loc[idx][x_field], grouped_data_frame.loc[idx][y_field])
                    for y_field in y_field_list
                }
        else:
            result[tuple()] = {
                y_field: (data_frame[x_field], data_frame[y_field]) for y_field in y_field_list
            }
        return result

    def aggregate_data(self, x_values, y_values, aggregation_methods):
        """
        aggregation nethod for a list of y_values grouped by their corresponding xValue
        
        @param x_values: list
            list of xValues for points in a data set
        @param y_values: list
            list if y_values for points in a data set
        @param aggregation_methods: list
            list of arguments that are accepted by pandas data frames' groupby as aggregation methods
        @return: pd.DataFrame
            a pd.DataFrame with xValues as indices of a groupby by the argument aggregationMethods
        """
        if len(x_values) != len(y_values):
            raise ValueError("data lists are of unequal length")
        df = pd.DataFrame({"x_field": x_values, "y_field": y_values})
        y_values_df = df[df["y_field"].notna()]
        na_values_df = df[df["x_field"].isna()]
        return y_values_df.groupby("x_field").agg(aggregation_methods)["y_field"], \
            na_values_df["y_field"]

    def make_figure(self,
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
                    **kwargs
                    ):
        fig, ax = plt.subplots()

        data = self.get_data_points(
            x_field=x_field,
            y_field_list=y_field_list,
            constraints=constraints,
            splitting_fields=split_fields
        )

        for splitfieldKey, dataDictionary in data.items():
            for yField, y_field_values in dataDictionary.items():
                x_coordinate_list = y_field_values[0]
                y_coordinate_list = y_field_values[1]
                if splitfieldKey:
                    label = str(splitfieldKey) + "_" + yField
                else:
                    label = yField
                y_field_list_stripped = ", ".join(y_field_list)

                if plottype == "line":
                    if aggregate_method is None:
                        aggregate_method = 'mean'
                    if error_method is None:
                        error_method = deviation_of_the_mean

                    y_values, na_values = self.aggregate_data(
                        x_coordinate_list,
                        y_coordinate_list,
                        [aggregate_method, error_method],
                    )
                    ax.errorbar(y_values.index,
                                y_values.iloc[:, 0],
                                label=label,
                                yerr=y_values.iloc[:, 1],
                                **kwargs)

                    ax.axhline(np.mean(na_values))

                #                    print(y_field_values[1][y_field_values[0].isna()])
                #                    ax.hline()

                if plottype == "scatterplot":
                    if 's' not in kwargs:
                        kwargs['s'] = 7
                    ax.scatter(x_coordinate_list, y_coordinate_list, **kwargs, label=label)
                    # linear regression
                    if regression:
                        m, b = np.polyfit(x_coordinate_list, y_coordinate_list, 1)
                        ax.plot(x_coordinate_list, [m * z + b for z in x_coordinate_list], color='red', label=label)

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
                        clip=(0.0, max(y_coordinate_list))
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


class DataExtractionAgent:
    """
    class for reading, building and accessing optimization data in a pandas data frame
    """
    # a dictonary to look up which solver specific dictionaries are saved in json files
    # of that solver
    solver_keys = {
        "sqa": ["sqa_backend", "ising_interface"],
        "qpu": ["dwave_backend", "ising_interface", "cut_samples"],
        "qpu_read": ["dwave_backend", "ising_interface", "cut_samples"],
        "pypsa_glpk": ["pypsa_backend"],
        "classical": []
    }
    # list of all keys for data entries, that every result file of a solver has
    general_fields = [
        "file_name",
        "input_file",
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
        constructor for a data extraction. data files specified in glob_dict are read, filtered by
        constraints and then saved into a pandas data frame. stored data can be queried by a getter
        for a given solver, the data is read from the folder f"{solver}_results_{suffix}"

        @param suffix: str
            suffix or results folder in which to apply glob. 
        """
        self.path_dictionary = {
            solver: "_".join([prefix, solver, suffix]) for solver in self.solver_keys.keys()
        }
        self.df = None

    @classmethod
    def read_csv(cls, filename: str):
        agent = DataExtractionAgent()
        agent.df = pd.read_csv(filename)
        agent.df = agent.df.apply(pd.to_numeric, errors='ignore')
        return agent

    @classmethod
    def extract_from_json(cls, glob_dict: dict, constraints: dict = {}):
        agent = DataExtractionAgent()
        agent.df = pd.DataFrame()
        for solver, glob_list in glob_dict.items():
            agent.df = agent.df.append(agent.extract_data(solver, glob_list, ))
        agent.df = agent.filter_by_constraint(constraints)
        agent.df = agent.df.apply(pd.to_numeric, errors='ignore')
        return agent

    def expand_solver_dict(self, dict_key, dict_value):
        """
        expand a dictionary containing solver dependent information into a list of dictionary
        to seperate information of seperate runs saved in the same file
        
        @param dict_key: str
            dictionary key that was used to accessed solver depdendent data
        @param dict_value: str
            dictionary that contains solver specific data to be expanded
        @return: list
            a list of dictionaries which are to be used in a cross product
        """
        # normal return value if no special case for expansion applies
        result = [dict_value]
        if dict_key == "cut_samples":
            result = [
                {
                    "sample_id": key,
                    **value
                }
                for key, value in dict_value.items()
            ]
        return result

    def filter_by_constraint(self, constraints):
        """
        method to filter data frame by constraints. This class implements the filter rule as a
        dictionary with a key, value pair filtering the data frame for rows that have a non trivial entry
        in the column key by checking if that value is an element in value

        @param constraints: dict
            a parameter that describes by which rules to filter the data frame. 
        @return: pd.DataFrame
            a data frame filtered accoring to the given constraints
        """
        result = self.df
        for key, value in constraints.items():
            result = result[result[key].isin(value) | result[key].isna()]
        return result

    def get_data(self, constraints: dict = None):
        """
        getter for accessing result data. The results are filtered by constraints. 
        
        @param constraints: dict
            a dictionary with data frame columns indices as key and lists of values. The data frame 
            is filtered by rows that contain such a value in that columns
        
        @return: pd.DataFrame
            a data frame of all relevant data filtered by constraints
        """
        return self.filter_by_constraint(constraints)

    def extract_data(self, solver, glob_list, ):
        """
        extracts all relevant data to be saved in a pandas DataFrame from the results of a given solver. 
        The result files to be read are specified in a list of globs expressions
        
        @param solver: str
            label of the solver for which to extract data
        @param glob_list: list
            list of strings, which are glob expressions to specifiy result files
        @return: pd.DataFrame
            a pandas data frame containing all relevant data of the solver results
        """
        # plotData = collections.defaultdict(collections.defaultdict)
        result = pd.DataFrame()
        for glob_expr in glob_list:
            for file_name in glob.glob(path.join(self.path_dictionary[solver], glob_expr)):
                result = result.append(self.extract_data_from_file(solver, file_name, ))
        return result

    def extract_data_from_file(self, solver, file_name, ):
        """
        reads the result file of a solver and builds a pandas data frame containing a entry for
        each run saved in the file

        @param solver: str
            label of the solver for which to extract data
        @param file_name: str
            path of the file that contains solver results
        @return: pd.DataFrame
            a pandas data frame containing a row for every run saved in the file
        """
        with open(file_name) as file:
            file_data = json.load(file)
        # Data Frame containing a single row with columns as the data fields that every solver run
        # contains. Additional solver dependent information is merged on this data frame
        general_data = pd.DataFrame({"solver": solver,
                                     **{key: [file_data[key]] for key in self.general_fields}
                                     }
                                    )
        solver_dependent_data = [
            self.expand_solver_dict(solverDependentKey, file_data[solverDependentKey])
            for solverDependentKey in self.solver_keys[solver]
        ]
        for solver_dependent_field in solver_dependent_data:
            general_data = pd.merge(general_data, pd.DataFrame(solver_dependent_field), how="cross")
        return general_data


if __name__ == "__main__":
    main()

