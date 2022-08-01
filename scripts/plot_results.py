import glob
import json
import collections

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from os import path, getenv, sep

import pandas as pd

RESULT_SUFFIX = "sweep"
PRINT_NUM_READ_FILES = False
FILEFORMAT = "png"
BINSIZE = 1
PLOTLIMIT = 500
# export computing cost rate
COSTPERHOUR = getenv('costperhour')
# TODO remove
HACKCOUNTER = 0


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
    return np.mean([COSTPERHOUR * value for value in values])


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


def extract_information(file_regex: str, x_field: str, y_field: str,
                        split_fields: list = ["problem_size"], reduction_method=np.mean,
                        error_method=deviation_of_the_mean, constraints: dict = {}, embedding: bool = False) -> dict:
    """
    Extracts the information to be plotted.

    @param file_regex: str
        A glob pattern to get the data
    @param x_field: str
        The name of the x-Axis key. Nested keys can be reached by providing a list of strings, which represent the path
        through the nested dicts.
    @param y_field: str
        The name of the y-axis key. Nested keys can be reached by providing a list of strings, which represent the path
        through the nested dicts.
    @param split_fields: list
        A list with keys for which the extracted data should be grouped.
    @param reduction_method: func
        The function to be used to reduce values into one float value.
    @param error_method: func
        The function to be used to plot error bars.
    @param constraints: dict
        A dictionary of values that the .json file has to have to be included in the plot. The key has to be identical
        to the key in the .json file. In combination with it a list with all acceptable values has to be given. If a
        .json file doesn't have this key or a listed value, it is ignored.
    @param embedding: bool
        If true, embedded information from dWaveBackend is extracted to be plotted.
    @return: dict
        A dict with the information to be plotted.
    """
    files_read = 1
    plot_data = collections.defaultdict(collections.defaultdict)
    for file_name in glob.glob(file_regex):
        with open(file_name) as file:
            element = json.load(file)
            if embedding:
                element = add_embedding_information(json_dict=element)
            else:
                element = add_plottable_information(json_dict=element)

        # TODO indentation of for loop + cast error for ""
        for key, values in constraints.items():
            #            try:
            #                # value of constraint is not in constrains list
            if float(resolve_key(element, key)) not in values:
                break
        #            except KeyError:
        #                pass
        # value of constraint is found in constrains list
        else:
            # create a new key using the splitField
            key = tuple(
                e
                for e in [
                    splitField + "=" + str(resolve_key(element, splitField) or "")
                    for splitField in split_fields
                ]
            )

            xvalue = resolve_key(element, x_field)

            if xvalue not in plot_data[key]:
                plot_data[key][xvalue] = []

            yvalue = resolve_key(element, y_field)

            plot_data[key][xvalue].append(yvalue)
        files_read += 1

    # now perform reduction
    result = collections.defaultdict(list)
    for outer_key in plot_data:
        for inner_key in plot_data[outer_key]:
            if isinstance(inner_key, str):
                # H, T in sqa data
                if inner_key.startswith("["):
                    xvalue = inner_key[1:-1].split(',')
                else:
                    xvalue = inner_key
            else:
                xvalue = inner_key
            result[outer_key].append(
                [
                    # sometimes xvalue is still a string and has to be cast to a float
                    xvalue,
                    reduction_method(plot_data[outer_key][inner_key]),
                    error_method(plot_data[outer_key][inner_key]) if error_method is not None else 0
                ]
            )
        result[outer_key].sort()

    if PRINT_NUM_READ_FILES:
        print(f"files read for {file_regex} : {files_read}")

    return result


def extract_cut_samples(
        cut_samples_dict_list: list,
        x_key: str = "energy",
        y_key: str = "optimizedCost") \
        -> [list, list]:
    """
    Reduction method to extract data from all shots of a single batch. It creates two lists for making
    a scatter plot: First list consists of the spin glass's systems energy for the x-axis and second list is
    the corresponding optimized cost of the optimization problem. x_key is the dictionary key of the 
    metric to be extraced and used on the x-axis. y_key is the dictionary key of the 
    metric to be extraced and used on the y-axis.

    @param cut_samples_dict_list: object
        List of dicts to be searched through.
    @return: [list, list]
        A list with the extracted energy and one with the optimized costs.
    """
    x_axis_list = []
    y_axis_list = []
    for cut_samplesDict in cut_samples_dict_list:
        for value in cut_samplesDict.values():
            x_axis_list.append(value[x_key])
            y_axis_list.append(value[y_key])
    return x_axis_list, y_axis_list


def make_fig(plot_info: dict, output_file: str,
             logscale_x: bool = False, logscale_y: bool = False, xlabel: str = None, ylabel: str = None,
             title: str = None,
             fileformat: str = "pdf", plottype: str = "line", ) -> None:
    """
    Generates the plot and saves it to the specified location.

    @param plot_info: dict
        The data to be plotted.
    @param output_file: str
        The path to where the plot will be saved.
    @param logscale_x: bool
        Turns the x-axis into a log scale.
    @param logscale_y: bool
        Turns the y-axis into a log scale.
    @param xlabel: str
        The label of the x-axis.
    @param ylabel: str
        The label of the y-axis.
    @param title: str
        The title of the figure.
    @param fileformat: str
        The format of the saved file.
    @param plottype: str
        Indicates the type of plot to be created, e.g. scatter, line, etc..
    @return: None
    """
    fig, ax = plt.subplots()
    for key, values in plot_info.items():
        if plottype == "histogramm":
            # if condition is truthy if the function values have not been reduced earlier. thus in
            # 'yvalues' we have a list of values that go in to the histogramm with weight 1.
            # if the condition is falsy, the x_field should be the yvalues we want to plot, using arbitrary yvalues that
            # get reduced by len, thus counting how often a particular value appears in x_field
            if hasattr(values[0][1], "__getitem__"):
                for entries in values:
                    flattened_list = [item for sublist in entries[1] for item in sublist]
                    if not flattened_list:
                        continue
                    ax.hist(flattened_list, bins=[i * BINSIZE for i in range(int(min(flattened_list) / BINSIZE) - 2,
                                                                             int(max(flattened_list) / BINSIZE) + 2,
                                                                             1)],
                            label=key)
            else:
                sorted_values = sorted(values)
                ax.hist([e[0] for e in values], bins=[i * BINSIZE for i in range(0, PLOTLIMIT // BINSIZE + 1, 1)],
                        label=key, weights=[e[1] for e in sorted_values])

        if plottype == "scatterplot":
            # no use yet
            pass

        # in order to make a scatterplot of all shots for one file, we need to break
        # convention and put all relevant data as a dict where we would usually put
        # the y-value. We have to transform that into points to be plotted by giving
        # a reduction method, that can also extract the data from the dictionary
        if plottype == "scatterCutSample":
            x = []
            y = []
            for e in values:
                x += e[1][0]
                y += e[1][1]

            ax.scatter(x, y, s=8)

            # linear regression
            m, b = np.polyfit(x, y, 1)
            ax.plot(x, [m * z + b for z in x], color='red')

        if plottype == "boxplot":
            global HACKCOUNTER
            HACKCOUNTER += 1
            print("CALLING PLOT")
            sorted_values = sorted(values)
            ax.boxplot(
                [e[1] for e in sorted_values],
            )

        # default plot type of a function graph
        if plottype == "line":
            sorted_values = sorted(values)
            ax.errorbar([e[0] for e in sorted_values], [e[1] for e in sorted_values], label=key,
                        yerr=[e[2] for e in sorted_values])

    plt.legend()
    if logscale_x:
        ax.set_xscale("log")
    if logscale_y:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.suptitle(title)

    if fileformat == "":
        fig.savefig(output_file + ".png")
        fig.savefig(output_file + ".pdf")
    else:
        fig.savefig(output_file + "." + fileformat)


# TODO : probably clashes with default empty string initializes of backend specific variables
def resolve_key(element: dict, field: str) -> any:
    """
    Finds the given key within the given dictionary, even if the key is nested in it.

    @param element: dict
        The dict to be searched through.
    @param field: str
        The key to be searched for.
    @return: any
        The value of the specific key.
    """
    out = None
    # search for key in the given dict
    if field in element:
        out = element[field]
    # if key is not found, loop through all elements of the dict. If they are a dict themselves, enter and perform the
    # search again
    else:
        for key in element:
            if isinstance(element[key], dict):
                out = resolve_key(element[key], field)
                # if the key and therefore a value for it is found, break out of the loop and return the value
                if out is not None:
                    break

    return out


# TODO add embeddings to output data or rewrite how to extract embeddings from embedding dictionarys
def add_embedding_information(json_dict: dict) -> dict:
    """
    Add embedding information, specific to the dWave Backends, extracted from their backend specific results.

    @param json_dict: dict
        The dict where the embedding inforamtion is extracted from and added to.
    @return: dict
        The adjusted input dict.
    """
    embedding_dict = json_dict["info"]["embedding_context"]["embedding"]
    logical_qubits = [int(key) for key in embedding_dict.keys()]
    embedded_qubits = [item for sublist in embedding_dict.values() for item in sublist]

    json_dict["dwave_backend"]["embedded_qubits"] = len(embedded_qubits)
    json_dict["dwave_backend"]["logical_qubits"] = len(logical_qubits)
    json_dict["dwave_backend"]["embedFactor"] = float(json_dict["dwave_backend"]["embedded_qubits"]) \
        / float(json_dict["dwave_backend"]["logical_qubits"])

    return json_dict


def add_plottable_information(json_dict: dict) -> dict:
    """
    Add plottable information, specific to the dWave Backends, extracted from their backend specific results.

    @param json_dict: dict
        The dict where the plottable inforamtion is extracted from and added to.
    @return: dict
        The adjusted input dict.
    """
    if "cut_samples" in json_dict:
        json_dict["dwave_backend"]["sampleValues"] = [json_dict["cut_samples"][key]["optimizedCost"]
                                                      for key in json_dict["cut_samples"].keys()]

    # often, one of those two solutions is significantly better than the other
    if json_dict["dwave_backend"]["lowest_flow"] is not None:
        json_dict["dwave_backend"]["minChoice"] = min(json_dict["dwave_backend"]["lowest_flow"],
                                                      json_dict["dwave_backend"]["closest_flow"])

    return json_dict


# TODO odd bug of not working if solver str is empty
def plot_group(plotname: str, solver: str, file_regex_list: list, x_field: str, y_field_list: list = None,
               split_fields: list = ["problem_size"], logscale_x: bool = True, logscale_y: bool = False,
               path: list = None, reduction_method: list = None, line_names: list = None,
               embedding_data: bool = False, error_method=deviation_of_the_mean, constraints: dict = {},
               plottype: str = "line", xlabel: str = None, ylabel: str = None) -> None:
    """
    Extracts data from all files in the fileRegexList list and plots them into a single plot with the given plotname
    and set fileformat. The given files have to be .JSON files.

    @param plotname: str
        The name of the generated plot, without the filetype.
    @param solver: str
        A string to indicate which solver was used. This sets some helpful default values
    @param file_regex_list: list
        A list of all regular expressions to get data. Each item in the list is plotted as it's own line (possibly
        further split up, if split_fields is not empty). A whitespace in a regex string splits multiple regular
        expressions to be plotted into a single line.
    @param x_field: str
        The name of the x-axis key. Nested keys can be reached by providing a list of strings, which represent the path
        through the nested dicts.
    @param y_field_list: list
        The list of names of the y-axis keys. It has to be the same size as fileRegexList. The i-th entry in this list
        is for the i-th entry in the fileRegexList. Nested keys can be reached by providing a list of strings, which
        represent the path through the nested dicts.
    @param split_fields: list
        A list with keys for which the extracted data should be grouped.
    @param logscale_x: bool
        Turns the x-axis into a log scale.
    @param logscale_y: bool
        Turns the y-axis into a log scale.
    @param path: list
        A list of the paths to the data. It has to be the same size as fileRegexList. If nothing is given the data is
        searched for in the standard folder f'results_{solver}_sweep'.
    @param reduction_method: func
        A list of functions to be used to reduce values into one float value. It has to be the same size as
        fileRegexList.
    @param line_names: list
        A list with the labels for the lines to be plotted. It has to be the same size as fileRegexList. If nothing is
        given, the keys of y_field_list will be used as labels.
    @param embedding_data: bool
        If true, embedded information from dWaveBackend is extracted to be plotted.
    @param error_method: func
        The function to be used to plot error bars.
    @param constraints: dict
        A dictionary of values that the .json file has to have to be included in the plot. The key has to be identical
        to the key in the .json file. In combination with it a list with all acceptable values has to be given. If a
        .json file doesn't have this key or a listed value, it is ignored.
    @param plottype: str
        Indicates the type of plot to be created, e.g. scatter, line, etc..
    @param xlabel: str
        The label of the x-axis.
    @param ylabel: str
        The label of the y-axis.
    @return: None
    """
    if y_field_list is None:
        y_field_list = ["total_cost"] * len(file_regex_list)

    if len(file_regex_list) != len(y_field_list):
        print("number of regex doesn't match number of y_field's selected")
        return

    if path is None:
        path = [f"results_{solver}_{RESULT_SUFFIX}"] * len(file_regex_list)

    if reduction_method is None:
        reduction_method = [np.mean] * len(file_regex_list)

    if line_names is None:
        line_names = y_field_list

    if xlabel is None:
        xlabel = x_field

    if ylabel is None:
        ylabel = y_field_list

    plot_info = {}
    for idx in range(len(file_regex_list)):
        for regex in file_regex_list[idx].split():
            iterator = extract_information(file_regex=f"{path[idx]}/{regex}",
                                           x_field=x_field,
                                           y_field=y_field_list[idx],
                                           split_fields=split_fields,
                                           reduction_method=reduction_method[idx],
                                           error_method=error_method,
                                           constraints=constraints,
                                           embedding=embedding_data).items()

            for key, value in iterator:
                plot_info_key = f"{solver}_{key}_{line_names[idx]}"
                if plot_info_key in plot_info:
                    plot_info[plot_info_key] += value
                else:
                    plot_info[plot_info_key] = value
    make_fig(
        plot_info,
        f"plots/{plotname}",
        fileformat=FILEFORMAT,
        logscale_x=logscale_x,
        logscale_y=logscale_y,
        xlabel=xlabel,
        ylabel=ylabel,
        plottype=plottype
        #        title=''
    )


def main():
    plt.style.use("seaborn")

    global BINSIZE
    BINSIZE = 1

    #    regex = "*Nocost*5input_1[5]_[0]_20*fullsplit*"
    regex = "*1[0-4]_[0-9]_20.nc_100_200_full*60_200_1"
    # DataAgent = DataExtractionAgent({"qpu_read" : [regex]} , "sweep")

    networkbatch = "infoNocost_220124cost5input_"

    plot_agent = PlottingAgent({
        "sqa": [networkbatch + "*[36592]0_?_20.nc*"],
        "pypsa_glpk": [networkbatch + "*[36592]0_?_20.nc*"],
    },
        constraints={
            "problem_size": [30, 60, 90, 120, 150],
            "trotterSlices": [500],
        })
    plot_agent.make_figure("large_problem_sqa_glpk",
                          split_fields=["solver"],
                          )
    plot_agent.make_figure("large_problem_time_sqa_glpk",
                          y_field_list=["optimization_time"],
                          split_fields=["solver"],
                          )
    plot_agent.make_figure("large_problem_time_and_cost_sqa_glpk",
                          y_field_list=["optimization_time", "total_cost"],
                          split_fields=["solver"],
                          )

    return

    plot_agent = PlottingAgent({
        "qpu": [networkbatch + "1[0-9]_[0-9]_20.nc_*_200_fullsplit_60_1",
                networkbatch + "1[5-9]_1[0-9]_20.nc_*_200_fullsplit_60_1"
                ],
        #                    "pypsa_glpk" : [networkbatch + "1[0-9]_[0-9]_20.nc_30_?"],
    },
        constraints={}
    )

    constraints = {"annealing_time": [100], }
    plot_agent.make_figure("annealing_time_to_cost_len",
                          x_field="annealing_time",
                          y_field_list=["total_cost", ],
                          constraints={"sample_id": [0]},
                          aggregate_method=len,
                          error_method=lambda x: 0,
                          )
    plot_agent.make_figure("problemsize_to_cost_mean",
                          x_field="problem_size",
                          y_field_list=["LowestEnergy", "lowest_flow", "closest_flow"],
                          constraints={"sample_id": [1], **constraints}
                          )
    plot_agent.make_figure("problemsize_to_optimized_quantum_cost_mean",
                          x_field="problem_size",
                          y_field_list=["quantumCost", "optimizedCost"],
                          constraints=constraints
                          )
    plot_agent.make_figure("problemsize_to_optimized_quantum_cost_mean_constraint_100",
                          x_field="problem_size",
                          y_field_list=["quantumCost", "optimizedCost"],
                          constraints={"quantumCost": list(range(100)), **constraints}
                          )
    plot_agent.make_figure("problemsize_to_maxcut_totalcost_mean",
                          x_field="problem_size",
                          y_field_list=["LowestEnergy", "cut_samples_cost"],
                          constraints={"sample_id": [1], **constraints}
                          )
    plot_agent.make_figure("annealing_time_to_cost_mean",
                          x_field="annealing_time",
                          y_field_list=["total_cost", "lowest_flow", "closest_flow", "cut_samples_cost"],
                          constraints={"problem_size": [10, 11, 12, 13], "sample_id": [0], },
                          )
    plot_agent.make_figure("annealing_time_to_cost_len",
                          x_field="annealing_time",
                          y_field_list=["total_cost", ],
                          constraints={"problem_size": [10, 11, 12, 13], "sample_id": [0], },
                          aggregate_method=len,
                          error_method=lambda x: 0,
                          )
    #    networklist = ["".join(["Nocost_220124cost5input_15_",str(i),"_20.nc"]) for i in [0,1,2]]
    #    for i in [0,1,2]:
    #        plot_agent.make_figure(f"scatter_quantumcost_to_optimizedcost_{i}",
    #                x_field="quantumCost",
    #                y_field_list=["optimizedCost"],
    #                constraints={"input_file" : [networklist],
    #                        "annealing_time" : [100]},
    #                plottype="scatterplot"
    #                )

    networkbatch = "infoNocost_220124cost5input_"
    plot_agent = PlottingAgent({
        "sqa": [networkbatch + "1[3-5]_[0-9]_20.nc_0.1_8.0_*_fullsplit_1",
                ],
    },
        constraints={
            "trotterSlices": list(range(10, 110, 10)),
            "optimizationCycles": list(range(4, 10, 2)) + list(range(10, 100, 5))
        }
    )
    plot_agent.make_figure("optimizationCycles_to_cost_mean",
                          x_field="optimizationCycles",
                          y_field_list=["total_cost"],
                          )
    plot_agent.make_figure("optimizationCycles_to_cost_mean_split_trotter",
                          x_field="optimizationCycles",
                          y_field_list=["total_cost"],
                          split_fields=["trotterSlices"],
                          constraints={"trotterSlices": [10, 20, 40, 80, 100]}
                          )
    plot_agent.make_figure("trotter_to_cost_mean",
                          x_field="trotterSlices",
                          y_field_list=["total_cost"],
                          )
    plot_agent.make_figure("trotter_to_cost_mean_split_optimizationCycles",
                          x_field="trotterSlices",
                          y_field_list=["total_cost"],
                          split_fields=["optimizationCycles"],
                          constraints={"optimizationCycles": [4, 10, 30, 70]}
                          )
    plot_agent.make_figure(f"scatter_trotter_to_cost",
                          x_field="trotterSlices",
                          y_field_list=["total_cost"],
                          split_fields=["optimizationCycles"],
                          constraints={"optimizationCycles": [4, 30, 70]},
                          plottype="scatterplot",
                          s=12,
                          regression=False,
                          )
    plot_agent.make_figure(f"scatter_optimizationCycles_to_cost",
                          x_field="optimizationCycles",
                          y_field_list=["total_cost"],
                          split_fields=["trotterSlices"],
                          constraints={"optimizationCycles": [4, 30, 70]},
                          plottype="scatterplot",
                          s=12,
                          regression=False,
                          )

    plot_agent = PlottingAgent.extract_from_json({
        "qpu_read": [networkbatch + "1[0-9]_[0-9]_20.nc_100_200_fullsplit_60_100_1"],
    })

    #    plot_agent = PlottingAgent({"qpu_read" : [regex]})

    plot_agent.make_figure("testplot",
                          x_field="quantumCost",
                          y_field_list=["optimizedCost"],
                          split_fields=[],
                          aggregate_method='mean'
                          )
    plot_agent.make_figure("testscatter",
                          x_field="quantumCost",
                          y_field_list=["optimizedCost"],
                          constraints={"problem_size": [14]},
                          plottype="scatterplot",
                          s=5,
                          )
    plot_agent.make_figure("testplotsize",
                          x_field="problem_size",
                          y_field_list=["LowestEnergy"],
                          constraints={"sample_id": [1]},
                          split_fields=[],
                          aggregate_method='mean',
                          error_method=deviation_of_the_mean,
                          )
    plot_agent.make_figure("testhistogramm",
                          y_field_list=["quantumCost", "optimizedCost"],
                          plottype="histogramm",
                          bins=100,
                          )
    plot_agent.make_figure("testcumulative",
                          y_field_list=["quantumCost", "optimizedCost"],
                          plottype="cumulative",
                          bins=100,
                          )
    plot_agent.make_figure("testdensity",
                          y_field_list=["quantumCost", "optimizedCost"],
                          plottype="density",
                          #            constraints={"problem_size" : [14]}
                          )
    plot_agent.make_figure("testboxplot",
                          y_field_list=["quantumCost"],
                          plottype="boxplot",
                          )

    return

    regex = "*1[0-4]_[0-4]_20.nc_100_200_full*60_200_1"
    #    plot_group(f"scatterplot_nocostinput_100_Anneal_70_chain_strength",
    #            "qpu_read",
    #            [
    #            regex,
    #            ],
    #            "problem_size",
    #            y_field_list = ["cut_samples"],
    #            reduction_method = [extract_cut_samples],
    #            error_method = None,
    #            split_fields = ["annealing_time"],
    #            plottype="scatterCutSample",
    #            logscale_x=False,
    #            xlabel="energy",
    #            ylabel="cost",
    #    )
    plot_group(f"problemsize_to_cost_nocostinput_100_Anneal_70_chain_strength",
               "qpu_read",
               [
                   regex,
               ],
               "problem_size",
               y_field_list=["total_cost"],
               reduction_method=[np.mean],
               error_method=None,
               split_fields=["annealing_time"],
               plottype="line",
               logscale_x=False,
               xlabel="energy",
               ylabel="cost",
               )

    return

    constraints = {
        'annealing_time': [100],
        'num_reads': [200],
        'problem_size': [10, 11, 12, 13, 14],
        'chain_strength': [60],
    }
    plot_group("scale_to_cost_median",
               "qpu",
               [
                   regex,
                   regex,
                   regex,
                   regex,
               ],
               "scale",
               y_field_list=["total_cost", "lowest_flow", "closest_flow", "cut_samples_cost"],
               split_fields=[],
               logscale_x=False,
               reduction_method=[len] * 4,
               error_method=deviation_of_the_mean,
               constraints=constraints,
               )
    plot_group("scale_to_cost_mean",
               "qpu",
               [
                   regex,
                   regex,
                   regex,
                   regex,
               ],
               "scale",
               y_field_list=["total_cost", "lowest_flow", "closest_flow", "cut_samples_cost"],
               split_fields=[],
               logscale_x=False,
               reduction_method=[np.mean] * 4,
               error_method=deviation_of_the_mean,
               constraints=constraints,
               )

    return

    constraints = {
        'annealing_time': [100],
        'num_reads': [200],
        'problem_size': [10, 11, 12, 13, 14]
    }
    plot_group("chain_strength_to_cost_mean",
               "qpu",
               [
                   regex,
                   regex,
                   regex,
                   regex,
               ],
               "chain_strength",
               y_field_list=["total_cost", "lowest_flow", "closest_flow", "cut_samples_cost"],
               split_fields=[],
               logscale_x=False,
               reduction_method=[np.mean] * 4,
               error_method=deviation_of_the_mean,
               constraints=constraints,
               )
    plot_group("chain_strength_to_cost_median",
               "qpu",
               [
                   regex,
                   regex,
                   regex,
                   regex,
               ],
               "chain_strength",
               y_field_list=["total_cost", "lowest_flow", "closest_flow", "cut_samples_cost"],
               split_fields=[],
               logscale_x=False,
               reduction_method=[len] * 4,
               error_method=deviation_of_the_mean,
               constraints=constraints,
               )

    constraints = {
        'annealing_time': [100],
        'num_reads': [200],
        'problem_size': list(range(10, 45, 1)),
    }
    plot_group("size_to_cost_mean",
               "qpu",
               [
                   regex,
                   regex,
                   regex,
                   regex,
               ],
               "problem_size",
               y_field_list=["total_cost", "lowest_flow", "closest_flow", "cut_samples_cost"],
               split_fields=[],
               logscale_x=False,
               reduction_method=[np.mean] * 4,
               error_method=deviation_of_the_mean,
               constraints=constraints,
               )
    plot_group("size_to_cost_median",
               "qpu",
               [
                   regex,
                   regex,
                   regex,
                   regex,
               ],
               "problem_size",
               y_field_list=["total_cost", "lowest_flow", "closest_flow", "cut_samples_cost"],
               split_fields=[],
               logscale_x=False,
               reduction_method=[len] * 4,
               error_method=deviation_of_the_mean,
               constraints=constraints,
               )

    constraints = {
        'annealing_time': [10, 50, 100, 200, 300, 500, 1000, 1500, 2000],
        'num_reads': [200],
        'chain_strength': [60],
        'problem_size': [10, 11, 12, 13, 14]
    }
    plot_group("annealTime_to_cost_mean",
               "qpu",
               [
                   regex,
                   regex,
                   regex,
                   regex,
               ],
               "annealing_time",
               y_field_list=["total_cost", "lowest_flow", "closest_flow", "cut_samples_cost"],
               split_fields=[],
               logscale_x=False,
               reduction_method=[np.mean] * 4,
               error_method=deviation_of_the_mean,
               constraints=constraints,
               )
    plot_group("annealTime_to_cost_median",
               "qpu",
               [
                   regex,
                   regex,
                   regex,
                   regex,
               ],
               "annealing_time",
               y_field_list=["total_cost", "lowest_flow", "closest_flow", "cut_samples_cost"],
               split_fields=[],
               logscale_x=False,
               reduction_method=[len] * 4,
               error_method=deviation_of_the_mean,
               constraints=constraints,
               )

    return

    regex = "*15_0_20.nc_0.1_8.0_*_fullsplit*"
    plot_group("trotterslices_to_cost",
               "sqa",
               [
                   regex
               ],
               "optimizationCycles",
               y_field_list=["total_cost"],
               split_fields=["trotterSlices"],
               reduction_method=[np.mean],
               logscale_x=False,
               constraints={
                   "optimizationCycles": [4, 6, 8, 10]
               }
               )
    return

    constraints = {"problem_size": [12]}
    regex = "*Nocost*input_??_*nc*10*_fullsplit*"

    plot_group("trotterSlices_to_cost_boxplot",
               "sqa",
               [
                   regex,
               ],
               "trotterSlices",
               y_field_list=["total_cost"],
               split_fields=[],
               logscale_x=False,
               reduction_method=[lambda x: x],
               constraints=constraints,
               plottype="boxplot",
               )
    return

    plot_group("trotterSlices_to_cost_mean_split_optCycles",
               "sqa",
               [
                   regex,
               ],
               "trotterSlices",
               y_field_list=["total_cost"],
               split_fields=["optimizationCycles"],
               logscale_x=False,
               reduction_method=[np.mean],
               error_method=deviation_of_the_mean,
               constraints=constraints,
               )

    plot_group("optimizationCycles_to_total_cost_mean",
               "sqa",
               [
                   regex,
               ],
               "optimizationCycles",
               y_field_list=["total_cost", ],
               split_fields=["trotterSlices"],
               logscale_x=False,
               reduction_method=[np.mean],
               error_method=deviation_of_the_mean,
               constraints=constraints,
               )

    return

    regex = "*15_0_20.nc_0.1_8.0_*_fullsplit*"
    plot_group("trotterslices_to_cost",
               "sqa",
               [
                   regex
               ],
               "optimizationCycles",
               y_field_list=["total_cost"],
               split_fields=["trotterSlices"],
               reduction_method=[np.mean],
               logscale_x=False,
               constraints={
                   "optimizationCycles": [4, 6, 8, 10, 12, 14, 16, 20, 40, 100]
               }
               )

    return

    regex = "*15_0_20.nc_100_200_full*70_200_1"
    plot_group(f"scatterplot_nocostinput_100_Anneal_70_chain_strength",
               "qpu_read",
               [
                   regex,
               ],
               "problem_size",
               y_field_list=["cut_samples"],
               reduction_method=[extract_cut_samples],
               error_method=None,
               split_fields=["annealing_time"],
               plottype="scatterCutSample",
               logscale_x=False,
               xlabel="energy",
               ylabel="cost",
               )

    return

    plot_group("newising_chain_strength_to_cut_samples_cost",
               "qpu_read",
               [
                   "*newising_20_[0]_20.nc_[1]00_365*"
               ],
               "chain_strength",
               y_field_list=["cut_samples_cost"],
               split_fields=["annealing_time"],
               logscale_x=False,
               )

    for chain in [50, 70, 90]:
        regex = f"*newising_20_[0]_20.nc_[1]00_365_30_0_1_{chain}_365_1"
        plot_group(f"cumulativeCostDistribution_for_100_Anneal_{chain}_chain_strength",
                   "qpu_read",
                   [
                       regex,
                   ],
                   "problem_size",
                   y_field_list=["sampleValues"],
                   reduction_method=[cumulative_distribution],
                   error_method=None,
                   split_fields=["annealing_time"],
                   plottype="histogramm",
                   logscale_x=False,
                   xlabel="energy",
                   ylabel="cost",
                   )
        plot_group(f"scatterplot_new_ising_100_Anneal_{chain}_chain_strength",
                   "qpu_read",
                   [
                       regex,
                   ],
                   "problem_size",
                   y_field_list=["cut_samples"],
                   reduction_method=[extract_cut_samples],
                   error_method=None,
                   split_fields=["annealing_time"],
                   plottype="scatterCutSample",
                   logscale_x=False,
                   xlabel="energy",
                   ylabel="cost",
                   )
    return

    plot_group(f"scatterplot_new_ising_365Anneal",
               "qpu_read",
               [
                   f"*newising_10_[0]_20.nc_110_365_30_0_1_80_365_1",
               ],
               "problem_size",
               y_field_list=["cut_samples"],
               reduction_method=[extract_cut_samples],
               error_method=None,
               split_fields=["annealing_time"],
               plottype="scatterCutSample",
               logscale_x=False,
               xlabel="energy",
               ylabel="cost",
               )
    return

    # TODO add embeddings for other scales for first plot

    regex = 'embedding_rep_0_ord_1_nocostnewising*.nc.json'
    plot_group("embedding_size_to_embedded_qubits_newising",
               "qpu",
               [
                   regex,
               ],
               x_field="problem_size",
               y_field_list=["embedded_qubits"],
               split_fields=["scale"],
               path=["networks"],
               embedding_data=True,
               logscale_x=False,
               logscale_y=False,
               )
    plot_group("embedding_size_to_logical_qubits_newising",
               "qpu",
               [
                   regex,
               ],
               x_field="problem_size",
               y_field_list=["logical_qubits"],
               split_fields=[],
               path=["networks"],
               embedding_data=True,
               logscale_x=False,
               logscale_y=False,
               )
    plot_group("embedding_scale_to_embedFactor_newising",
               "qpu",
               [
                   regex,
               ],
               x_field="problem_size",
               y_field_list=["embedFactor"],
               split_fields=[],
               path=["networks"],
               embedding_data=True,
               logscale_x=False,
               logscale_y=False,
               )

    regex = "*input_[7-9]_*_20.nc_*_70_0_[01]_250_1"
    constraints = {'mangled_total_anneal_time': [19, 20],
                   'chain_strength': [250],
                   'slack_var_factor': [70.0],
                   'maxOrder': [0, 1],
                   }
    plot_group("annealReadRatio_to_cost_mean",
               "qpu",
               [
                   regex,
                   regex,
                   regex,
               ],
               "annealReadRatio",
               y_field_list=["total_cost", "lowest_flow", "closest_flow"],
               split_fields=[],
               logscale_x=True,
               reduction_method=[mean_of_square_root] * 3,
               constraints=constraints,
               )
    plot_group("annealTime_to_cost_mean",
               "qpu",
               [
                   regex,
                   regex,
                   regex,
               ],
               "annealing_time",
               y_field_list=["total_cost", "lowest_flow", "closest_flow"],
               reduction_method=[mean_of_square_root] * 3,
               logscale_x=True,
               split_fields=[],
               constraints=constraints,
               )

    return

    plot_group(f"scatterplot_energy_to_optimizedCost_for_anneal_split",
               "qpu_read",
               [
                   f"*put_15_[0]_20.nc_5_365_30_0_1_80_365_1",
                   f"*put_15_[0]_20.nc_1000_365_30_0_1_80_365_1",
                   f"*put_15_[0]_20.nc_2000_365_30_0_1_80_365_1"
               ],
               "problem_size",
               y_field_list=["cut_samples"] * 3,
               reduction_method=[extract_cut_samples] * 3,
               error_method=None,
               split_fields=["annealing_time"],
               plottype="scatterCutSample",
               logscale_x=False,
               xlabel="energy",
               ylabel="cost",
               )
    return

    for annealTime in [1, 5, 110, 1000, 2000]:
        regex = f"*put_15_[0]_20.nc_{annealTime}_365_30_0_1_80_365_1"
        plot_group(f"scatterplot_energy_to_optimizedCost_for_anneal_{annealTime}",
                   "qpu_read",
                   [
                       regex,
                   ],
                   "problem_size",
                   y_field_list=["cut_samples"],
                   reduction_method=[extract_cut_samples],
                   error_method=None,
                   split_fields=[],
                   plottype="scatterCutSample",
                   logscale_x=False,
                   xlabel="energy",
                   ylabel="cost",
                   )

    regex = '*put_15_[0]_20.nc_110_365_30_0_1_80_365_1'
    plot_group("cumulativeCostDistribution_for_fullInitialEnergies",
               "qpu_read",
               [
                   regex,
               ],
               "problem_size",
               y_field_list=["cut_samples"],
               reduction_method=[extract_cut_samples],
               error_method=None,
               split_fields=[],
               plottype="scatterCutSample",
               logscale_x=False,
               xlabel="energy",
               ylabel="cost",
               )

    regex = '*nocostinput_*'
    plot_group("afterChange_cumulativeCostDistribution_for_fullInitialEnergies",
               "qpu",
               [
                   regex,
               ],
               "problem_size",
               y_field_list=["cut_samples"],
               reduction_method=[extract_cut_samples],
               error_method=None,
               split_fields=[],
               plottype="scatterCutSample",
               logscale_x=False,
               xlabel="energy",
               ylabel="cost",
               )

    plot_group(plotname="afterChange_glpk_scale_to_cost_mean",
               solver="pypsa_glpk",
               file_regex_list=[
                   '*nocostinput_*1',
                   '*nocostinput_*1',
               ],
               x_field="scale",
               y_field_list=["total_cost"] * 2,
               split_fields=[],
               reduction_method=[np.mean, np.std],
               constraints={
                   'problem_size': [10, 11, 12, 13, 14],
                   'scale': list(range(10, 45, 5)),
               },
               line_names=["total_cost", "standard_deviation"],
               logscale_x=False,
               logscale_y=False,
               )

    return

    plot_group("costDistribution_for_fullSampleOpt",
               "qpu_read",
               [
                   regex,
               ],
               "problem_size",
               y_field_list=["sampleValues", ],
               reduction_method=[lambda x: x],
               split_fields=[],
               plottype="histogramm",
               logscale_x=False,
               ylabel="count",
               xlabel="sampleValues",
               )
    plot_group("cumulativeCostDistribution_for_fullSampleOpt",
               "qpu_read",
               [
                   regex,
               ],
               "problem_size",
               y_field_list=["sampleValues", ],
               reduction_method=[cumulative_distribution],
               split_fields=[],
               plottype="histogramm",
               logscale_x=False,
               ylabel="count",
               xlabel="sampleValues",
               )
    plot_group("cumulativeCostDistribution_for_fullInitialEnergies",
               "qpu_read",
               [
                   regex,
               ],
               "problem_size",
               y_field_list=[["serial", "vectors", "energy", "data"]],
               reduction_method=[cumulative_distribution],
               split_fields=[],
               plottype="histogramm",
               logscale_x=False,
               ylabel="count",
               xlabel="energy",
               )

    plot_group("glpk_scale_to_cost_mean",
               "pypsa_glpk",
               [
                   '*nocostinput_*1',
                   '*nocostinput_*1',
               ],
               x_field="scale",
               y_field_list=["total_cost"] * 2,
               split_fields=[],
               reduction_method=[np.mean, np.std],
               constraints={
                   'problem_size': [10, 11, 12, 13, 14],
                   'scale': list(range(10, 45, 5)),
               },
               line_names=["total_cost", "standard_deviation"],
               logscale_x=False,
               logscale_y=False,
               )
    plot_group("glpk_scale_to_cost_split_size_10_11_12_mean",
               "pypsa_glpk",
               [
                   '*nocostinput_*1',
               ],
               x_field="scale",
               y_field_list=["total_cost"],
               split_fields=["problem_size"],
               constraints={
                   'problem_size': [10, 11, 12],
                   'scale': list(range(10, 45, 5)),
               },
               logscale_x=False,
               logscale_y=False,
               )
    plot_group("glpk_scale_to_cost_split_size_12_13_14_mean",
               "pypsa_glpk",
               [
                   '*nocostinput_*1',
               ],
               x_field="scale",
               y_field_list=["total_cost"],
               split_fields=["problem_size"],
               constraints={
                   'problem_size': [12, 13, 14],
                   'scale': list(range(10, 45, 5)),
               },
               logscale_x=False,
               logscale_y=False,
               )

    # TODO add embeddings for other scales for first plot
    plot_group("embedding_size_to_embedded_qubits",
               "qpu",
               [
                   'embedding_rep_0_ord_1_nocostinput*.nc.json'
               ],
               x_field="problem_size",
               y_field_list=["embedded_qubits"],
               split_fields=["scale"],
               path=["networks"],
               embedding_data=True,
               logscale_x=False,
               logscale_y=False,
               )
    plot_group("embedding_size_to_logical_qubits",
               "qpu",
               [
                   'embedding_rep_0_ord_1_nocostinput*20.nc.json'
               ],
               x_field="problem_size",
               y_field_list=["logical_qubits"],
               split_fields=[],
               path=["networks"],
               embedding_data=True,
               logscale_x=False,
               logscale_y=False,
               )
    plot_group("embedding_scale_to_embedFactor",
               "qpu",
               [
                   'embedding_rep*'
               ],
               x_field="scale",
               y_field_list=["embedFactor"],
               split_fields=[],
               path=["networks"],
               embedding_data=True,
               logscale_x=False,
               logscale_y=False,
               )
    plot_group("embedding_size_to_embedFactor",
               "qpu",
               [
                   'embedding_rep_0_ord_1_nocostinput*20.nc.json'
               ],
               x_field="problem_size",
               y_field_list=["embedFactor"],
               split_fields=[],
               path=["networks"],
               embedding_data=True,
               logscale_x=False,
               logscale_y=False,
               )

    qpu_regex = "*110_365_30_0_1_80_1"
    glpk_regex = "*"
    for strategy in ["total_cost", "lowest_flow", "closest_flow"]:
        plot_group(f"qpu_size_to_{strategy}_even_scale_mean",
                   "f",
                   [
                       qpu_regex,
                   ],
                   "problem_size",
                   split_fields=["scale"],
                   logscale_x=False,
                   y_field_list=[strategy],
                   path=["results_qpu_sweep"],
                   line_names=["qpu"],
                   constraints={
                       'problem_size': [10, 11, 12, 13, 14],
                       'scale': [10, 20, 30, 40]
                   },
                   reduction_method=[np.mean]
                   )
        plot_group(f"qpu_size_to_{strategy}_odd_scale_mean",
                   "f",
                   [
                       qpu_regex,
                   ],
                   "problem_size",
                   split_fields=["scale"],
                   logscale_x=False,
                   y_field_list=[strategy],
                   path=["results_qpu_sweep"],
                   line_names=["qpu"],
                   constraints={
                       'problem_size': [10, 11, 12, 13, 14],
                       'scale': [15, 25, 35, 45]
                   },
                   reduction_method=[np.mean]
                   )
    plot_group(f"qpu_glpk_scale_to_cost_mean",
               "qpu",
               [
                   qpu_regex,
                   qpu_regex,
                   qpu_regex,
                   glpk_regex,
               ],
               "scale",
               y_field_list=["total_cost", "lowest_flow", "closest_flow", "total_cost", ],
               logscale_x=False,
               split_fields=[],
               constraints={
                   'problem_size': [10, 11, 12, 13, 14],
               },
               path=[
                   "results_qpu_sweep",
                   "results_qpu_sweep",
                   "results_qpu_sweep",
                   "results_pypsa_glpk_sweep",
               ],
               line_names=["total_cost", "lowest_flow", "closest_flow", "glpk"]
               )

    regex = "*input*30_0_1_80_1 *70_0_1_250_1"
    constraints = {
        'slack_var_factor': [30],
        'chain_strength': [80],
        'problem_size': [15, 16, 17],
        'mangled_total_anneal_time': [20, 40]
    }
    plot_group(f"num_reads_to_cost_slack_30_mean",
               "qpu",
               [
                   regex,
                   regex,
                   regex,
               ],
               "num_reads",
               y_field_list=["total_cost", "lowest_flow", "closest_flow"],
               logscale_x=False,
               split_fields=[],
               constraints=constraints,
               )
    constraints = {
        'slack_var_factor': [70],
        'chain_strength': [250],
        'problem_size': [7, 8, 9],
        'mangled_total_anneal_time': [20, 40]
    }
    plot_group(f"num_reads_to_cost_slack_70_mean",
               "qpu",
               [
                   regex,
                   regex,
                   regex,
               ],
               "num_reads",
               y_field_list=["total_cost", "lowest_flow", "closest_flow"],
               logscale_x=False,
               split_fields=[],
               constraints=constraints,
               )

    regex = '*20.nc_110_365_30_0_1_80_*_1'
    plot_group("opt_size_to_cost_split_sampleCutSize_mean",
               "qpu_read",
               [
                   regex,
                   "*20.nc*"
               ],
               "problem_size",
               y_field_list=["cut_samples_cost", "total_cost"],
               split_fields=["sampleCutSize"],
               logscale_x=False,
               line_names=['cut_sampplesCost', 'glpk'],
               constraints={"sampleCutSize": [1, 2, 5, 10, 30, 100, 365]},
               path=[
                   "results_qpu_read_sweep",
                   "results_pypsa_glpk_sweep"
               ],
               )
    plot_group("opt_size_to_cost_mean",
               "qpu_read",
               [
                   regex,
                   regex,
                   regex,
                   "*20.nc_30_*",
               ],
               "problem_size",
               y_field_list=["total_cost", "lowest_flow", "closest_flow", "total_cost"],
               split_fields=[],
               reduction_method=[np.mean] * 4,
               logscale_x=False,
               line_names=["qpu_total_cost", "LowFlow", "CloseFlow", "glpk_total_cost"],
               path=["results_qpu_read_sweep"] * 3 + ["results_pypsa_glpk_sweep"],
               constraints={"sampleCutSize": [1]},
               )
    plot_group("opt_size_to_cost_median",
               "qpu_read",
               [
                   regex,
                   regex,
                   regex,
                   "*20.nc_30_*",
               ],
               "problem_size",
               y_field_list=["total_cost", "lowest_flow", "closest_flow", "total_cost"],
               split_fields=[],
               reduction_method=[np.median] * 4,
               logscale_x=False,
               line_names=["qpu_total_cost", "LowFlow", "CloseFlow", "glpk_total_cost"],
               path=["results_qpu_read_sweep"] * 3 + ["results_pypsa_glpk_sweep"],
               constraints={"sampleCutSize": [1]},
               )

    for slackVar in range(10, 60, 10):
        for chainStrength in range(20, 60, 10):
            regex = f"*input_[7-9]*_20.nc_78_258_{slackVar}_0_1_{chainStrength}_1"
            for strategy in ["total_cost", "lowest_flow", "closest_flow"]:
                plot_group(f"costDistribution_for_{strategy}_slack_{slackVar}_chain_{chainStrength}",
                           "qpu",
                           [
                               regex,
                           ],
                           strategy,
                           y_field_list=[strategy],
                           reduction_method=[len],
                           logscale_x=False,
                           split_fields=[],
                           plottype="histogramm",
                           error_method=None,
                           )
    regex = "*input_[7-9]*_20.nc_78_258_[1-5]0_0_1_[2-5]0_1"
    for strategy in ["total_cost", "lowest_flow", "closest_flow"]:
        plot_group(f"costDistribution_for_{strategy}",
                   "qpu",
                   [
                       regex,
                   ],
                   strategy,
                   y_field_list=[strategy],
                   reduction_method=[len],
                   logscale_x=False,
                   split_fields=[],
                   plottype="histogramm",
                   error_method=None,
                   )

    regex = "*input_[7-9]_*_20.nc_*_70_0_[01]_250_1"
    constraints = {'mangled_total_anneal_time': [19, 20],
                   'chain_strength': [250],
                   'slack_var_factor': [70.0],
                   'maxOrder': [0, 1],
                   }
    plot_group("annealReadRatio_to_cost_mean",
               "qpu",
               [
                   regex,
                   regex,
                   regex,
               ],
               "annealReadRatio",
               y_field_list=["total_cost", "lowest_flow", "closest_flow"],
               split_fields=[],
               logscale_x=True,
               constraints=constraints,
               )
    plot_group("annealTime_to_cost_mean",
               "qpu",
               [
                   regex,
                   regex,
                   regex,
               ],
               "annealing_time",
               y_field_list=["total_cost", "lowest_flow", "closest_flow"],
               logscale_x=True,
               split_fields=[],
               constraints=constraints,
               )

    regex = '*put_[7-9]*70_0_[01]_250_1'
    constraints = {'mangled_total_anneal_time': [20],
                   'maxOrder': [0, 1],
                   'line_representation': [0],
                   'slack_var_factor': [70.0],
                   'chain_strength': [250],
                   }
    for strategy in ["total_cost", "lowest_flow", "closest_flow"]:
        plot_group(f"anneal_read_ratio_to{strategy}_split_maxOrd_mean",
                   "qpu",
                   [
                       regex,
                   ],
                   "annealReadRatio",
                   y_field_list=[strategy],
                   logscale_x=True,
                   logscale_y=False,
                   split_fields=["maxOrder", ],
                   constraints=constraints,
                   )
        plot_group(f"anneal_read_ratio_to{strategy}_split_maxOrd_median",
                   "qpu",
                   [
                       regex,
                   ],
                   "annealReadRatio",
                   y_field_list=[strategy],
                   logscale_x=True,
                   logscale_y=False,
                   reduction_method=[np.median],
                   split_fields=["maxOrder", ],
                   constraints=constraints,
                   )

    regex = "*input_[7-9]*_20.nc_78_258_*"
    chainStrengthList = list(range(30, 80, 20)) + [100]
    constraints = {'slack_var_factor': range(10, 50, 10),
                   'chain_strength': chainStrengthList,
                   'num_reads': [258],
                   'annealing_time': [78],
                   }
    plot_group("slackvar_to_cost_mean",
               "qpu",
               [regex] * 3,
               "slack_var_factor",
               y_field_list=["total_cost", "lowest_flow", "closest_flow"],
               reduction_method=[np.mean] * 3,
               logscale_x=False,
               split_fields=[],
               constraints=constraints,
               )
    plot_group("slackvar_to_cost_split_chains_close_flow_mean",
               "qpu",
               [regex],
               "slack_var_factor",
               y_field_list=["closest_flow"],
               reduction_method=[np.mean],
               logscale_x=False,
               split_fields=["chain_strength"],
               constraints=constraints,
               )
    plot_group("slackvar_to_cost_split_chains_low_flow_mean",
               "qpu",
               [regex],
               "slack_var_factor",
               y_field_list=["lowest_flow"],
               reduction_method=[np.mean],
               logscale_x=False,
               split_fields=["chain_strength"],
               constraints=constraints,
               )
    plot_group("slackvar_to_cost_split_chains_total_cost_mean",
               "qpu",
               [regex],
               "slack_var_factor",
               y_field_list=["total_cost"],
               reduction_method=[np.mean],
               logscale_x=False,
               split_fields=["chain_strength"],
               constraints=constraints,
               )
    for chainStrength in chainStrengthList:
        constraints["chain_strength"] = [chainStrength]
        plot_group(f"slackvar_to_cost_chain_{chainStrength}_mean",
                   "qpu",
                   [regex] * 3,
                   "slack_var_factor",
                   y_field_list=["total_cost", "lowest_flow", "closest_flow"],
                   reduction_method=[np.mean] * 3,
                   logscale_x=False,
                   split_fields=[],
                   constraints=constraints,
                   )

    regex = "*put_[7-9]_[0-9]_20.nc_78_258_[1-5]*[0-9][0]_1"
    constraints = {'slack_var_factor': range(10, 60, 10),
                   'chain_strength': list(range(30, 100, 10)) + [100],
                   'line_representation': [0],
                   'maxOrder': [1],
                   }
    plot_group("chain_strength_to_cost_mean",
               "qpu",
               [regex] * 3,
               "chain_strength",
               y_field_list=["total_cost", "lowest_flow", "closest_flow"],
               reduction_method=[np.mean] * 3,
               logscale_x=False,
               split_fields=[],
               constraints=constraints,
               )
    plot_group("chain_strength_to_cost_median",
               "qpu",
               [regex] * 3,
               "chain_strength",
               y_field_list=["total_cost", "lowest_flow", "closest_flow"],
               reduction_method=[np.median] * 3,
               logscale_x=False,
               split_fields=[],
               constraints=constraints,
               )
    plot_group("chain_strength_to_cost_split_slackvar_mean_low_flow",
               "qpu",
               [regex],
               "chain_strength",
               y_field_list=["lowest_flow"],
               reduction_method=[np.mean],
               logscale_x=False,
               split_fields=["slack_var_factor"],
               constraints=constraints,
               )
    plot_group("chain_strength_to_cost_split_slackvar_mean_close_flow",
               "qpu",
               [regex],
               "chain_strength",
               y_field_list=["closest_flow"],
               reduction_method=[np.mean],
               logscale_x=False,
               split_fields=["slack_var_factor"],
               constraints=constraints,
               )
    plot_group("chain_strength_to_cost_split_slackvar_mean_total_cost",
               "qpu",
               [regex],
               "chain_strength",
               y_field_list=["total_cost"],
               reduction_method=[np.mean],
               logscale_x=False,
               split_fields=["slack_var_factor"],
               constraints=constraints,
               )

    constraints = {'mangled_total_anneal_time': [20],
                   'annealing_time': [78],
                   'num_reads': [258],
                   'line_representation': [0],
                   'maxOrder': [1],
                   'slack_var_factor': list(range(10, 60, 10)),
                   'chain_strength': [20, 30, 40, 70, 50, 60, 250],
                   }
    plot_group("SlackVarFactor_to_chain_breaks",
               "qpu",
               [
                   "*input_[789]*[0-9]0_1",
               ],
               "slack_var_factor",
               y_field_list=[["serial", "vectors", "chain_break_fraction", "data"]],
               reduction_method=[np.mean] * 3,
               logscale_x=False,
               split_fields=["chain_strength"],
               constraints=constraints,
               )

    plot_group("glpk_size_to_time_and_cost_mean",
               "pypsa_glpk",
               [
                   '*20.nc_30*',
                   '*20.nc_30*',
               ],
               x_field="problem_size",
               y_field_list=["time", "total_cost"],
               split_fields=[],
               logscale_x=False,
               logscale_y=False,
               )

    plot_group("glpk_size_to_cost_mean",
               "pypsa_glpk",
               [
                   '*',
               ],
               x_field="problem_size",
               split_fields=[],
               logscale_x=False,
               logscale_y=False,
               )

    regex = '*input_1[5-7]*20.nc_110_365_30_0_1_80_*_1'
    plot_group("sampleCutSize_to_cut_samples_cost_mean",
               "qpu_read",
               [
                   regex,
               ],
               "sampleCutSize",
               y_field_list=["cut_samples_cost"],
               split_fields=[],
               constraints={'mangled_total_anneal_time': [40],
                            'sampleCutSize': list(range(0, 10, 1)) + \
                                             list(range(10, 30, 5)) + [30] + \
                                             list(range(50, 100, 50)) + [100],
                            },
               logscale_x=True,
               )

    regex = '*input_1[5-7]*20.nc_*'
    constraints = {'slack_var_factor': [30],
                   'chain_strength': [80],
                   'num_reads': [365],
                   'line_representation': [0],
                   'maxOrder': [1],
                   'sampleCutSize': [100],
                   'annealing_time': [10, 20, 40, 50, 70, 80, 110],
                   'problem_size': [15, 16, 17],
                   }
    plot_group("annealTime_to_sampleCost_same_reads_mean",
               "qpu_read",
               [
                   regex
               ],
               "annealing_time",
               reduction_method=[np.mean],
               y_field_list=["cut_samples_cost"],
               split_fields=[],
               constraints=constraints,
               logscale_x=False)
    plot_group("annealTime_to_sampleCost_same_reads_median",
               "qpu_read",
               [
                   regex
               ],
               "annealing_time",
               reduction_method=[np.median],
               y_field_list=["cut_samples_cost"],
               split_fields=[],
               constraints=constraints,
               logscale_x=False)
    plot_group("annealTime_to_cost_same_reads_mean",
               "qpu_read",
               [
                   regex,
                   regex,
                   regex,
               ],
               "annealing_time",
               y_field_list=["total_cost", "lowest_flow", "closest_flow"],
               split_fields=[],
               constraints=constraints,
               logscale_x=False)
    plot_group("annealTime_to_cost_same_reads_median",
               "qpu_read",
               [
                   regex,
                   regex,
                   regex
               ],
               "annealing_time",
               reduction_method=[np.median] * 3,
               y_field_list=["total_cost", "lowest_flow", "closest_flow"],
               split_fields=[],
               constraints=constraints,
               logscale_x=False)

    regex = '*input_1[5-7]*20.nc*'
    plot_group("sampleCutSize_to_cost_split_annealTime_mean",
               "qpu_read",
               [
                   regex,
               ],
               "sampleCutSize",
               reduction_method=[np.mean],
               y_field_list=["cut_samples_cost"],
               split_fields=["annealing_time"],
               constraints={'mangled_total_anneal_time': [40],
                            'slack_var_factor': [30.0],
                            'chain_strength': [80],
                            'annealing_time': [50, 110, 125, 250, 2000]}
               )
    plot_group("sampleCutSize_to_cost_split_annealTime_median",
               "qpu_read",
               [
                   regex,
               ],
               "sampleCutSize",
               reduction_method=[np.median],
               y_field_list=["cut_samples_cost"],
               split_fields=["annealing_time"],
               constraints={'mangled_total_anneal_time': [40, 200],
                            'slack_var_factor': [30.0],
                            'chain_strength': [80],
                            'annealing_time': [50, 110, 125, 250]}
               )

    # chr(956) in python3 is \mu
    plot_group("annealReadRatio_to_access_time",
               "qpu",
               ["*"],
               "annealReadRatio",
               y_field_list=[["serial", "info", "timing", "qpu_access_time"]],
               split_fields=["mangled_total_anneal_time"],
               constraints={"mangled_total_anneal_time": [20, 40]},
               ylabel="total qpu access time in " + chr(956) + "s",
               logscale_x=True,
               logscale_y=True,
               )

    regex = "*20.nc_110_365_30_0_1_80_1"
    constraints = {'slack_var_factor': [30],
                   'chain_strength': [80],
                   'num_reads': [365],
                   'annealing_time': [110],
                   }
    plot_group("size_to_chain_breaks_mean",
               "qpu",
               [
                   regex,
               ],
               "problem_size",
               y_field_list=[["serial", "vectors", "chain_break_fraction", "data"]],
               reduction_method=[np.mean],
               logscale_x=False,
               split_fields=["chain_strength"],
               )

    # TODO 
    constraints = {"maxOrder": [1, 2, 3, 4]}
    for scale in ["25", "20"]:
        plot_group(f"sqa_line_representation_to_cost_scale_{scale}_split_maxOrd_mean",
                   "sqa",
                   [
                       f"*_*_{scale}.nc*",
                   ],
                   "line_representation",
                   reduction_method=[
                       np.mean,
                   ],
                   split_fields=["maxOrder"],
                   logscale_x=False,
                   logscale_y=False,
                   constraints=constraints,
                   )
        plot_group(f"sqa_line_representation_to_cost_scale_{scale}_split_maxOrd_median",
                   "sqa",
                   [
                       f"*_*_{scale}.nc*",
                   ],
                   "line_representation",
                   reduction_method=[
                       np.median,
                   ],
                   split_fields=["maxOrder"],
                   logscale_x=False,
                   logscale_y=False,
                   constraints=constraints,
                   )

    plot_group("CostVsTraining_small",
               "sqa",
               ["info_input_10*"],
               x_field="steps",
               logscale_x=True,
               logscale_y=True,
               path=["results_sqa_backup"]
               )

    plot_group("CostVsTraining_medium",
               "sqa",
               ["info_input_25*"],
               x_field="steps",
               logscale_x=True,
               logscale_y=True,
               path=["results_sqa_backup"]
               )

    plot_group("CostVsTraining_large",
               "sqa",
               ["info_input_50*"],
               x_field="steps",
               logscale_x=True,
               logscale_y=True,
               path=["results_sqa_backup"]
               )

    plot_group("classical_small",
               "classical",
               ["info_input_10*"],
               x_field="steps",
               logscale_x=True,
               logscale_y=True,
               path=["results_classical"]
               )

    plot_group("sqa_H_to_cost",
               "sqa",
               ["info_input_10*nc_*"],
               x_field="H",
               logscale_y=True,
               logscale_x=False,
               path=["results_sqa_sweep_old"],
               )

    plot_group("sqa_H_to_cost_with_low_T",
               "sqa",
               ["info_input_10*nc_0.00[01]*"],
               x_field="H",
               logscale_y=True,
               logscale_x=False,
               path=["results_sqa_sweep_old"],
               )

    plot_group("sqa_T_to_all_cost",
               "sqa",
               ["info_input_10*nc_*"],
               x_field="T",
               logscale_y=True,
               logscale_x=False,
               path=["results_sqa_sweep_old"],
               )

    plot_group("sqa_T_to_cost_with_good_H",
               "sqa",
               ["info_input_10*nc_*_[67].0_?"],
               x_field="T",
               logscale_y=True,
               logscale_x=False,
               path=["results_sqa_sweep_old"],
               )

    plot_group(f"sqa_binary_split_approx_comparison",
               "sqa",
               [
                   # "*2[50].nc*",
                   # "*2[50].nc*",
                   "*",
               ],
               "line_representation",
               reduction_method=[
                   np.mean,
               ],
               line_names=[
                   "mean",
               ],
               split_fields=["maxOrder"],
               logscale_x=False,
               logscale_y=False,
               )
    return


if __name__ == "__main__":
    main()
