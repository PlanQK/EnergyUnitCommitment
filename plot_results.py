import glob
import json
import collections
from typing import Optional

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
#TODO remove
HACKCOUNTER = 0
 
    
def meanOfSquareRoot(values: list) -> float:
    """
    Reduction method to reduce the given values into one float by first applying 
    a square root to all values and then averaging them by using "mean"-method"

    @param values: list
        A list of values to be reduced.
    @return: float
        Reduced value.
    """
    return np.mean([np.sqrt(value) for value in values])


def meanOfAnnealingComputingCost(values : list) -> float :
    return np.mean([COSTPERHOUR*value for value in values])


def deviationOfTheMean(values: list) -> float:
    """
    Reduction method to reduce the given values into one float using the "deviation of the mean"-method

    @param values: list
        A list of values to be reduced.
    @return: float
        Reduced value.
    """
    return np.std(values) / np.sqrt(len(values))


def averageOfBetterThanMedian(values: list) -> float:
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


def averageOfBest(values: list) -> float:
    """
    Reduction method to reduce the given values into one float using the "average of best"-method

    @param values: list
        A list of values to be reduced.
    @return: float
        Reduced value.
    """
    values.sort()
    return np.mean(values[:-1])


def averageOfBestPercent(values: list, percentage: float) -> float:
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


def cumulativeDistribution(values: list) -> list:
    """
    Reduction method to construct a cumulativeDistribution for a list of lists of values
    It returns a list of values that a histogramm of that list will look like the cumulative
    distribution of all values

    @param values: list
        A list of lists of values for which to construct a cumulative Distribution
    @param: list
        A list which histogramm is a cumulative distribution
    """
    result = []
    maxVal = max(max(values,key=max)) + 1
    for valueLists in values:
        for val in valueLists:
            result += list(range(int(val), int(maxVal), 1))
    return [result]
    

class PlottingAgent:
    """
    class for creating plots. on initialization reads and prepares data files. Plots based on that 
    data can then be created and written by calling makeFigure
    """
    def __init__(self, fileformat = "png"):
        """
        constructor for a plotting agent. On initialization, constructs a DataExtractionAgent to handle
        data management
        
        @param globDict: dict
            a dictionary containg lists of glob expressions for the solver whose data will be plotted
        """
        self.savepath = "plots"
        self.fileformat = fileformat


    @classmethod
    def read_csv(cls, csv_file: str, fileformat = "png"):
        agent = PlottingAgent(fileformat)
        agent.dataExtractionAgent = DataExtractionAgent.read_csv(csv_file)
        return agent
    
    @classmethod
    def extract_from_json(cls, globDict, constraints={}, fileformat = "png"):
        agent = PlottingAgent(fileformat)
        agent.dataExtractionAgent = DataExtractionAgent.extract_from_json(
                globDict=globDict, constraints=constraints, 
        ) 
        return agent
        
    def to_csv(self, filename:str):
        self.dataExtractionAgent.df.to_csv(filename)
    

    def getDataPoints(self,
                    xField, 
                    yFieldList,
                    splittingFields,
                    constraints,
        ):
        """
        returns all stored data points with x-value in xField and y-values in yFieldList. 
        The result is a dictonary with keys being the index of the groupby by the
        chosen splittingFields. The value is a dictionary over the chosen yFields with values being a pair or
        lists. The first one consists of x-values and the second entry consists of unsanitized y-values
        
        @param xField: str
            label of the columns containing x-values
        @param yFieldList: str
            list of labels of the columns containing y-values
        @param splittingFields: list
            list of labels by which columns to groupby the points
        @param constraints: dict
            rules for restricting which data to access from the stored data frame
        @return: dict
             a dictonary over groupby indices with values being dictionaries over yField labels
        """
        dataFrame = self.dataExtractionAgent.getData(constraints)[[xField] + yFieldList + splittingFields]
        result = {}
        if splittingFields:
            groupedDataFrame = dataFrame.groupby(splittingFields, dropna=False).agg(list)
            for idx in groupedDataFrame.index:
                # if there is only 1 splitField, the index is not a multiindex
                if len(splittingFields) == 1:
                    idx_tuple = [idx]
                else:
                    idx_tuple = idx
                key = tuple([
                        splitField + "=" + str(idx_tuple[counter])
                        for counter, splitField in enumerate(splittingFields)
                ])
                result[key] = {
                        yField : (groupedDataFrame.loc[idx][xField], groupedDataFrame.loc[idx][yField]) 
                        for yField in yFieldList
                }
        else:
            result[tuple()] = {
                        yField : (dataFrame[xField], dataFrame[yField]) for yField in yFieldList
                }
        return result
    

    def aggregateData(self, xValues, yValues, aggregationMethods, ):
        """
        aggregation nethod for a list of yValues grouped by their corresponding xValue
        
        @param xValues: list
            list of xValues for points in a data set
        @param yValues: list
            list if yValues for points in a data set
        @param aggregationMethods: list
            list of arguments that are accepted by pandas data frames' groupby as aggregation methods
        @return: pd.DataFrame
            a pd.DataFrame with xValues as indices of a groupby by the argument aggregationMethods
        """
        if len(xValues) != len(yValues):
            raise ValueError("data lists are of unequal length")
        df = pd.DataFrame({"xField" : xValues, "yField" : yValues})
        yValues_df = df[df["yField"].notna()]
        naValues_df = df[df["xField"].isna()]
        return yValues_df.groupby("xField").agg(aggregationMethods)["yField"] , \
                np.mean(naValues_df["yField"])

    def makeFigure(self,
            plotname: str,
            xField: str = "problemSize", 
            yFieldList: list = ["totalCost"], 
            splitFields: list = [],
            constraints: dict = {},
            aggregateMethod = None,
            errorMethod = None,
            logscalex: bool = False,
            logscaley: bool = False,
            xlabel: str = None,
            ylabel: str = None,
            title: str = None,
            plottype: str = "line", 
            regression: bool = True,
            **kwargs
    ):
        fig, ax = plt.subplots()


        data = self.getDataPoints(
                        xField=xField,
                        yFieldList=yFieldList,
                        constraints=constraints,
                        splittingFields=splitFields
        )

        for splitfieldKey, dataDictionary in data.items():
            for yField, yFieldValues in dataDictionary.items() :
                xCoordinateList = yFieldValues[0]
                yCoordinateList = yFieldValues[1]
                if splitfieldKey:
                    label = str(splitfieldKey) + "_" + yField
                else:
                    label = yField
                yFieldListStripped = ", ".join(yFieldList)

                if plottype == "line":
                    if aggregateMethod is None:
                        aggregateMethod = 'mean'
                    if errorMethod is None:
                        errorMethod = deviationOfTheMean
                    yValues, naValues = self.aggregateData(
                                xCoordinateList,
                                yCoordinateList,
                                [aggregateMethod, errorMethod],
                    )
                    ax.errorbar(yValues.index,
                            yValues.iloc[:,0],
                            label=label,
                            yerr=yValues.iloc[:,1],
                            **kwargs)

                    ax.axhline(naValues)

#                    print(yFieldValues[1][yFieldValues[0].isna()])
#                    ax.hline()

                if plottype == "scatterplot":
                    if 's' not in kwargs:
                        kwargs['s'] = 7
                    ax.scatter(xCoordinateList,yCoordinateList, **kwargs, label=label)
                    # linear regression
                    if regression:
                        m, b = np.polyfit(xCoordinateList, yCoordinateList, 1)
                        ax.plot(xCoordinateList, [m * z + b for z in xCoordinateList], color='red', label=label)

                if plottype == "histogramm":
                    ax.hist(yCoordinateList,label=label, **kwargs)
                    if xlabel is None:
                        xlabel = yFieldListStripped
                    if ylabel is None:
                        ylabel = "count"

                if plottype == "boxplot":
                    ax.boxplot(yCoordinateList)
                    if xlabel is None:
                        xlabel = " "

                if plottype == "cumulative":
                    if True:
                        sorted_data = np.sort(yCoordinateList)
                        plt.step(sorted_data, np.arange(sorted_data.size), label=label)
                    else:
                        values, base = np.histogram(yCoordinateList, bins=40)
                        cumulative = np.cumsum(values)
                        plt.step(base[:-1], cumulative, label=label)
                    if xlabel is None:
                        xlabel = yFieldListStripped
                    if ylabel is None:
                        ylabel = "count"
                    
                if plottype == "density":
                    sns.kdeplot(
                            list(yCoordinateList),
                            label=label,
                            clip =(0.0, max(yCoordinateList))
                    )
                    if xlabel is None:
                        xlabel = yFieldListStripped
                    if ylabel is None:
                        ylabel = "density"

        if xlabel is None:
            xlabel = xField
        if ylabel is None:
            ylabel = yFieldList
        if title is None:
            title = plottype

        # adjusting additonal settings
        plt.legend()
        if logscalex:
            ax.set_xscale("log")
        if logscaley:
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
    solverKeys = {
            "sqa" : ["sqaBackend", "isingInterface"],
            "qpu" : ["dwaveBackend", "isingInterface","cutSamples"],
            "qpu_read" : ["dwaveBackend", "isingInterface","cutSamples"],
            "pypsa_glpk" : ["pypsaBackend"],
            "classical" : []
    }
    # list of all keys for data entries, that every result file of a solver has
    generalFields = [
            "fileName",
            "inputFile",
            "problemSize",
            "scale",
            "timeout",
            "totalCost",
            "marginalCost",
            "powerImbalance",
            "kirchhoffCost",
            "optimizationTime",
            "postprocessingTime",
    ]


    def __init__(self, prefix: str = "results", suffix: str = "sweep"):
        """
        constructor for a data extraction. data files specified in globDict are read, filtered by
        constraints and then saved into a pandas data frame. stored data can be queried by a getter
        for a given solver, the data is read from the folder f"{solver}_results_{suffix}"
        
        @param globDict: dict
            a dictionary with keys specifying a solver and the value a being a list of glob expressions
            to specify files
        @param constraints: dict
            a dictionary that describes by which rules to filter the files read
        @param suffix: str
            suffix or results folder in which to apply glob. 
        """
        self.pathDictionary = {
            solver : "_".join([prefix, solver, suffix]) for solver in self.solverKeys.keys()
        }

    @classmethod
    def read_csv(cls, filename:str):
        agent = DataExtractionAgent()
        agent.df = pd.read_csv(filename)
        agent.df = agent.df.apply(pd.to_numeric, errors='ignore')
        print(agent.df.columns)
        return agent

    @classmethod
    def extract_from_json(cls, globDict: dict, constraints: dict = {}):
        agent = DataExtractionAgent()
        agent.df = pd.DataFrame()
        for solver, globList in globDict.items():
            agent.df = agent.df.append(agent.extractData(solver, globList,))
        agent.df = agent.filterByConstraint(constraints)
        agent.df = agent.df.apply(pd.to_numeric, errors='ignore')
        return agent
    

    def expandSolverDict(self, dictKey, dictValue):
        """
        expand a dictionary containing solver dependent information into a list of dictionary
        to seperate information of seperate runs saved in the same file
        
        @param dictKey: str
            dictionary key that was used to accessed solver depdendent data
        @param dictValue: str
            dictionary that contains solver specific data to be expanded
        @return: list
            a list of dictionaries which are to be used in a cross product
        """
        # normal return value if no special case for expansion applies
        result = [dictValue] 
        if dictKey == "cutSamples":
            result = [
                    {
                        "sample_id": key,
                        **value 
                    }
                    for key, value in dictValue.items()
            ]
        if dictKey == "sqaBackend":
            eigenValues = sorted(dictValue["eigenValues"])
            del dictValue["eigenValues"]
            result = [
                    {
                        "order_ev": idx,
                        "eigenValue": eigenValue,
                        **dictValue
                    }
                    for idx, eigenValue in enumerate(eigenValues) 
            ]
        return result


    def filterByConstraint(self, constraints):
        """
        method to filter data frame by constraints. This class implements the filter rule as a
        dictionary with a key, value pair filtering the data frame for rows that have a non trivial entry
        in the column key by checking if that value is an element in value
        
        @param dataFrame: pd.DataFrame
            a data frame on which to apply the constraints
        @param constraints: dict
            a parameter that describes by which rules to filter the data frame. 
        @return: pd.DataFrame
            a data frame filtered accoring to the given constraints
        """
        result = self.df
        for key, value in constraints.items():
            result = result[result[key].isin(value) | result[key].isna()]
        return result


    def getData(self, constraints: dict = None):
        """
        getter for accessing result data. The results are filtered by constraints. 
        
        @param constraints: dict
            a dictionary with data frame columns indices as key and lists of values. The data frame 
            is filtered by rows that contain such a value in that columns
        
        @return: pd.DataFrame
            a data frame of all relevant data filtered by constraints
        """
        return self.filterByConstraint(constraints)


    def extractData(self, solver, globList,):
        """
        extracts all relevant data to be saved in a pandas DataFrame from the results of a given solver. 
        The result files to be read are specified in a list of globs expressions
        
        @param solver: str
            label of the solver for which to extract data
        @param globList: list
            list of strings, which are glob expressions to specifiy result files
        @return: pd.DataFrame
            a pandas data frame containing all relevant data of the solver results
        """
        # plotData = collections.defaultdict(collections.defaultdict)
        result = pd.DataFrame()
        for globExpr in globList:
            for fileName in glob.glob(path.join(self.pathDictionary[solver], globExpr)):
                result = result.append(self.extractDataFromFile(solver, fileName,))
        return result
        

    def extractDataFromFile(self,solver, fileName,):
        """
        reads the result file of a solver and builds a pandas data frame containing a entry for
        each run saved in the file

        @param solver: str
            label of the solver for which to extract data
        @param fileName: str
            path of the file that contains solver results
        @return: pd.DataFrame
            a pandas data frame containing a row for every run saved in the file
        """
        with open(fileName) as file:
            fileData = json.load(file)
        # Data Frame containing a single row with columns as the data fields that every solver run
        # contains. Additional solver dependent information is merged on this data frame
        generalData = pd.DataFrame({"solver" : solver, 
                                **{key : [fileData[key]] for key in self.generalFields}
                                }
        )
        solverDependentData = [
                self.expandSolverDict(solverDependentKey, fileData[solverDependentKey])
                for solverDependentKey in self.solverKeys[solver]
        ]
        for solverDependentField in solverDependentData:
            generalData = pd.merge(generalData, pd.DataFrame(solverDependentField), how="cross")
        return generalData


def extractInformation(fileRegex: str, xField: str, yField: str,
                       splitFields: list = ["problemSize"], reductionMethod=np.mean,
                       errorMethod=deviationOfTheMean, constraints: dict = {}, embedding: bool = False) -> dict:
    """
    Extracts the information to be plotted.

    @param fileRegex: str
        A glob pattern to get the data
    @param xField: str
        The name of the x-Axis key. Nested keys can be reached by providing a list of strings, which represent the path
        through the nested dicts.
    @param yField: str
        The name of the y-axis key. Nested keys can be reached by providing a list of strings, which represent the path
        through the nested dicts.
    @param splitFields: list
        A list with keys for which the extracted data should be grouped.
    @param reductionMethod: func
        The function to be used to reduce values into one float value.
    @param errorMethod: func
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
    filesRead = 1
    plotData = collections.defaultdict(collections.defaultdict)
    for fileName in glob.glob(fileRegex):
        with open(fileName) as file:
            element = json.load(file)
            if embedding:
                element = addEmbeddingInformation(jsonDict=element)
            else:
                element = addPlottableInformation(jsonDict=element)

        # TODO indentation of for loop + cast error for ""
        for key, values in constraints.items():
#            try:
#                # value of constraint is not in constrains list
            if float(resolveKey(element, key)) not in values:
                break
#            except KeyError:
#                pass
        # value of constraint is found in constrains list
        else:
            # create a new key using the splitField
            key = tuple(
                e
                for e in [
                    splitField + "=" + str(resolveKey(element, splitField) or "")
                    for splitField in splitFields
                ]
            )

            xvalue = resolveKey(element, xField)

            if xvalue not in plotData[key]:
                plotData[key][xvalue] = []

            yvalue = resolveKey(element, yField)

            plotData[key][xvalue].append(yvalue)
        filesRead += 1

    # now perform reduction
    result = collections.defaultdict(list)
    for outerKey in plotData:
        for innerKey in plotData[outerKey]:
            if isinstance(innerKey, str):
                # H, T in sqa data
                if innerKey.startswith("["):
                    xvalue = innerKey[1:-1].split(',')
                else:
                    xvalue = innerKey
            else:
                xvalue = innerKey
            result[outerKey].append(
                [
                    # sometimes xvalue is still a string and has to be cast to a float
                    xvalue,
                    reductionMethod(plotData[outerKey][innerKey]),
                    errorMethod(plotData[outerKey][innerKey]) if errorMethod is not None else 0
                ]
            )
        result[outerKey].sort()

    if PRINT_NUM_READ_FILES:
        print(f"files read for {fileRegex} : {filesRead}")

    return result



def extractCutSamples(cutSamplesDictList: object, x_key: str="energy", y_key: str="optimizedCost") -> [list, list]:
    """
    Reduction method to extract data from all shots of a single batch. It creates two lists for making
    a scatter plot: First list consists of the spin glass's systems energy for the x-axis and second list is
    the corresponding optimized cost of the optimization problem. x_key is the dictionary key of the 
    metric to be extraced and used on the x-axis. y_key is the dictionary key of the 
    metric to be extraced and used on the y-axis.

    @param cutSamplesDictList: object
        List of dicts to be searched through.
    @param key: str
        dictionary key of metric to be extracted
    @return: [list, list]
        A list with the extracted energy and one with the optimized costs.
    """
    x_axis_list = []
    y_axis_list = []
    for cutSamplesDict in cutSamplesDictList:
        for value in cutSamplesDict.values():
            x_axis_list.append(value[x_key])
            y_axis_list.append(value[y_key])
    return x_axis_list, y_axis_list


def makeFig(plotInfo: dict, outputFile: str,
            logscalex: bool = False, logscaley: bool = False, xlabel: str = None, ylabel: str = None, title: str = None,
            fileformat: str = "pdf", plottype: str = "line", ) -> None:
    """
    Generates the plot and saves it to the specified location.

    @param plotInfo: dict
        The data to be plotted.
    @param outputFile: str
        The path to where the plot will be saved.
    @param logscalex: bool
        Turns the x-axis into a log scale.
    @param logscaley: bool
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
    for key, values in plotInfo.items():
        if plottype == "histogramm":
            # if condition is truthy if the function values have not been reduced earlier. thus in 'yvalues' we have a list of
            # values that go in to the histogramm with weight 1.
            # if the condition is falsy, the xField should be the yvalues we want to plot, using arbitrary yvalues that
            # get reduced by len, thus counting how often a particular value appears in xField
            if hasattr(values[0][1], "__getitem__"):
                for entries in values:
                    flattenedList = [item for sublist in entries[1] for item in sublist]
                    if not flattenedList:
                        continue
                    ax.hist(flattenedList, bins=[i * BINSIZE for i in range(int(min(flattenedList) / BINSIZE) - 2,
                                                                            int(max(flattenedList) / BINSIZE) + 2, 1)],
                            label=key)
            else:
                sortedValues = sorted(values)
                ax.hist([e[0] for e in values], bins=[i * BINSIZE for i in range(0, PLOTLIMIT // BINSIZE + 1, 1)],
                        label=key, weights=[e[1] for e in sortedValues])

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


            ax.scatter(x,y,s=8)
            
            # linear regression
            m, b = np.polyfit(x, y, 1)
            ax.plot(x, [m * z + b for z in x], color='red')

        if plottype == "boxplot":
            global HACKCOUNTER
            HACKCOUNTER += 1
            print("CALLING PLOT")
            sortedValues = sorted(values)
            ax.boxplot(
                    [e[1] for e in sortedValues],
                    )

        # default plot type of a function graph
        if plottype == "line":
            sortedValues = sorted(values)
            ax.errorbar([e[0] for e in sortedValues], [e[1] for e in sortedValues], label=key,
                        yerr=[e[2] for e in sortedValues])

    plt.legend()
    if logscalex:
        ax.set_xscale("log")
    if logscaley:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.suptitle(title)


    if fileformat == "":
        fig.savefig(outputFile + ".png")
        fig.savefig(outputFile + ".pdf")
    else:
        fig.savefig(outputFile + "." + fileformat)

# TODO : probably clashes with default empty string initializes of backend specific variables
def resolveKey(element: dict, field: str) -> any:
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
                out = resolveKey(element[key], field)
                # if the key and therefore a value for it is found, break out of the loop and return the value
                if out is not None:
                    break

    return out


# TODO add embeddings to output data or rewrite how to extract embeddings from embedding dictionarys
def addEmbeddingInformation(jsonDict: dict) -> dict:
    """
    Add embedding information, specific to the dWave Backends, extracted from their backend specific results.

    @param jsonDict: dict
        The dict where the embedding inforamtion is extracted from and added to.
    @return: dict
        The adjusted input dict.
    """
    embeddingDict = jsonDict["info"]["embedding_context"]["embedding"]
    logicalQubits = [int(key) for key in embeddingDict.keys()]
    embeddedQubits = [item for sublist in embeddingDict.values() for item in sublist]

    jsonDict["dwaveBackend"]["embeddedQubits"] = len(embeddedQubits)
    jsonDict["dwaveBackend"]["logicalQubits"] = len(logicalQubits)
    jsonDict["dwaveBackend"]["embedFactor"] = float(jsonDict["dwaveBackend"]["embeddedQubits"]) \
                                              / float(jsonDict["dwaveBackend"]["logicalQubits"])

    return jsonDict


def addPlottableInformation(jsonDict: dict) -> dict:
    """
    Add plottable information, specific to the dWave Backends, extracted from their backend specific results.

    @param jsonDict: dict
        The dict where the plottable inforamtion is extracted from and added to.
    @return: dict
        The adjusted input dict.
    """
    if "cutSamples" in jsonDict:
        jsonDict["dwaveBackend"]["sampleValues"] = [jsonDict["cutSamples"][key]["optimizedCost"]
                                                    for key in jsonDict["cutSamples"].keys()]

    # often, one of those two solutions is significantly better than the other
    if jsonDict["dwaveBackend"]["LowestFlow"] is not None:
        jsonDict["dwaveBackend"]["minChoice"] = min(jsonDict["dwaveBackend"]["LowestFlow"],
                                                    jsonDict["dwaveBackend"]["ClosestFlow"])

    return jsonDict

# TODO odd bug of not working if solver str is empty
def plotGroup(plotname: str, solver: str, fileRegexList: list, xField: str, yFieldList: list = None,
              splitFields: list = ["problemSize"], logscalex: bool = True, logscaley: bool = False,
              PATH: list = None, reductionMethod: list = None, lineNames: list = None,
              embeddingData: bool = False, errorMethod=deviationOfTheMean, constraints: dict = {},
              plottype: str = "line", xlabel: str = None, ylabel: str = None) -> None:
    """
    Extracts data from all files in the fileRegexList list and plots them into a single plot with the given plotname
    and set fileformat. The given files have to be .JSON files.

    @param plotname: str
        The name of the generated plot, without the filetype.
    @param solver: str
        A string to indicate which solver was used. This sets some helpful default values
    @param fileRegexList: list
        A list of all regular expressions to get data. Each item in the list is plotted as it's own line (possibly
        further split up, if splitFields is not empty). A whitespace in a regex string splits multiple regular
        expressions to be plotted into a single line.
    @param xField: str
        The name of the x-axis key. Nested keys can be reached by providing a list of strings, which represent the path
        through the nested dicts.
    @param yFieldList: list
        The list of names of the y-axis keys. It has to be the same size as fileRegexList. The i-th entry in this list
        is for the i-th entry in the fileRegexList. Nested keys can be reached by providing a list of strings, which
        represent the path through the nested dicts.
    @param splitFields: list
        A list with keys for which the extracted data should be grouped.
    @param logscalex: bool
        Turns the x-axis into a log scale.
    @param logscaley: bool
        Turns the y-axis into a log scale.
    @param PATH: list
        A list of the paths to the data. It has to be the same size as fileRegexList. If nothing is given the data is
        searched for in the standard folder f'results_{solver}_sweep'.
    @param reductionMethod: func
        A list of functions to be used to reduce values into one float value. It has to be the same size as
        fileRegexList.
    @param lineNames: list
        A list with the labels for the lines to be plotted. It has to be the same size as fileRegexList. If nothing is
        given, the keys of yFieldList will be used as labels.
    @param embeddingData: bool
        If true, embedded information from dWaveBackend is extracted to be plotted.
    @param errorMethod: func
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
    if yFieldList is None:
        yFieldList = ["totalCost"] * len(fileRegexList)

    if len(fileRegexList) != len(yFieldList):
        print("number of regex doesn't match number of yField's selected")
        return

    if PATH is None:
        PATH = [f"results_{solver}_{RESULT_SUFFIX}"] * len(fileRegexList)

    if reductionMethod is None:
        reductionMethod = [np.mean] * len(fileRegexList)

    if lineNames is None:
        lineNames = yFieldList

    if xlabel is None:
        xlabel = xField

    if ylabel is None:
        ylabel = yFieldList

    plotInfo = {}
    for idx in range(len(fileRegexList)):
        for regex in fileRegexList[idx].split():
            iterator = extractInformation(fileRegex=f"{PATH[idx]}/{regex}",
                                          xField=xField,
                                          yField=yFieldList[idx],
                                          splitFields=splitFields,
                                          reductionMethod=reductionMethod[idx],
                                          errorMethod=errorMethod,
                                          constraints=constraints,
                                          embedding=embeddingData).items()

            for key, value in iterator:
                plotInfoKey = f"{solver}_{key}_{lineNames[idx]}"
                if plotInfoKey in plotInfo:
                    plotInfo[plotInfoKey] += value
                else:
                    plotInfo[plotInfoKey] = value
    makeFig(
        plotInfo,
        f"plots/{plotname}",
        fileformat=FILEFORMAT,
        logscalex=logscalex,
        logscaley=logscaley,
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
    #DataAgent = DataExtractionAgent({"qpu_read" : [regex]} , "sweep")

    networkbatch = "infoNocost_220124cost5input_" 

    PlotAgent = PlottingAgent({
            "sqa" : [networkbatch+ "*[36592]0_?_20.nc*"],
            "pypsa_glpk" : [networkbatch+ "*[36592]0_?_20.nc*"],
    },
    constraints={
        "problemSize" : [30,60,90,120,150],
        "trotterSlices" : [500],
    })
    PlotAgent.makeFigure("large_problem_sqa_glpk",
            splitFields=["solver"], 
    )
    PlotAgent.makeFigure("large_problem_time_sqa_glpk",
            yFieldList = ["optimizationTime"],
            splitFields=["solver"], 
    )
    PlotAgent.makeFigure("large_problem_time_and_cost_sqa_glpk",
            yFieldList = ["optimizationTime", "totalCost"],
            splitFields=["solver"], 
    )
    
    return

    PlotAgent = PlottingAgent({
                    "qpu" : [networkbatch + "1[0-9]_[0-9]_20.nc_*_200_fullsplit_60_1" ,
                        networkbatch + "1[5-9]_1[0-9]_20.nc_*_200_fullsplit_60_1"
                    ],
#                    "pypsa_glpk" : [networkbatch + "1[0-9]_[0-9]_20.nc_30_?"],
                    },
                    constraints={}
                    )

    constraints={"annealing_time" : [100],  }
    PlotAgent.makeFigure("annealing_time_to_cost_len",
            xField = "annealing_time",
            yFieldList = ["totalCost",],
            constraints = {"sample_id" : [0]},
            aggregateMethod = len,
            errorMethod= lambda x: 0,
            )
    PlotAgent.makeFigure("problemsize_to_cost_mean",
            xField = "problemSize",
            yFieldList=["LowestEnergy", "LowestFlow", "ClosestFlow"],
            constraints={"sample_id" : [1], **constraints}
            )
    PlotAgent.makeFigure("problemsize_to_optimized_quantum_cost_mean",
            xField = "problemSize",
            yFieldList=["quantumCost", "optimizedCost"],
            constraints=constraints
            )
    PlotAgent.makeFigure("problemsize_to_optimized_quantum_cost_mean_constraint_100",
            xField = "problemSize",
            yFieldList=["quantumCost", "optimizedCost"],
            constraints={"quantumCost" : list(range(100)) , **constraints}
            )
    PlotAgent.makeFigure("problemsize_to_maxcut_totalcost_mean",
            xField = "problemSize",
            yFieldList=["LowestEnergy", "cutSamplesCost"],
            constraints={"sample_id" : [1], **constraints}
            )
    PlotAgent.makeFigure("annealing_time_to_cost_mean",
            xField = "annealing_time",
            yFieldList = ["totalCost", "LowestFlow","ClosestFlow","cutSamplesCost"],
            constraints = {"problemSize" : [10,11,12,13], "sample_id" : [0], },
            )
    PlotAgent.makeFigure("annealing_time_to_cost_len",
            xField = "annealing_time",
            yFieldList = ["totalCost",],
            constraints = {"problemSize" : [10,11,12,13], "sample_id" : [0], },
            aggregateMethod = len,
            errorMethod= lambda x: 0,
            )
#    networklist = ["".join(["Nocost_220124cost5input_15_",str(i),"_20.nc"]) for i in [0,1,2]]
#    for i in [0,1,2]:
#        PlotAgent.makeFigure(f"scatter_quantumcost_to_optimizedcost_{i}",
#                xField="quantumCost",
#                yFieldList=["optimizedCost"],
#                constraints={"inputFile" : [networklist],
#                        "annealing_time" : [100]},
#                plottype="scatterplot"
#                )

    networkbatch = "infoNocost_220124cost5input_" 
    PlotAgent = PlottingAgent({
                    "sqa" : [networkbatch + "1[3-5]_[0-9]_20.nc_0.1_8.0_*_fullsplit_1" ,
                    ],
                    },
                    constraints={
                        "trotterSlices" : list(range(10,110,10)),
                        "optimizationCycles" : list(range(4,10,2))+list(range(10,100,5))
                    }
                    )
    PlotAgent.makeFigure("optimizationCycles_to_cost_mean",
            xField="optimizationCycles",
            yFieldList=["totalCost"],
            )
    PlotAgent.makeFigure("optimizationCycles_to_cost_mean_split_trotter",
            xField="optimizationCycles",
            yFieldList=["totalCost"],
            splitFields=["trotterSlices"],
            constraints={"trotterSlices" : [10,20,40,80,100]}
            )
    PlotAgent.makeFigure("trotter_to_cost_mean",
            xField="trotterSlices",
            yFieldList=["totalCost"],
            )
    PlotAgent.makeFigure("trotter_to_cost_mean_split_optimizationCycles",
            xField="trotterSlices",
            yFieldList=["totalCost"],
            splitFields=["optimizationCycles"],
            constraints={"optimizationCycles" : [4,10,30,70]}
            )
    PlotAgent.makeFigure(f"scatter_trotter_to_cost",
            xField="trotterSlices",
            yFieldList=["totalCost"],
            splitFields=["optimizationCycles"],
            constraints={"optimizationCycles": [4, 30 ,70]},
            plottype="scatterplot",
            s=12,
            regression=False,
            )
    PlotAgent.makeFigure(f"scatter_optimizationCycles_to_cost",
            xField="optimizationCycles",
            yFieldList=["totalCost"],
            splitFields=["trotterSlices"],
            constraints={"optimizationCycles": [4, 30 ,70]},
            plottype="scatterplot",
            s=12,
            regression=False,
            )

    PlotAgent = PlottingAgent({
                "qpu_read" : [networkbatch + "1[0-9]_[0-9]_20.nc_100_200_fullsplit_60_100_1"],
                })

#    PlotAgent = PlottingAgent({"qpu_read" : [regex]})

    PlotAgent.makeFigure("testplot",
            xField="quantumCost",
            yFieldList=["optimizedCost"],
            splitFields=[],
            aggregateMethod='mean'
            )
    PlotAgent.makeFigure("testscatter",
            xField="quantumCost",
            yFieldList=["optimizedCost"],
            constraints={"problemSize" : [14]},
            plottype="scatterplot",
            s=5,
            )
    PlotAgent.makeFigure("testplotsize",
            xField="problemSize",
            yFieldList=["LowestEnergy"],
            constraints={"sample_id" : [1]},
            splitFields=[],
            aggregateMethod='mean',
            errorMethod=deviationOfTheMean,
            )
    PlotAgent.makeFigure("testhistogramm",
            yFieldList=["quantumCost", "optimizedCost"],
            plottype="histogramm",
            bins=100,
            )
    PlotAgent.makeFigure("testcumulative",
            yFieldList=["quantumCost", "optimizedCost"],
            plottype="cumulative",
            bins=100,
            )
    PlotAgent.makeFigure("testdensity",
            yFieldList=["quantumCost", "optimizedCost"],
            plottype="density",
#            constraints={"problemSize" : [14]}
            )
    PlotAgent.makeFigure("testboxplot",
            yFieldList=["quantumCost"],
            plottype="boxplot",
            )

    return

    regex = "*1[0-4]_[0-4]_20.nc_100_200_full*60_200_1"
#    plotGroup(f"scatterplot_nocostinput_100_Anneal_70_chain_strength",
#            "qpu_read",
#            [
#            regex,
#            ],
#            "problemSize",
#            yFieldList = ["cutSamples"],
#            reductionMethod = [extractCutSamples],
#            errorMethod = None,
#            splitFields = ["annealing_time"],
#            plottype="scatterCutSample",
#            logscalex=False,
#            xlabel="energy",
#            ylabel="cost",
#    )
    plotGroup(f"problemsize_to_cost_nocostinput_100_Anneal_70_chain_strength",
            "qpu_read",
            [
            regex,
            ],
            "problemSize",
            yFieldList = ["totalCost"],
            reductionMethod = [np.mean],
            errorMethod = None,
            splitFields = ["annealing_time"],
            plottype="line",
            logscalex=False,
            xlabel="energy",
            ylabel="cost",
    )

    return

    constraints = {
            'annealing_time' : [100],
            'num_reads' : [200],
            'problemSize' : [10,11,12,13,14],
            'chain_strength' : [60],
    }
    plotGroup("scale_to_cost_median",
            "qpu",
            [
            regex,
            regex,
            regex,
            regex,
            ],
            "scale",
            yFieldList = ["totalCost", "LowestFlow","ClosestFlow","cutSamplesCost"],
            splitFields=[],
            logscalex=False,
            reductionMethod=[len]*4,
            errorMethod=deviationOfTheMean,
            constraints=constraints,
            )
    plotGroup("scale_to_cost_mean",
            "qpu",
            [
            regex,
            regex,
            regex,
            regex,
            ],
            "scale",
            yFieldList = ["totalCost", "LowestFlow","ClosestFlow","cutSamplesCost"],
            splitFields=[],
            logscalex=False,
            reductionMethod=[np.mean]*4,
            errorMethod=deviationOfTheMean,
            constraints=constraints,
    )

    return


    constraints = {
            'annealing_time' : [100],
            'num_reads' : [200],
            'problemSize' : [10,11,12,13,14]
    }
    plotGroup("chain_strength_to_cost_mean",
            "qpu",
            [
            regex,
            regex,
            regex,
            regex,
            ],
            "chain_strength",
            yFieldList = ["totalCost", "LowestFlow","ClosestFlow","cutSamplesCost"],
            splitFields=[],
            logscalex=False,
            reductionMethod=[np.mean]*4 ,
            errorMethod=deviationOfTheMean,
            constraints=constraints,
            )
    plotGroup("chain_strength_to_cost_median",
            "qpu",
            [
            regex,
            regex,
            regex,
            regex,
            ],
            "chain_strength",
            yFieldList = ["totalCost", "LowestFlow","ClosestFlow","cutSamplesCost"],
            splitFields=[],
            logscalex=False,
            reductionMethod=[len]*4 ,
            errorMethod=deviationOfTheMean,
            constraints=constraints,
            )

    constraints = {
            'annealing_time' : [100],
            'num_reads' : [200],
            'problemSize' : list(range(10,45,1)),
    }
    plotGroup("size_to_cost_mean",
            "qpu",
            [
            regex,
            regex,
            regex,
            regex,
            ],
            "problemSize",
            yFieldList = ["totalCost", "LowestFlow","ClosestFlow","cutSamplesCost"],
            splitFields=[],
            logscalex=False,
            reductionMethod=[np.mean]*4,
            errorMethod=deviationOfTheMean,
            constraints=constraints,
            )
    plotGroup("size_to_cost_median",
            "qpu",
            [
            regex,
            regex,
            regex,
            regex,
            ],
            "problemSize",
            yFieldList = ["totalCost", "LowestFlow","ClosestFlow","cutSamplesCost"],
            splitFields=[],
            logscalex=False,
            reductionMethod=[len]*4,
            errorMethod=deviationOfTheMean,
            constraints=constraints,
            )

    constraints = {
            'annealing_time' : [10,50,100,200,300,500,1000,1500,2000],
            'num_reads' : [200],
            'chain_strength' : [60],
            'problemSize' : [10,11,12,13,14]
    }
    plotGroup("annealTime_to_cost_mean",
            "qpu",
            [
            regex,
            regex,
            regex,
            regex,
            ],
            "annealing_time",
            yFieldList = ["totalCost", "LowestFlow","ClosestFlow","cutSamplesCost"],
            splitFields=[],
            logscalex=False,
            reductionMethod=[np.mean]*4 ,
            errorMethod=deviationOfTheMean,
            constraints=constraints,
    )
    plotGroup("annealTime_to_cost_median",
            "qpu",
            [
            regex,
            regex,
            regex,
            regex,
            ],
            "annealing_time",
            yFieldList = ["totalCost", "LowestFlow","ClosestFlow","cutSamplesCost"],
            splitFields=[],
            logscalex=False,
            reductionMethod=[len]*4 ,
            errorMethod=deviationOfTheMean,
            constraints=constraints,
    )
    
 
    return

    regex =  "*15_0_20.nc_0.1_8.0_*_fullsplit*"
    plotGroup("trotterslices_to_cost",
              "sqa",
              [
                  regex
              ],
              "optimizationCycles",
              yFieldList=["totalCost"],
              splitFields=["trotterSlices"],
              reductionMethod=[np.mean] ,
              logscalex=False,
              constraints={
                    "optimizationCycles" : [4,6,8,10]
              }
              )
    return

    constraints={"problemSize" : [12]}
    regex= "*Nocost*input_??_*nc*10*_fullsplit*"

    plotGroup("trotterSlices_to_cost_boxplot",
        "sqa",
        [
        regex,
        ],
        "trotterSlices",
        yFieldList = ["totalCost"],
        splitFields=[],
        logscalex=False,
        reductionMethod=[lambda x :x] ,
        constraints=constraints,
        plottype="boxplot",
    )
    return

    plotGroup("trotterSlices_to_cost_mean_split_optCycles",
        "sqa",
        [
        regex,
        ],
        "trotterSlices",
        yFieldList = ["totalCost"],
        splitFields=["optimizationCycles"],
        logscalex=False,
        reductionMethod=[np.mean] ,
        errorMethod=deviationOfTheMean,
        constraints=constraints,
    )

    plotGroup("optimizationCycles_to_totalCost_mean",
        "sqa",
        [
        regex,
        ],
        "optimizationCycles",
        yFieldList = ["totalCost",],
        splitFields=["trotterSlices"],
        logscalex=False,
        reductionMethod=[np.mean],
        errorMethod=deviationOfTheMean,
        constraints=constraints,
    )
    

    return
    



    regex =  "*15_0_20.nc_0.1_8.0_*_fullsplit*"
    plotGroup("trotterslices_to_cost",
              "sqa",
              [
                  regex
              ],
              "optimizationCycles",
              yFieldList=["totalCost"],
              splitFields=["trotterSlices"],
              reductionMethod=[np.mean] ,
              logscalex=False,
              constraints={
                    "optimizationCycles" : [4,6,8,10,12,14,16,20,40,100]
              }
              )

    return

    regex = "*15_0_20.nc_100_200_full*70_200_1"
    plotGroup(f"scatterplot_nocostinput_100_Anneal_70_chain_strength",
            "qpu_read",
            [
            regex,
            ],
            "problemSize",
            yFieldList = ["cutSamples"],
            reductionMethod = [extractCutSamples],
            errorMethod = None,
            splitFields = ["annealing_time"],
            plottype="scatterCutSample",
            logscalex=False,
            xlabel="energy",
            ylabel="cost",
    )

    return



    plotGroup("newising_chain_strength_to_cutSamplesCost",
            "qpu_read",
            [
            "*newising_20_[0]_20.nc_[1]00_365*"
            ],
            "chain_strength",
            yFieldList = ["cutSamplesCost"],
            splitFields=["annealing_time"],
            logscalex=False,
            )

    for chain in [50, 70, 90]:
        regex = f"*newising_20_[0]_20.nc_[1]00_365_30_0_1_{chain}_365_1"
        plotGroup(f"cumulativeCostDistribution_for_100_Anneal_{chain}_chain_strength",
                "qpu_read",
                [
                regex,
                ],
                "problemSize",
                yFieldList = ["sampleValues"],
                reductionMethod = [cumulativeDistribution],
                errorMethod = None,
                splitFields=["annealing_time"],
                plottype="histogramm",
                logscalex=False,
                xlabel="energy",
                ylabel="cost",
        )
        plotGroup(f"scatterplot_new_ising_100_Anneal_{chain}_chain_strength",
                "qpu_read",
                [
                regex,
                ],
                "problemSize",
                yFieldList = ["cutSamples"],
                reductionMethod = [extractCutSamples],
                errorMethod = None,
                splitFields = ["annealing_time"],
                plottype="scatterCutSample",
                logscalex=False,
                xlabel="energy",
                ylabel="cost",
        )
    return

    plotGroup(f"scatterplot_new_ising_365Anneal",
            "qpu_read",
            [
            f"*newising_10_[0]_20.nc_110_365_30_0_1_80_365_1",
            ],
            "problemSize",
            yFieldList = ["cutSamples"],
            reductionMethod = [extractCutSamples],
            errorMethod = None,
            splitFields = ["annealing_time"],
            plottype="scatterCutSample",
            logscalex=False,
            xlabel="energy",
            ylabel="cost",
    )
    return




    # TODO add embeddings for other scales for first plot

    regex = 'embedding_rep_0_ord_1_nocostnewising*.nc.json'
    plotGroup("embedding_size_to_embeddedQubits_newising",
            "qpu",
            [
            regex,
            ],
            xField = "problemSize",
            yFieldList = ["embeddedQubits"],
            splitFields=["scale"],
            PATH=["sweepNetworks"],
            embeddingData = True,
            logscalex=False,
            logscaley=False,
            )
    plotGroup("embedding_size_to_logicalQubits_newising",
            "qpu",
            [
            regex,
            ],
            xField = "problemSize",
            yFieldList = ["logicalQubits"],
            splitFields=[],
            PATH=["sweepNetworks"],
            embeddingData = True,
            logscalex=False,
            logscaley=False,
            )
    plotGroup("embedding_scale_to_embedFactor_newising",
            "qpu",
            [
            regex,
            ],
            xField = "problemSize",
            yFieldList = ["embedFactor"],
            splitFields=[],
            PATH=["sweepNetworks"],
            embeddingData = True,
            logscalex=False,
            logscaley=False,
            )



    regex = "*input_[7-9]_*_20.nc_*_70_0_[01]_250_1"
    constraints={'mangledTotalAnnealTime' : [19,20],
            'chain_strength' : [250],
            'slackVarFactor' : [70.0],
            'maxOrder' : [0,1],
    }
    plotGroup("annealReadRatio_to_cost_mean",
            "qpu",
            [
            regex,
            regex,
            regex,
            ],
            "annealReadRatio",
            yFieldList = ["totalCost", "LowestFlow","ClosestFlow"],
            splitFields=[],
            logscalex=True,
            reductionMethod=[meanOfSquareRoot]*3,
            constraints=constraints,
            )
    plotGroup("annealTime_to_cost_mean",
            "qpu",
            [
            regex,
            regex,
            regex,
            ],
            "annealing_time",
            yFieldList = ["totalCost", "LowestFlow","ClosestFlow"],
            reductionMethod=[meanOfSquareRoot]*3,
            logscalex=True,
            splitFields=[],
            constraints=constraints,
            )

    return



    plotGroup(f"scatterplot_energy_to_optimizedCost_for_anneal_split",
            "qpu_read",
            [
            f"*put_15_[0]_20.nc_5_365_30_0_1_80_365_1",
            f"*put_15_[0]_20.nc_1000_365_30_0_1_80_365_1",
            f"*put_15_[0]_20.nc_2000_365_30_0_1_80_365_1"
            ],
            "problemSize",
            yFieldList = ["cutSamples"]*3,
            reductionMethod = [extractCutSamples]*3,
            errorMethod = None,
            splitFields = ["annealing_time"],
            plottype="scatterCutSample",
            logscalex=False,
            xlabel="energy",
            ylabel="cost",
    )
    return


    for annealTime in [1,5,110,1000,2000]:
        regex = f"*put_15_[0]_20.nc_{annealTime}_365_30_0_1_80_365_1"
        plotGroup(f"scatterplot_energy_to_optimizedCost_for_anneal_{annealTime}",
                "qpu_read",
                [
                regex,
                ],
                "problemSize",
                yFieldList = ["cutSamples"],
                reductionMethod = [extractCutSamples],
                errorMethod = None,
                splitFields = [],
                plottype="scatterCutSample",
                logscalex=False,
                xlabel="energy",
                ylabel="cost",
        )

    regex = '*put_15_[0]_20.nc_110_365_30_0_1_80_365_1'
    plotGroup("cumulativeCostDistribution_for_fullInitialEnergies",
            "qpu_read",
            [
            regex,
            ],
            "problemSize",
            yFieldList = ["cutSamples"],
            reductionMethod = [extractCutSamples],
            errorMethod = None,
            splitFields = [],
            plottype="scatterCutSample",
            logscalex=False,
            xlabel="energy",
            ylabel="cost",
    )

    regex = '*nocostinput_*'
    plotGroup("afterChange_cumulativeCostDistribution_for_fullInitialEnergies",
              "qpu",
              [
                  regex,
              ],
              "problemSize",
              yFieldList=["cutSamples"],
              reductionMethod=[extractCutSamples],
              errorMethod=None,
              splitFields=[],
              plottype="scatterCutSample",
              logscalex=False,
              xlabel="energy",
              ylabel="cost",
              )

    plotGroup(plotname="afterChange_glpk_scale_to_cost_mean",
              solver="pypsa_glpk",
              fileRegexList=[
                  '*nocostinput_*1',
                  '*nocostinput_*1',
              ],
              xField="scale",
              yFieldList=["totalCost"] * 2,
              splitFields=[],
              reductionMethod=[np.mean, np.std],
              constraints={
                  'problemSize': [10, 11, 12, 13, 14],
                  'scale': list(range(10, 45, 5)),
              },
              lineNames=["totalCost", "standard_deviation"],
              logscalex=False,
              logscaley=False,
              )

    return

    plotGroup("costDistribution_for_fullSampleOpt",
              "qpu_read",
              [
                  regex,
              ],
              "problemSize",
              yFieldList=["sampleValues", ],
              reductionMethod=[lambda x: x],
              splitFields=[],
              plottype="histogramm",
              logscalex=False,
              ylabel="count",
              xlabel="sampleValues",
              )
    plotGroup("cumulativeCostDistribution_for_fullSampleOpt",
              "qpu_read",
              [
                  regex,
              ],
              "problemSize",
              yFieldList=["sampleValues", ],
              reductionMethod=[cumulativeDistribution],
              splitFields=[],
              plottype="histogramm",
              logscalex=False,
              ylabel="count",
              xlabel="sampleValues",
              )
    plotGroup("cumulativeCostDistribution_for_fullInitialEnergies",
              "qpu_read",
              [
                  regex,
              ],
              "problemSize",
              yFieldList=[["serial", "vectors", "energy", "data"]],
              reductionMethod=[cumulativeDistribution],
              splitFields=[],
              plottype="histogramm",
              logscalex=False,
              ylabel="count",
              xlabel="energy",
              )

    plotGroup("glpk_scale_to_cost_mean",
              "pypsa_glpk",
              [
                  '*nocostinput_*1',
                  '*nocostinput_*1',
              ],
              xField="scale",
              yFieldList=["totalCost"] * 2,
              splitFields=[],
              reductionMethod=[np.mean, np.std],
              constraints={
                  'problemSize': [10, 11, 12, 13, 14],
                  'scale': list(range(10, 45, 5)),
              },
              lineNames=["totalCost", "standard_deviation"],
              logscalex=False,
              logscaley=False,
              )
    plotGroup("glpk_scale_to_cost_split_size_10_11_12_mean",
              "pypsa_glpk",
              [
                  '*nocostinput_*1',
              ],
              xField="scale",
              yFieldList=["totalCost"],
              splitFields=["problemSize"],
              constraints={
                  'problemSize': [10, 11, 12],
                  'scale': list(range(10, 45, 5)),
              },
              logscalex=False,
              logscaley=False,
              )
    plotGroup("glpk_scale_to_cost_split_size_12_13_14_mean",
              "pypsa_glpk",
              [
                  '*nocostinput_*1',
              ],
              xField="scale",
              yFieldList=["totalCost"],
              splitFields=["problemSize"],
              constraints={
                  'problemSize': [12, 13, 14],
                  'scale': list(range(10, 45, 5)),
              },
              logscalex=False,
              logscaley=False,
              )


    # TODO add embeddings for other scales for first plot
    plotGroup("embedding_size_to_embeddedQubits",
              "qpu",
              [
                  'embedding_rep_0_ord_1_nocostinput*.nc.json'
              ],
              xField="problemSize",
              yFieldList=["embeddedQubits"],
              splitFields=["scale"],
              PATH=["sweepNetworks"],
              embeddingData=True,
              logscalex=False,
              logscaley=False,
              )
    plotGroup("embedding_size_to_logicalQubits",
              "qpu",
              [
                  'embedding_rep_0_ord_1_nocostinput*20.nc.json'
              ],
              xField="problemSize",
              yFieldList=["logicalQubits"],
              splitFields=[],
              PATH=["sweepNetworks"],
              embeddingData=True,
              logscalex=False,
              logscaley=False,
              )
    plotGroup("embedding_scale_to_embedFactor",
              "qpu",
              [
                  'embedding_rep*'
              ],
              xField="scale",
              yFieldList=["embedFactor"],
              splitFields=[],
              PATH=["sweepNetworks"],
              embeddingData=True,
              logscalex=False,
              logscaley=False,
              )
    plotGroup("embedding_size_to_embedFactor",
              "qpu",
              [
                  'embedding_rep_0_ord_1_nocostinput*20.nc.json'
              ],
              xField="problemSize",
              yFieldList=["embedFactor"],
              splitFields=[],
              PATH=["sweepNetworks"],
              embeddingData=True,
              logscalex=False,
              logscaley=False,
              )

    qpu_regex = "*110_365_30_0_1_80_1"
    glpk_regex = "*"
    for strategy in ["totalCost", "LowestFlow", "ClosestFlow"]:
        plotGroup(f"qpu_size_to_{strategy}_even_scale_mean",
                  "f",
                  [
                      qpu_regex,
                  ],
                  "problemSize",
                  splitFields=["scale"],
                  logscalex=False,
                  yFieldList=[strategy],
                  PATH=["results_qpu_sweep"],
                  lineNames=["qpu"],
                  constraints={
                      'problemSize': [10, 11, 12, 13, 14],
                      'scale': [10, 20, 30, 40]
                  },
                  reductionMethod=[np.mean]
                  )
        plotGroup(f"qpu_size_to_{strategy}_odd_scale_mean",
                  "f",
                  [
                      qpu_regex,
                  ],
                  "problemSize",
                  splitFields=["scale"],
                  logscalex=False,
                  yFieldList=[strategy],
                  PATH=["results_qpu_sweep"],
                  lineNames=["qpu"],
                  constraints={
                      'problemSize': [10, 11, 12, 13, 14],
                      'scale': [15, 25, 35, 45]
                  },
                  reductionMethod=[np.mean]
                  )
    plotGroup(f"qpu_glpk_scale_to_cost_mean",
              "qpu",
              [
                  qpu_regex,
                  qpu_regex,
                  qpu_regex,
                  glpk_regex,
              ],
              "scale",
              yFieldList=["totalCost", "LowestFlow", "ClosestFlow", "totalCost", ],
              logscalex=False,
              splitFields=[],
              constraints={
                  'problemSize': [10, 11, 12, 13, 14],
              },
              PATH=[
                  "results_qpu_sweep",
                  "results_qpu_sweep",
                  "results_qpu_sweep",
                  "results_pypsa_glpk_sweep",
              ],
              lineNames=["totalCost", "LowestFlow", "ClosestFlow", "glpk"]
              )

    regex = "*input*30_0_1_80_1 *70_0_1_250_1"
    constraints = {
        'slackVarFactor': [30],
        'chain_strength': [80],
        'problemSize': [15, 16, 17],
        'mangledTotalAnnealTime': [20, 40]
    }
    plotGroup(f"num_reads_to_cost_slack_30_mean",
              "qpu",
              [
                  regex,
                  regex,
                  regex,
              ],
              "num_reads",
              yFieldList=["totalCost", "LowestFlow", "ClosestFlow"],
              logscalex=False,
              splitFields=[],
              constraints=constraints,
              )
    constraints = {
        'slackVarFactor': [70],
        'chain_strength': [250],
        'problemSize': [7, 8, 9],
        'mangledTotalAnnealTime': [20, 40]
    }
    plotGroup(f"num_reads_to_cost_slack_70_mean",
              "qpu",
              [
                  regex,
                  regex,
                  regex,
              ],
              "num_reads",
              yFieldList=["totalCost", "LowestFlow", "ClosestFlow"],
              logscalex=False,
              splitFields=[],
              constraints=constraints,
              )

    regex = '*20.nc_110_365_30_0_1_80_*_1'
    plotGroup("opt_size_to_cost_split_sampleCutSize_mean",
              "qpu_read",
              [
                  regex,
                  "*20.nc*"
              ],
              "problemSize",
              yFieldList=["cutSamplesCost", "totalCost"],
              splitFields=["sampleCutSize"],
              logscalex=False,
              lineNames=['cutSampplesCost', 'glpk'],
              constraints={"sampleCutSize": [1, 2, 5, 10, 30, 100, 365]},
              PATH=[
                  "results_qpu_read_sweep",
                  "results_pypsa_glpk_sweep"
              ],
              )
    plotGroup("opt_size_to_cost_mean",
              "qpu_read",
              [
                  regex,
                  regex,
                  regex,
                  "*20.nc_30_*",
              ],
              "problemSize",
              yFieldList=["totalCost", "LowestFlow", "ClosestFlow", "totalCost"],
              splitFields=[],
              reductionMethod=[np.mean] * 4,
              logscalex=False,
              lineNames=["qpu_totalCost", "LowFlow", "CloseFlow", "glpk_totalCost"],
              PATH=["results_qpu_read_sweep"] * 3 + ["results_pypsa_glpk_sweep"],
              constraints={"sampleCutSize": [1]},
              )
    plotGroup("opt_size_to_cost_median",
              "qpu_read",
              [
                  regex,
                  regex,
                  regex,
                  "*20.nc_30_*",
              ],
              "problemSize",
              yFieldList=["totalCost", "LowestFlow", "ClosestFlow", "totalCost"],
              splitFields=[],
              reductionMethod=[np.median] * 4,
              logscalex=False,
              lineNames=["qpu_totalCost", "LowFlow", "CloseFlow", "glpk_totalCost"],
              PATH=["results_qpu_read_sweep"] * 3 + ["results_pypsa_glpk_sweep"],
              constraints={"sampleCutSize": [1]},
              )

    for slackVar in range(10, 60, 10):
        for chainStrength in range(20, 60, 10):
            regex = f"*input_[7-9]*_20.nc_78_258_{slackVar}_0_1_{chainStrength}_1"
            for strategy in ["totalCost", "LowestFlow", "ClosestFlow"]:
                plotGroup(f"costDistribution_for_{strategy}_slack_{slackVar}_chain_{chainStrength}",
                          "qpu",
                          [
                              regex,
                          ],
                          strategy,
                          yFieldList=[strategy],
                          reductionMethod=[len],
                          logscalex=False,
                          splitFields=[],
                          plottype="histogramm",
                          errorMethod=None,
                          )
    regex = "*input_[7-9]*_20.nc_78_258_[1-5]0_0_1_[2-5]0_1"
    for strategy in ["totalCost", "LowestFlow", "ClosestFlow"]:
        plotGroup(f"costDistribution_for_{strategy}",
                  "qpu",
                  [
                      regex,
                  ],
                  strategy,
                  yFieldList=[strategy],
                  reductionMethod=[len],
                  logscalex=False,
                  splitFields=[],
                  plottype="histogramm",
                  errorMethod=None,
                  )

    regex = "*input_[7-9]_*_20.nc_*_70_0_[01]_250_1"
    constraints = {'mangledTotalAnnealTime': [19, 20],
                   'chain_strength': [250],
                   'slackVarFactor': [70.0],
                   'maxOrder': [0, 1],
                   }
    plotGroup("annealReadRatio_to_cost_mean",
              "qpu",
              [
                  regex,
                  regex,
                  regex,
              ],
              "annealReadRatio",
              yFieldList=["totalCost", "LowestFlow", "ClosestFlow"],
              splitFields=[],
              logscalex=True,
              constraints=constraints,
              )
    plotGroup("annealTime_to_cost_mean",
              "qpu",
              [
                  regex,
                  regex,
                  regex,
              ],
              "annealing_time",
              yFieldList=["totalCost", "LowestFlow", "ClosestFlow"],
              logscalex=True,
              splitFields=[],
              constraints=constraints,
              )

    regex = '*put_[7-9]*70_0_[01]_250_1'
    constraints = {'mangledTotalAnnealTime': [20],
                   'maxOrder': [0, 1],
                   'lineRepresentation': [0],
                   'slackVarFactor': [70.0],
                   'chain_strength': [250],
                   }
    for strategy in ["totalCost", "LowestFlow", "ClosestFlow"]:
        plotGroup(f"anneal_read_ratio_to{strategy}_split_maxOrd_mean",
                  "qpu",
                  [
                      regex,
                  ],
                  "annealReadRatio",
                  yFieldList=[strategy],
                  logscalex=True,
                  logscaley=False,
                  splitFields=["maxOrder", ],
                  constraints=constraints,
                  )
        plotGroup(f"anneal_read_ratio_to{strategy}_split_maxOrd_median",
                  "qpu",
                  [
                      regex,
                  ],
                  "annealReadRatio",
                  yFieldList=[strategy],
                  logscalex=True,
                  logscaley=False,
                  reductionMethod=[np.median],
                  splitFields=["maxOrder", ],
                  constraints=constraints,
                  )

    regex = "*input_[7-9]*_20.nc_78_258_*"
    chainStrengthList = list(range(30, 80, 20)) + [100]
    constraints = {'slackVarFactor': range(10, 50, 10),
                   'chain_strength': chainStrengthList,
                   'num_reads': [258],
                   'annealing_time': [78],
                   }
    plotGroup("slackvar_to_cost_mean",
              "qpu",
              [regex] * 3,
              "slackVarFactor",
              yFieldList=["totalCost", "LowestFlow", "ClosestFlow"],
              reductionMethod=[np.mean] * 3,
              logscalex=False,
              splitFields=[],
              constraints=constraints,
              )
    plotGroup("slackvar_to_cost_split_chains_close_flow_mean",
              "qpu",
              [regex],
              "slackVarFactor",
              yFieldList=["ClosestFlow"],
              reductionMethod=[np.mean],
              logscalex=False,
              splitFields=["chain_strength"],
              constraints=constraints,
              )
    plotGroup("slackvar_to_cost_split_chains_low_flow_mean",
              "qpu",
              [regex],
              "slackVarFactor",
              yFieldList=["LowestFlow"],
              reductionMethod=[np.mean],
              logscalex=False,
              splitFields=["chain_strength"],
              constraints=constraints,
              )
    plotGroup("slackvar_to_cost_split_chains_totalCost_mean",
              "qpu",
              [regex],
              "slackVarFactor",
              yFieldList=["totalCost"],
              reductionMethod=[np.mean],
              logscalex=False,
              splitFields=["chain_strength"],
              constraints=constraints,
              )
    for chainStrength in chainStrengthList:
        constraints["chain_strength"] = [chainStrength]
        plotGroup(f"slackvar_to_cost_chain_{chainStrength}_mean",
                  "qpu",
                  [regex] * 3,
                  "slackVarFactor",
                  yFieldList=["totalCost", "LowestFlow", "ClosestFlow"],
                  reductionMethod=[np.mean] * 3,
                  logscalex=False,
                  splitFields=[],
                  constraints=constraints,
                  )

    regex = "*put_[7-9]_[0-9]_20.nc_78_258_[1-5]*[0-9][0]_1"
    constraints = {'slackVarFactor': range(10, 60, 10),
                   'chain_strength': list(range(30, 100, 10)) + [100],
                   'lineRepresentation': [0],
                   'maxOrder': [1],
                   }
    plotGroup("chain_strength_to_cost_mean",
              "qpu",
              [regex] * 3,
              "chain_strength",
              yFieldList=["totalCost", "LowestFlow", "ClosestFlow"],
              reductionMethod=[np.mean] * 3,
              logscalex=False,
              splitFields=[],
              constraints=constraints,
              )
    plotGroup("chain_strength_to_cost_median",
              "qpu",
              [regex] * 3,
              "chain_strength",
              yFieldList=["totalCost", "LowestFlow", "ClosestFlow"],
              reductionMethod=[np.median] * 3,
              logscalex=False,
              splitFields=[],
              constraints=constraints,
              )
    plotGroup("chain_strength_to_cost_split_slackvar_mean_low_flow",
              "qpu",
              [regex],
              "chain_strength",
              yFieldList=["LowestFlow"],
              reductionMethod=[np.mean],
              logscalex=False,
              splitFields=["slackVarFactor"],
              constraints=constraints,
              )
    plotGroup("chain_strength_to_cost_split_slackvar_mean_close_flow",
              "qpu",
              [regex],
              "chain_strength",
              yFieldList=["ClosestFlow"],
              reductionMethod=[np.mean],
              logscalex=False,
              splitFields=["slackVarFactor"],
              constraints=constraints,
              )
    plotGroup("chain_strength_to_cost_split_slackvar_mean_totalCost",
              "qpu",
              [regex],
              "chain_strength",
              yFieldList=["totalCost"],
              reductionMethod=[np.mean],
              logscalex=False,
              splitFields=["slackVarFactor"],
              constraints=constraints,
              )

    constraints = {'mangledTotalAnnealTime': [20],
                   'annealing_time': [78],
                   'num_reads': [258],
                   'lineRepresentation': [0],
                   'maxOrder': [1],
                   'slackVarFactor': list(range(10, 60, 10)),
                   'chain_strength': [20, 30, 40, 70, 50, 60, 250],
                   }
    plotGroup("SlackVarFactor_to_chain_breaks",
              "qpu",
              [
                  "*input_[789]*[0-9]0_1",
              ],
              "slackVarFactor",
              yFieldList=[["serial", "vectors", "chain_break_fraction", "data"]],
              reductionMethod=[np.mean] * 3,
              logscalex=False,
              splitFields=["chain_strength"],
              constraints=constraints,
              )

    plotGroup("glpk_size_to_time_and_cost_mean",
              "pypsa_glpk",
              [
                  '*20.nc_30*',
                  '*20.nc_30*',
              ],
              xField="problemSize",
              yFieldList=["time", "totalCost"],
              splitFields=[],
              logscalex=False,
              logscaley=False,
              )

    plotGroup("glpk_size_to_cost_mean",
              "pypsa_glpk",
              [
                  '*',
              ],
              xField="problemSize",
              splitFields=[],
              logscalex=False,
              logscaley=False,
              )

    regex = '*input_1[5-7]*20.nc_110_365_30_0_1_80_*_1'
    plotGroup("sampleCutSize_to_cutSamplesCost_mean",
              "qpu_read",
              [
                  regex,
              ],
              "sampleCutSize",
              yFieldList=["cutSamplesCost"],
              splitFields=[],
              constraints={'mangledTotalAnnealTime': [40],
                           'sampleCutSize': list(range(0, 10, 1)) + \
                                            list(range(10, 30, 5)) + [30] + \
                                            list(range(50, 100, 50)) + [100],
                           },
              logscalex=True,
              )

    regex = '*input_1[5-7]*20.nc_*'
    constraints = {'slackVarFactor': [30],
                   'chain_strength': [80],
                   'num_reads': [365],
                   'lineRepresentation': [0],
                   'maxOrder': [1],
                   'sampleCutSize': [100],
                   'annealing_time': [10, 20, 40, 50, 70, 80, 110],
                   'problemSize': [15, 16, 17],
                   }
    plotGroup("annealTime_to_sampleCost_same_reads_mean",
              "qpu_read",
              [
                  regex
              ],
              "annealing_time",
              reductionMethod=[np.mean],
              yFieldList=["cutSamplesCost"],
              splitFields=[],
              constraints=constraints,
              logscalex=False)
    plotGroup("annealTime_to_sampleCost_same_reads_median",
              "qpu_read",
              [
                  regex
              ],
              "annealing_time",
              reductionMethod=[np.median],
              yFieldList=["cutSamplesCost"],
              splitFields=[],
              constraints=constraints,
              logscalex=False)
    plotGroup("annealTime_to_cost_same_reads_mean",
              "qpu_read",
              [
                  regex,
                  regex,
                  regex,
              ],
              "annealing_time",
              yFieldList=["totalCost", "LowestFlow", "ClosestFlow"],
              splitFields=[],
              constraints=constraints,
              logscalex=False)
    plotGroup("annealTime_to_cost_same_reads_median",
              "qpu_read",
              [
                  regex,
                  regex,
                  regex
              ],
              "annealing_time",
              reductionMethod=[np.median] * 3,
              yFieldList=["totalCost", "LowestFlow", "ClosestFlow"],
              splitFields=[],
              constraints=constraints,
              logscalex=False)

    regex = '*input_1[5-7]*20.nc*'
    plotGroup("sampleCutSize_to_cost_split_annealTime_mean",
              "qpu_read",
              [
                  regex,
              ],
              "sampleCutSize",
              reductionMethod=[np.mean],
              yFieldList=["cutSamplesCost"],
              splitFields=["annealing_time"],
              constraints={'mangledTotalAnnealTime': [40],
                           'slackVarFactor': [30.0],
                           'chain_strength': [80],
                           'annealing_time': [50, 110, 125, 250, 2000]}
              )
    plotGroup("sampleCutSize_to_cost_split_annealTime_median",
              "qpu_read",
              [
                  regex,
              ],
              "sampleCutSize",
              reductionMethod=[np.median],
              yFieldList=["cutSamplesCost"],
              splitFields=["annealing_time"],
              constraints={'mangledTotalAnnealTime': [40, 200],
                           'slackVarFactor': [30.0],
                           'chain_strength': [80],
                           'annealing_time': [50, 110, 125, 250]}
              )

    # chr(956) in python3 is \mu
    plotGroup("annealReadRatio_to_access_time",
              "qpu",
              ["*"],
              "annealReadRatio",
              yFieldList=[["serial", "info", "timing", "qpu_access_time"]],
              splitFields=["mangledTotalAnnealTime"],
              constraints={"mangledTotalAnnealTime": [20, 40]},
              ylabel="total qpu access time in " + chr(956) + "s",
              logscalex=True,
              logscaley=True,
              )

    regex = "*20.nc_110_365_30_0_1_80_1"
    constraints = {'slackVarFactor': [30],
                   'chain_strength': [80],
                   'num_reads': [365],
                   'annealing_time': [110],
                   }
    plotGroup("size_to_chain_breaks_mean",
              "qpu",
              [
                  regex,
              ],
              "problemSize",
              yFieldList=[["serial", "vectors", "chain_break_fraction", "data"]],
              reductionMethod=[np.mean],
              logscalex=False,
              splitFields=["chain_strength"],
              )

    # TODO 
    constraints = {"maxOrder": [1, 2, 3, 4]}
    for scale in ["25", "20"]:
        plotGroup(f"sqa_line_representation_to_cost_scale_{scale}_split_maxOrd_mean",
                  "sqa",
                  [
                      f"*_*_{scale}.nc*",
                  ],
                  "lineRepresentation",
                  reductionMethod=[
                      np.mean,
                  ],
                  splitFields=["maxOrder"],
                  logscalex=False,
                  logscaley=False,
                  constraints=constraints,
                  )
        plotGroup(f"sqa_line_representation_to_cost_scale_{scale}_split_maxOrd_median",
                  "sqa",
                  [
                      f"*_*_{scale}.nc*",
                  ],
                  "lineRepresentation",
                  reductionMethod=[
                      np.median,
                  ],
                  splitFields=["maxOrder"],
                  logscalex=False,
                  logscaley=False,
                  constraints=constraints,
                  )

    plotGroup("CostVsTraining_small",
              "sqa",
              ["info_input_10*"],
              xField="steps",
              logscalex=True,
              logscaley=True,
              PATH=["results_sqa_backup"]
              )

    plotGroup("CostVsTraining_medium",
              "sqa",
              ["info_input_25*"],
              xField="steps",
              logscalex=True,
              logscaley=True,
              PATH=["results_sqa_backup"]
              )

    plotGroup("CostVsTraining_large",
              "sqa",
              ["info_input_50*"],
              xField="steps",
              logscalex=True,
              logscaley=True,
              PATH=["results_sqa_backup"]
              )

    plotGroup("classical_small",
              "classical",
              ["info_input_10*"],
              xField="steps",
              logscalex=True,
              logscaley=True,
              PATH=["results_classical"]
              )

    plotGroup("sqa_H_to_cost",
              "sqa",
              ["info_input_10*nc_*"],
              xField="H",
              logscaley=True,
              logscalex=False,
              PATH=["results_sqa_sweep_old"],
              )

    plotGroup("sqa_H_to_cost_with_low_T",
              "sqa",
              ["info_input_10*nc_0.00[01]*"],
              xField="H",
              logscaley=True,
              logscalex=False,
              PATH=["results_sqa_sweep_old"],
              )

    plotGroup("sqa_T_to_all_cost",
              "sqa",
              ["info_input_10*nc_*"],
              xField="T",
              logscaley=True,
              logscalex=False,
              PATH=["results_sqa_sweep_old"],
              )

    plotGroup("sqa_T_to_cost_with_good_H",
              "sqa",
              ["info_input_10*nc_*_[67].0_?"],
              xField="T",
              logscaley=True,
              logscalex=False,
              PATH=["results_sqa_sweep_old"],
              )

    plotGroup(f"sqa_binary_split_approx_comparison",
              "sqa",
              [
                  # "*2[50].nc*",
                  # "*2[50].nc*",
                  "*",
              ],
              "lineRepresentation",
              reductionMethod=[
                  np.mean,
              ],
              lineNames=[
                  "mean",
              ],
              splitFields=["maxOrder"],
              logscalex=False,
              logscaley=False,
              )
    return


if __name__ == "__main__":
    main()
