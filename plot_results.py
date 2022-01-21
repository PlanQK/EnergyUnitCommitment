import glob
import json
import collections
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from os import path, getenv

RESULT_SUFFIX = "sweep"
PRINT_NUM_READ_FILES = False
FILEFORMAT = "png"
BINSIZE = 1
PLOTLIMIT = 500
# export computing cost rate
COSTPERHOUR = getenv('costperhour')
 

def meanOfSquareRoot(values):
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


def cumulativeDistribution(values: list) -> list:
    result = []
    maxVal = max(values[0])
    for valueLists in values:
        curMax = max(valueLists)
        if curMax > maxVal:
            maxVal = curMax
    maxVal += 1

    for valueLists in values:
        for val in valueLists:
            result += list(range(int(val), int(maxVal), 1))
    return [result]


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


def extractCutSamples(cutSamplesDictList: object) -> [list, list]:
    """
    Used to create scatter plots from a single run.

    @param cutSamplesDictList: object
        List of dicts to be searched through.
    @return: [list, list]
        A list with the extracted energy and one with the optimized costs.
    """
    energy = []
    optimizedCost = []
    for cutSamplesDict in cutSamplesDictList:
        for value in cutSamplesDict.values():
            energy.append(value["energy"])
            optimizedCost.append(value["optimizedCost"])
    return energy, optimizedCost


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

        # default plot type of a function graph
        if plottype == "line":
            sortedValues = sorted(values)
            ax.errorbar([e[0] for e in sortedValues], [e[1] for e in sortedValues], label=key,
                        yerr=[e[2] / 2.0 for e in sortedValues])

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


def extractInformation(fileRegex: str, xField: str, yField: str,
                       splitFields: list = ["problemSize"], reductionMethod=np.mean,
                       errorMethod=deviationOfTheMean, constraints: dict = {}, embedding: bool = False) -> dict:
    """
    Extracts the information to be plotted.

    @param fileRegex: str
        A regular expressions to get the data
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

    for key, values in constraints.items():
        try:
            # value of constraint is not in constrains list
            if float(resolveKey(element, key)) not in values:
                break
        except KeyError:
            pass
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

        plotData[key][element[xField]].append(yvalue)
        filesRead += 1

    # now perform reduction
    result = collections.defaultdict(list)
    for outerKey in plotData:
        for innerKey in plotData[outerKey]:
            if isinstance(innerKey, str):
                # H, T in sqa data
                if innerKey.startswith("["):
                    xvalue = innerKey[1:-1].split(',')[0]
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
        searched for in a standard folder.
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
            ]
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
