import glob
import json
import collections
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

RESULT_SUFFIX = "sweep"
PRINT_NUM_READ_FILES = False
FILEFORMAT = "png"

def deviationOfTheMean(values : list) -> float :
    return np.std(values)/np.sqrt(len(values))

def cumulativeDistribution(values : list , rightLimit=150):
    result = []
    maxVal = max(values[0])
    for valueLists in values:
        curMax = max(valueLists)
        if curMax > maxVal:
            maxVal = curMax
    maxVal += 1
    
    for valueLists in values:
        for val in valueLists:
            result += list(range(int(val),int(maxVal),1))
    return [result]

def averageOfBetterThanMedian(values : list) -> float:
    median = np.median(values)
    result = 0
    count = 0
    for val in values:
        if val > median:
            continue
        count += 1
        result =+ val
    return float(result)/count

def averageOfBest(values : list) -> float:
    values.sort()
    return np.mean(values[:-1])

def averageOfBestPercent(values : list, percentage : float) -> float:
    values.sort()
    return np.mean(values[:int(percentage*len(values))])

    
    
def makeFig(plotInfo, outputFile, 
        logscalex=False, logscaley=False, xlabel=None, ylabel=None, title=None, 
        fileformat="pdf", histogramm=False, rightLimit=500):
    fig, ax = plt.subplots()
    # fig.set_size_inches((20, 20))
    for key, values in plotInfo.items():
        if histogramm:
            # if condition is truthy if the function values have not been reduced earlier. thus in 'yvalues' we have a list of
            # values that go in to the histogramm with weight 1.
            # if the condition is falsy, the xField should be the yvalues we want to plot, using arbitrary yvalues that
            # get reduced by len, thus counting how often a particular value appears in xField
            if hasattr(values[0][1],"__getitem__"):
                for entries in values:
                    flattenedList = [ item for sublist in entries[1] for item in sublist]
                    if not flattenedList:
                        continue
                    ax.hist( flattenedList ,bins=[i for i in range(int(min(flattenedList))-2,int(max(flattenedList))+2,1)], label=key) 
            else:
                sortedValues = sorted(values)
                ax.hist([e[0]  for e in values],bins=[i for i in range(0,rightLimit,1)],  label=key, weights=[e[1]  for e in sortedValues] )
        else:
            sortedValues = sorted(values)
            ax.errorbar([e[0] for e in sortedValues], [e[1] for e in sortedValues], label=key, yerr=[e[2]/2.0 for e in sortedValues] )
    plt.legend()
    if logscalex:
        ax.set_xscale("log")
    if logscaley:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.suptitle(title)
    if fileformat == "":
        fig.savefig(outputFile +".png")
        fig.savefig(outputFile +".pdf")
    else:
        fig.savefig(outputFile +"."+fileformat)


def extractEmbeddingInformation(
    fileRegex, xField, yField, splitFields=["problemSize"], reductionMethod=np.mean,errorMethod=deviationOfTheMean
):
    plotData = collections.defaultdict(collections.defaultdict)
    for fileName in glob.glob(fileRegex):
        with open(fileName) as file:
            fileName = fileName.split("/")[-1]
            element = json.load(file)
            logicalQubits = [int(key) for key in element.keys()]
            embeddedQubits = [item for sublist in element.values() for item in sublist ]

            element["embeddedQubits"] = len(embeddedQubits)
            element["logicalQubits"] = len(logicalQubits)
            element["embedFactor"] = float(element["embeddedQubits"]) / float(element["logicalQubits"])

            element["fileName"] = "_".join(fileName.split("_")[5:])[:-5]
            element["problemSize"] = fileName.split("_")[6]
            element["scale"] = fileName.split("_")[8][:-8]
            element["rep"] = fileName.split("_")[2]
            element["ord"] = fileName.split("_")[4]
            key = tuple(
                e
                for e in [
                    splitField + "=" + str(element[splitField])
                    for splitField in splitFields
                ]
            )
            if element[xField] not in plotData[key]:
                plotData[key][element[xField]] = []

            yvalue = element[yField]

            plotData[key][element[xField]].append(float(yvalue))
    
    # now perform reduction
    result = collections.defaultdict(list)
    for outerKey in plotData:
        for innerKey in plotData[outerKey]:
            xvalue = innerKey   
            result[outerKey].append(
                [
                    float(xvalue),
                    reductionMethod(plotData[outerKey][innerKey]),
                    errorMethod(plotData[outerKey][innerKey]),
                ]
            )
        result[outerKey].sort()
    return result


def extractPlottableInformation(
    fileRegex, xField, yField, 
    splitFields=["problemSize"], reductionMethod=np.mean,errorMethod=deviationOfTheMean,
    constraints={}
):
    """Transform the json data by averaging the yName values for each xName value.
    If splitFields is given generate multiple Lines. The reduction method needs to
    reduce a list of multiple values into one value (e.g. np.mean, max, min)
    """
    plotData = collections.defaultdict(collections.defaultdict)
    filesRead = 0
    for fileName in glob.glob(fileRegex):
        with open(fileName) as file:
            fileName = fileName.split("/")[-1]
            element = json.load(file)
            element["fileName"] = "_".join(fileName.split("_")[1:])
            element["problemSize"] = fileName.split("_")[2]
            element["scale"] = fileName.split("_")[4][:-3]
            if "cutSamples" in element:
                element["sampleCutSize"] = len(element["cutSamples"])
                element["sampleValues"] = [
                        element["cutSamples"][key]["optimizedCost"]
                        for key in element["cutSamples"].keys()
                ]
                element["dummyWeights"] = [1 for _ in element["sampleValues"]]


            if yField == "minChoice":
                element["minChoice"] = min(element["LowestFlow"], element["ClosestFlow"])

            try:
                element["maxOrder"] = fileName.split("_")[9]
            except IndexError:
                element["maxOrder"] = 0


            # if a constraint is broken, don't add the current files info.
            # else block is default execution path so it works with empty contraints
            for key,values in constraints.items():
                try:
                    if float(element[key]) not in values:
                        break
                except KeyError:
                    pass
            else:
                key = tuple(
                    e
                    for e in [
                        splitField + "=" + str(element.get(splitField) or "")
                        for splitField in splitFields
                    ]
                )

                try:
                    if element[xField] not in plotData[key]:
                        plotData[key][element[xField]] = []

                    # unpacking yvalue if it is in nested dict
                    if isinstance(yField,list):
                        yvalue = element
                        for dict_key in yField:
                            yvalue = yvalue[dict_key]
                    else:
                        yvalue = element[yField]

                    try:
                        plotData[key][element[xField]].append(float(yvalue))
                    # yvalue has not been reduced into a value and is still a list
                    except TypeError:
                        plotData[key][element[xField]].append([float(y) for y in yvalue])
                except KeyError:
                    pass
            filesRead += 1
                
    # now perform reduction
    result = collections.defaultdict(list)
    for outerKey in plotData:
        for innerKey in plotData[outerKey]:
            # extract numeric value in string innerKey
            if hasattr(innerKey, "__getitem__"):
                if innerKey == "":
                    xvalue = -1
                elif innerKey[0] == "[":
                    xvalue = innerKey[1:-1].split(',')[0]
                else:
                    xvalue = innerKey   
            else:
                xvalue = innerKey

            result[outerKey].append(
                [
                    float(xvalue),
                    reductionMethod(plotData[outerKey][innerKey]),
                    errorMethod(plotData[outerKey][innerKey]) if errorMethod is not None else 0
                ]
            )
        result[outerKey].sort()

    if PRINT_NUM_READ_FILES:
        print(f"files read for {fileRegex} : {filesRead}")
    return result



def plotGroup(
    plotname, solver, fileRegexList, xField, 
    yFieldList=None, splitFields=["problemSize"], logscalex=True, logscaley=False,
    PATH=None, reductionMethod=None, lineNames=None, embeddingData=False, errorMethod=deviationOfTheMean,
    constraints={},histogramm=False,xlabel=None,ylabel=None
    ):
    """
    extracts data from all files in the regexList list and plots it into a single
    plot as {plotname}.{FILEFORMAT}. fileInfo are json files

    plotname -- filename of plot without filetype
    solver -- string to indicate which solver was used. sets some helpful default values
    fileRegexList -- List of all regular expressions to get data. Each item in the list is
            plotted as it's own line (and further split up if splitFields is not empty)
            a whitespace in a regex string splits multiple regular expressions to be
            plotted into a single line
    PATH -- PATH to plot data
    xField -- name of x-axis key 
    yFieldList -- list of keys for y-axis values. The i-th entry is for the i-th regex
            in fileRegexList. A list of keys unpacks the value in a nested dictionary
    reductionMethod -- function to reduce list of values into a float value
    errorMethod -- complement to reductionMethod to plot error bars
    constraints -- dictionary of values that a .json file has to have to be included
            in the plot. A dict key is the name of the value, and the dict value
            has to be a list of all permissible values. If a .json file doesn't have a key,
            that constraint is ignored
    """

    if yFieldList is None:
        yFieldList=["totalCost"]*len(fileRegexList)

    if len(fileRegexList) != len(yFieldList):
        print("number of regex doesn't match number of yField's selected")
        return

    if PATH is None:
        PATH=[f"results_{solver}_{RESULT_SUFFIX}"]*len(fileRegexList)

    if reductionMethod is None:
        reductionMethod = [np.mean]*len(fileRegexList)

    if lineNames is None:
        lineNames=yFieldList

    if xlabel is None:
        xlabel = xField

    if ylabel is None:
        ylabel = yFieldList

    plotInfo = {}
    for idx in range(len(fileRegexList)):
        for regex in fileRegexList[idx].split():
            if embeddingData:
                iterator = extractEmbeddingInformation(
                        f"{PATH[idx]}/{regex}",
                        xField=xField,
                        yField=yFieldList[idx],
                        splitFields=splitFields,
                        reductionMethod=reductionMethod[idx],
                        errorMethod=errorMethod,
                ).items()
            else:
                iterator = extractPlottableInformation(
                        f"{PATH[idx]}/{regex}",
                        xField=xField,
                        yField=yFieldList[idx],
                        splitFields=splitFields,
                        reductionMethod=reductionMethod[idx],
                        errorMethod=errorMethod,
                        constraints=constraints,
                ).items()

            for key, value in iterator:
                plotInfoKey = f"{solver}_{key}_{lineNames[idx]}"
                if  plotInfoKey in plotInfo:
                    plotInfo[plotInfoKey] += value
                else:
                    plotInfo[plotInfoKey] = value
    makeFig(
        plotInfo,
        f"plots/{plotname}",
        fileformat=FILEFORMAT ,
        logscalex=logscalex,
        logscaley=logscaley,
        xlabel=xlabel,
        ylabel=ylabel,
        histogramm=histogramm
#        title=''
    )


def main():

    plt.style.use("seaborn")


    regex = '*put_15_0_20.nc_110_365_30_0_1_80_365_1'
    plotGroup("costDistribution_for_fullSampleOpt",
            "qpu_read",
            [
            regex,
            ],
            "problemSize",
            yFieldList = ["sampleValues",],
            reductionMethod = [lambda x:x],
            splitFields = [],
            histogramm=True,
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
            yFieldList = ["sampleValues",],
            reductionMethod = [cumulativeDistribution],
            splitFields = [],
            histogramm=True,
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
            yFieldList = [["serial",  "vectors" ,"energy", "data"]],
            reductionMethod = [cumulativeDistribution],
            splitFields = [],
            histogramm=True,
            logscalex=False,
            ylabel="count",
            xlabel="energy",
    )

            
    plotGroup("glpk_scale_to_cost_mean",
            "pypsa_glpk",
            [ 
              '*nocostinput_*1',  
            ],
            xField = "scale",
            yFieldList = ["totalCost"],
            splitFields=[],
            constraints={
                    'problemSize' : [10,11,12,13,14],
                    'scale' : list(range(10,45,5)),
            },
            logscalex=False,
            logscaley=False,
    )
    plotGroup("glpk_scale_to_cost_split_size_10_11_12_mean",
            "pypsa_glpk",
            [ 
              '*nocostinput_*1',  
            ],
            xField = "scale",
            yFieldList = ["totalCost"],
            splitFields=["problemSize"],
            constraints={
                    'problemSize' : [10,11,12],
                    'scale' : list(range(10,45,5)),
            },
            logscalex=False,
            logscaley=False,
    )
    plotGroup("glpk_scale_to_cost_split_size_12_13_14_mean",
            "pypsa_glpk",
            [ 
              '*nocostinput_*1',  
            ],
            xField = "scale",
            yFieldList = ["totalCost"],
            splitFields=["problemSize"],
            constraints={
                    'problemSize' : [12,13,14],
                    'scale' : list(range(10,45,5)),
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
            xField = "problemSize",
            yFieldList = ["embeddedQubits"],
            splitFields=["scale"],
            PATH=["sweepNetworks"],
            embeddingData = True,
            logscalex=False,
            logscaley=False,
            )
    plotGroup("embedding_size_to_logicalQubits",
            "qpu",
            [
                'embedding_rep_0_ord_1_nocostinput*20.nc.json'
            ],
            xField = "problemSize",
            yFieldList = ["logicalQubits"],
            splitFields=[],
            PATH=["sweepNetworks"],
            embeddingData = True,
            logscalex=False,
            logscaley=False,
            )
    plotGroup("embedding_scale_to_embedFactor",
            "qpu",
            [
                'embedding_rep*'
            ],
            xField = "scale",
            yFieldList = ["embedFactor"],
            splitFields=[],
            PATH=["sweepNetworks"],
            embeddingData = True,
            logscalex=False,
            logscaley=False,
            )
    plotGroup("embedding_size_to_embedFactor",
            "qpu",
            [
                'embedding_rep_0_ord_1_nocostinput*20.nc.json'
            ],
            xField = "problemSize",
            yFieldList = ["embedFactor"],
            splitFields=[],
            PATH=["sweepNetworks"],
            embeddingData = True,
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
                yFieldList = [strategy],
                PATH=["results_qpu_sweep"],
                lineNames=["qpu"],
                constraints={
                        'problemSize' : [10,11,12,13,14],
                        'scale' : [10,20,30,40]
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
                yFieldList = [strategy],
                PATH=["results_qpu_sweep"],
                lineNames=["qpu"],
                constraints={
                        'problemSize' : [10,11,12,13,14],
                        'scale' : [15,25,35,45]
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
            yFieldList = ["totalCost", "LowestFlow", "ClosestFlow", "totalCost",],
            logscalex=False,
            splitFields=[],
            constraints={
                    'problemSize' : [10,11,12,13,14],
            },
            PATH=[
                "results_qpu_sweep",
                "results_qpu_sweep",
                "results_qpu_sweep",
                "results_pypsa_glpk_sweep",
                ],
            lineNames = ["totalCost", "LowestFlow", "ClosestFlow", "glpk"]
    )


    regex = "*input*30_0_1_80_1 *70_0_1_250_1"
    constraints={
        'slackVarFactor' : [30],
        'chain_strength' : [80],
        'problemSize' : [15,16,17],
        'mangledTotalAnnealTime' : [20,40]
    }
    plotGroup(f"num_reads_to_cost_slack_30_mean",
            "qpu",
            [
            regex,
            regex,
            regex,
            ],
            "num_reads",
            yFieldList = ["totalCost", "LowestFlow","ClosestFlow"],
            logscalex=False,
            splitFields=[],
            constraints=constraints,
    )
    constraints={
        'slackVarFactor' : [70],
        'chain_strength' : [250],
        'problemSize' : [7,8,9],
        'mangledTotalAnnealTime' : [20,40]
    }
    plotGroup(f"num_reads_to_cost_slack_70_mean",
            "qpu",
            [
            regex,
            regex,
            regex,
            ],
            "num_reads",
            yFieldList = ["totalCost", "LowestFlow","ClosestFlow"],
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
            yFieldList = ["cutSamplesCost", "totalCost"],
            splitFields=["sampleCutSize"],
            logscalex=False,
            lineNames=['cutSampplesCost','glpk'],
            constraints={"sampleCutSize" : [1,2,5,10,30,100,365]},
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
            yFieldList = ["totalCost", "LowestFlow","ClosestFlow", "totalCost"],
            splitFields=[],
            reductionMethod=[np.mean]*4,
            logscalex=False,
            lineNames=["qpu_totalCost", "LowFlow","CloseFlow", "glpk_totalCost"],
            PATH=["results_qpu_read_sweep"]*3+["results_pypsa_glpk_sweep"],
            constraints={"sampleCutSize" : [1]},
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
            yFieldList = ["totalCost", "LowestFlow","ClosestFlow", "totalCost"],
            splitFields=[],
            reductionMethod=[np.median]*4,
            logscalex=False,
            lineNames=["qpu_totalCost", "LowFlow","CloseFlow", "glpk_totalCost"],
            PATH=["results_qpu_read_sweep"]*3+["results_pypsa_glpk_sweep"],
            constraints={"sampleCutSize" : [1]},
            )


    for slackVar in range(10,60,10):
        for chainStrength in range(20,60,10):
            regex = f"*input_[7-9]*_20.nc_78_258_{slackVar}_0_1_{chainStrength}_1"
            for strategy in ["totalCost", "LowestFlow", "ClosestFlow"]:
                plotGroup(f"costDistribution_for_{strategy}_slack_{slackVar}_chain_{chainStrength}",
                        "qpu",
                        [
                        regex,
                        ],
                        strategy,
                        yFieldList = [strategy],
                        reductionMethod=[len] ,
                        logscalex=False,
                        splitFields=[],
                        histogramm=True,
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
                yFieldList = [strategy],
                reductionMethod=[len] ,
                logscalex=False,
                splitFields=[],
                histogramm=True,
                errorMethod=None,
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
            logscalex=True,
            splitFields=[],
            constraints=constraints,
            )


    regex = '*put_[7-9]*70_0_[01]_250_1'
    constraints={'mangledTotalAnnealTime' : [20],
            'maxOrder' : [0,1],
            'lineRepresentation' : [0],
            'slackVarFactor' : [70.0],
            'chain_strength' : [250],
    }
    for strategy in ["totalCost", "LowestFlow", "ClosestFlow"]:
        plotGroup(f"anneal_read_ratio_to{strategy}_split_maxOrd_mean",
                "qpu",
                [
                regex,
                ],
                "annealReadRatio",
                yFieldList = [strategy],
                logscalex=True,
                logscaley=False,
                splitFields=["maxOrder",],
                constraints=constraints,
        )
        plotGroup(f"anneal_read_ratio_to{strategy}_split_maxOrd_median",
                "qpu",
                [
                regex,
                ],
                "annealReadRatio",
                yFieldList = [strategy],
                logscalex=True,
                logscaley=False,
                reductionMethod=[np.median],
                splitFields=["maxOrder",],
                constraints=constraints,
        )


    regex="*input_[7-9]*_20.nc_78_258_*"
    constraints={'slackVarFactor' : range(10, 50 , 10),
                'chain_strength' : list(range(30, 70, 20)) + [100],
                'num_reads' : [258],
                'annealing_time' : [78],
                }
    plotGroup("slackvar_to_cost_mean",
            "qpu",
            [regex]*3,
            "slackVarFactor",
            yFieldList = ["totalCost", "LowestFlow","ClosestFlow"],
            reductionMethod=[np.mean]*3 ,
            logscalex=False,
            splitFields=[],
            constraints=constraints,
            )
    plotGroup("slackvar_to_cost_split_chains_close_flow_mean",
            "qpu",
            [regex],
            "slackVarFactor",
            yFieldList = ["ClosestFlow"],
            reductionMethod=[np.mean] ,
            logscalex=False,
            splitFields=["chain_strength"],
            constraints=constraints,
            )
    plotGroup("slackvar_to_cost_split_chains_low_flow_mean",
            "qpu",
            [regex],
            "slackVarFactor",
            yFieldList = ["LowestFlow"],
            reductionMethod=[np.mean] ,
            logscalex=False,
            splitFields=["chain_strength"],
            constraints=constraints,
            )
    plotGroup("slackvar_to_cost_split_chains_totalCost_mean",
            "qpu",
            [regex],
            "slackVarFactor",
            yFieldList = ["totalCost"],
            reductionMethod=[np.mean] ,
            logscalex=False,
            splitFields=["chain_strength"],
            constraints=constraints,
            )


    regex="*put_[7-9]_[0-9]_20.nc_78_258_[1-5]*[0-9][0]_1"
    constraints={'slackVarFactor' : range(10, 60 , 10) ,
                'chain_strength' : list(range(30, 100, 10)) + [100] ,
                'lineRepresentation' : [0],
                'maxOrder' : [1],
                }
    plotGroup("chain_strength_to_cost_mean",
            "qpu",
            [regex]*3,
            "chain_strength",
            yFieldList = ["totalCost", "LowestFlow","ClosestFlow"],
            reductionMethod=[np.mean]*3 ,
            logscalex=False,
            splitFields=[],
            constraints=constraints,
            )
    plotGroup("chain_strength_to_cost_median",
            "qpu",
            [regex]*3,
            "chain_strength",
            yFieldList = ["totalCost", "LowestFlow","ClosestFlow"],
            reductionMethod=[np.median]*3 ,
            logscalex=False,
            splitFields=[],
            constraints=constraints,
            )
    plotGroup("chain_strength_to_cost_split_slackvar_mean_low_flow",
            "qpu",
            [regex],
            "chain_strength",
            yFieldList = ["LowestFlow"],
            reductionMethod=[np.mean] ,
            logscalex=False,
            splitFields=["slackVarFactor"],
            constraints=constraints,
            )
    plotGroup("chain_strength_to_cost_split_slackvar_mean_close_flow",
            "qpu",
            [regex],
            "chain_strength",
            yFieldList = ["ClosestFlow"],
            reductionMethod=[np.mean] ,
            logscalex=False,
            splitFields=["slackVarFactor"],
            constraints=constraints,
            )
    plotGroup("chain_strength_to_cost_split_slackvar_mean_totalCost",
            "qpu",
            [regex],
            "chain_strength",
            yFieldList = ["totalCost"],
            reductionMethod=[np.mean],
            logscalex=False,
            splitFields=["slackVarFactor"],
            constraints=constraints,
            )


    constraints={'mangledTotalAnnealTime' : [20],
            'annealing_time' : [78],
            'num_reads' : [258],
            'lineRepresentation' : [0],
            'maxOrder' : [1],
            'slackVarFactor' : list(range(10,60,10)),
            'chain_strength' : [20,30,40,70,50,60,250],
    }
    plotGroup("SlackVarFactor_to_chain_breaks",
            "qpu",
            [
            "*input_[789]*[0-9]0_1",
            ],
            "slackVarFactor",
            yFieldList = [["serial",  "vectors" ,"chain_break_fraction", "data"]],
            reductionMethod=[np.mean]*3 ,
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
            xField = "problemSize",
            yFieldList = ["time", "totalCost"],
            splitFields=[],
            logscalex=False,
            logscaley=False,
    )


    plotGroup("glpk_size_to_cost_mean",
            "pypsa_glpk",
            [ 
              '*',  
            ],
            xField = "problemSize",
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
            yFieldList = ["cutSamplesCost"],
            splitFields=[],
            constraints={'mangledTotalAnnealTime' : [40],
                    'sampleCutSize' : list(range(0,10,1)) + \
                            list(range(10,30,5)) + [30] + \
                            list(range(50,100,50)) + [100] ,
                    },
            logscalex=True,
            )

    regex = '*input_1[5-7]*20.nc_*'
    constraints={'slackVarFactor' : [30],
                'chain_strength' : [80],
                'num_reads' : [365],
                'lineRepresentation' : [0],
                'maxOrder' : [1],
                'sampleCutSize' : [100],
                'annealing_time' : [10, 20, 40, 50, 70, 80 ,110],
                'problemSize' : [15,16,17],
                }
    plotGroup("annealTime_to_cost_same_reads_mean",
            "qpu_read",
            [
            regex
            ],
            "annealing_time",
            reductionMethod = [np.mean],
            yFieldList = ["cutSamplesCost"],
            splitFields = [],
            constraints=constraints,
            logscalex=False)
    plotGroup("annealTime_to_cost_same_reads_median",
            "qpu_read",
            [
            regex
            ],
            "annealing_time",
            reductionMethod = [np.median],
            yFieldList = ["cutSamplesCost"],
            splitFields = [],
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
            yFieldList = ["cutSamplesCost"],
            splitFields = ["annealing_time"],
            constraints={'mangledTotalAnnealTime' : [40],
                    'slackVarFactor' : [30.0],
                    'chain_strength' : [80],
                    'annealing_time' : [50,110,125,250,2000]}
            )
    plotGroup("sampleCutSize_to_cost_split_annealTime_median",
            "qpu_read",
            [
            regex,
            ],
            "sampleCutSize",
            reductionMethod=[np.median],
            yFieldList = ["cutSamplesCost"],
            splitFields = ["annealing_time"],
            constraints={'mangledTotalAnnealTime' : [40,200],
                    'slackVarFactor' : [30.0],
                    'chain_strength' : [80],
                    'annealing_time' : [50,110,125,250]}
            )


    # chr(956) in python3 is \mu
    plotGroup("annealReadRatio_to_access_time",
            "qpu",
            ["*"],
            "annealReadRatio",
            yFieldList = [ ["serial", "info", "timing" , "qpu_access_time"] ],
            splitFields=["mangledTotalAnnealTime"],
            constraints={"mangledTotalAnnealTime" : [20,40]},
            ylabel = "total qpu access time in " + chr(956) + "s",
            )


    regex= "*20.nc_110_365_30_0_1_80_1"
    constraints={'slackVarFactor' : [30],
                'chain_strength' : [80],
                'num_reads' : [365],
                'annealing_time' : [110],
                }
    plotGroup("size_to_chain_breaks_mean",
            "qpu",
            [
            regex,
            ],
            "problemSize",
            yFieldList = [["serial",  "vectors" ,"chain_break_fraction", "data"]],
            reductionMethod=[np.mean] ,
            logscalex=False,
            splitFields=["chain_strength"],
            )

    # TODO 
    constraints = {"maxOrder" : [1,2,3,4]}
    for scale in ["25","20"]:
        plotGroup(f"sqa_line_representation_to_cost_scale_{scale}_split_maxOrd_mean",
            "sqa",
            [
            f"*_*_{scale}.nc*",
            ],
            "lineRepresentation",
            reductionMethod=[
                    np.mean,
            ],
            splitFields = ["maxOrder"],
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
            splitFields = ["maxOrder"],
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
        #"*2[50].nc*",
        #"*2[50].nc*",
        "*",
        ],
        "lineRepresentation",
        reductionMethod=[
                np.mean,
        ],
        lineNames = [
                "mean",
        ],
        splitFields = ["maxOrder"],
        logscalex=False,
        logscaley=False,
        )
    return

if __name__ == "__main__":
    main()
