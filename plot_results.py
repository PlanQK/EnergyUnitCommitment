import glob
import json
import collections
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def makeFig(plotInfo, outputFile, logscalex=False, logscaley=False):
    fig, ax = plt.subplots()
    # fig.set_size_inches((20, 20))
    for key, values in plotInfo.items():
        ax.plot([e[0] for e in values], [e[1] for e in values], label=key)
    plt.legend()
    if logscalex:
        ax.set_xscale("log")
    if logscaley:
        ax.set_yscale("log")

    fig.savefig(outputFile)


def extractPlottableInformation(
    fileRegex, xField, yField, splitFields=[], reductionMethod=np.mean
):
    """Transform the json data by averaging the yName values for each xName value.
    If splitFields is given generate multiple Lines. The reduction method needs to
    reduce a list of multiple values into one value (e.g. np.mean, max, min)
    """
    plotData = collections.defaultdict(collections.defaultdict)
    for fileName in glob.glob(fileRegex):
        with open(fileName) as file:
            fileName = fileName.split("/")[-1]
            element = json.load(file)
            element["fileName"] = "_".join(fileName.split("_")[1:])
            element["problemSize"] = fileName.split("_")[2]
            key = tuple(
                e
                for e in [
                    splitField + "=" + str(element[splitField])
                    for splitField in splitFields
                ]
            )
            if element[xField] not in plotData[key]:
                plotData[key][element[xField]] = []
            plotData[key][element[xField]].append(element[yField])
    # now perform reduction
    result = collections.defaultdict(list)
    for outerKey in plotData:
        for innerKey in plotData[outerKey]:
            result[outerKey].append(
                [
                    float(innerKey),
                    reductionMethod(plotData[outerKey][innerKey]),
                ]
            )
        result[outerKey].sort()
    return result


def main():
    # first collect all data
    plt.style.use("seaborn")

    plotInfo = {}
    for key, value in extractPlottableInformation(
        "results_sqa_backup/info_input_10*",
        xField="steps",
        yField="totalCost",
        splitFields=["problemSize"],
    ).items():
        plotInfo[f"sqa_{key}"] = value
    makeFig(
        plotInfo,
        "plots/CostVsTraining_small.pdf",
        logscalex=True,
    )

    plotInfo = {}
    for key, value in extractPlottableInformation(
        "results_classical/info_input_10*",
        xField="steps",
        yField="totalCost",
        splitFields=["problemSize"],
    ).items():
        plotInfo[f"classical_{key}"] = value
    makeFig(
        plotInfo,
        "plots/classical_small.pdf",
        logscalex=True,
    )

    plotInfo = extractPlottableInformation(
        "results_sqa_backup/info_input_25*",
        xField="steps",
        yField="totalCost",
        splitFields=["problemSize"],
    )
    makeFig(plotInfo, "plots/CostVsTraining_medium.pdf", logscalex=True)

    plotInfo = extractPlottableInformation(
        "results_sqa_backup/info_input_50*",
        xField="steps",
        yField="totalCost",
        splitFields=["problemSize"],
    )
    makeFig(plotInfo, "plots/CostVsTraining_large.pdf", logscalex=True)


if __name__ == "__main__":
    main()
