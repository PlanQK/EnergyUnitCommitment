import itertools
import operator

import matplotlib.pyplot as plt
import numpy
from matplotlib import rcParams
import numpy as np
import json
import random

from numpy import median, linalg
from qiskit.circuit import Parameter

from statistics import mean
from scipy.optimize import curve_fit


def openFile(filename: str, directory: str) -> dict:
    """
    Opens the given json file and returns its content as a dictionary.
    Args:
        filename: (str) The name of the file to be opened.
        directory: (str) The folder in which the file is located. Default: "results_qaoa_sweep/"

    Returns:
        data (dict) The content of the file with the given filename from the given directory.
    """
    with open(f"{directory}{filename}") as json_file:
        data = json.load(json_file)

    return data


def intTupleToString(a: tuple) -> str:
    """
    Changes a tuple into a string.
    Args:
        a: (tuple) The tuple to be changed

    Returns:
        string: (str) The string representation of the tuple data.
    """
    string = ""
    for i in range(len(a)):
        string += str(a[i])

    return string


def getBitstrings(nBits: int) -> list:
    """
    Returns a list with all possible bitstrings for the given number of bits.
    Args:
        nBits: (int) Number of bits.

    Returns:
        bitstrings: (list) A list with all possible bitstrings for the given number of bits.
    """
    bitstrings = []
    tupleBits = list(itertools.product([0,1], repeat=nBits))
    for i in range(len(tupleBits)):
        bitstrings.append(intTupleToString(tupleBits[i]))

    return bitstrings


def extractPlotData(filename: str, directory: str = "results_qaoa_sweep/") -> tuple:
    """
    Extracts data from the supplied file and stores it in dictionaries.
    Args:
        filename: (str) The filename of the experiment.
        directory: (str) The folder in which the file is located. Default: "results_qaoa_sweep/"

    Returns:
        bitstrings: (list) A list of all possible bitstrings for the experiment.
        plotData: (dict) Dictionary with all relevant data to plot for all repetitions.
        plotDataFull: (dict) Dictionary with all relevant data to plot for each iteration of all repetitions.
        labels: (dict) A dictionary with the labels for each plotData option.
        metaData: (dict) A dictionary with the metaData for the dataset.

    """
    data = openFile(filename=filename, directory=directory)
    metaData = {"shots": data["config"]["QaoaBackend"]["shots"],
                "backend": data["qaoaBackend"]["backend_name"],
                "max_iter": data["config"]["QaoaBackend"]["max_iter"],
                "repetitions": data["config"]["QaoaBackend"]["repetitions"],
                "optimizer": data["config"]["QaoaBackend"]["classical_optimizer"],
                "initial_guess": data["config"]["QaoaBackend"]["initial_guess"]}
    data = data["results"]
    bitstringLength = len(list(data["1"]["counts"].keys())[0])
    bitstrings = getBitstrings(nBits=bitstringLength)
    hpReps = int(len(metaData["initial_guess"]) / 2)

    plotData = {"cf": [],
                "duration": [],
                "filename": []}
    plotDataFull = {"cf": [],
                    "filename": []}
    labels = {"cf": "cf",
              "duration": "time"}
    for i in range(1, hpReps + 1):
        plotData[f"init_beta{i}"] = []
        plotData[f"beta{i}"] = []
        plotData[f"init_gamma{i}"] = []
        plotData[f"gamma{i}"] = []
        plotDataFull[f"init_beta{i}"] = []
        plotDataFull[f"beta{i}"] = []
        plotDataFull[f"init_gamma{i}"] = []
        plotDataFull[f"gamma{i}"] = []
        labels[f"init_beta{i}"] = f"initial {chr(946)}{chr(8320 + i)}"
        labels[f"beta{i}"] = f"{chr(946)}{chr(8320 + i)}"
        labels[f"init_gamma{i}"] = f"initial {chr(947)}{chr(8320 + i)}"
        labels[f"gamma{i}"] = f"{chr(947)}{chr(8320 + i)}"
    for bitstring in bitstrings:
        plotDataFull[f"{bitstring}prop"] = []
        plotDataFull[f"{bitstring}shots"] = []
        plotData[f"{bitstring}prop"] = []
        plotData[f"{bitstring}shots"] = []
        labels[f"{bitstring}prop"] = "probability"
        labels[f"{bitstring}shots"] = "number of shots"

    for key in data:
        tempFilename = data[key]["filename"]
        tempData = openFile(filename=tempFilename, directory="results_qaoa_sweep/")
        plotData["cf"].append(tempData["optimizeResults"]["fun"])
        for i in range(1, hpReps + 1):
            plotData[f"init_beta{i}"].append(tempData["initial_guess"][2 * (i - 1)])
            plotData[f"init_gamma{i}"].append(tempData["initial_guess"][2 * (i - 1) + 1])
            plotData[f"beta{i}"].append(tempData["optimizeResults"]["x"][2*(i-1)])
            plotData[f"gamma{i}"].append(tempData["optimizeResults"]["x"][2*(i-1)+1])
        plotData["duration"].append(tempData["duration"])
        plotData["filename"].append(tempFilename)
        for bitstring in bitstrings:
            if bitstring in data[key]["counts"]:
                plotData[f"{bitstring}shots"].append(data[key]["counts"][bitstring])
                plotData[f"{bitstring}prop"].append(data[key]["counts"][bitstring] / metaData["shots"])
            else:
                plotData[f"{bitstring}shots"].append(0)
                plotData[f"{bitstring}prop"].append(0)

        for i in range(1, tempData["iter_count"] + 1):
            plotDataFull["cf"].append(tempData[f"rep{i}"]["return"])
            for j in range(1, hpReps + 1):
                plotDataFull[f"init_beta{j}"].append(tempData["initial_guess"][2 * (j - 1)])
                plotDataFull[f"init_gamma{j}"].append(tempData["initial_guess"][2 * (j - 1) + 1])
                plotDataFull[f"beta{j}"].append(tempData[f"rep{i}"]["theta"][2 * (j - 1)])
                plotDataFull[f"gamma{j}"].append(tempData[f"rep{i}"]["theta"][2 * (j - 1) + 1])
            plotDataFull["filename"].append(tempFilename)
            for bitstring in bitstrings:
                if bitstring in tempData[f"rep{i}"]["counts"]:
                    plotDataFull[f"{bitstring}shots"].append(tempData[f"rep{i}"]["counts"][bitstring])
                    plotDataFull[f"{bitstring}prop"].append(tempData[f"rep{i}"]["counts"][bitstring] / metaData["shots"])
                else:
                    plotDataFull[f"{bitstring}shots"].append(0)
                    plotDataFull[f"{bitstring}prop"].append(0)

    return bitstrings, plotData, plotDataFull, labels, metaData


def extractHamiltonian(filename: str) -> list:
    """
    Extracts the hamiltonian matrix from the supplied file and returns it.
    Args:
        filename: (str) The filename of the experiment.

    Returns:
        hamiltonian: (list) The hamiltonian matrix used to build the quantum circuit.

    """
    data = openFile(filename=filename, directory="results_qaoa_sweep/")
    subdata = openFile(filename=data["results"]["1"]["filename"], directory="results_qaoa_sweep/")
    hamiltonian = subdata["components"]["hamiltonian"]

    return hamiltonian


def plotBoxplot(filename: str, plotname: str, savename: str, directory: str = "results_qaoa_sweep/"):
    """
    Creates a boxplot of the probability of all possible bitstrings.
    Args:
        filename: (str) filename of dateset to be plotted.
        plotname: (str) title of the plot.
        savename: (str) the name to be used to add to "Scatter_" as the filename of the png
        directory: (str) The folder in which the file is located. Default: "results_qaoa_sweep/"

    Returns:
        Saves the generated plot, with the name "BP_{plotname}.png" to the subfolder 'plots'
    """
    bitstrings, plotData, plotDataFull, labels, metaData = extractPlotData(filename=filename, directory=directory)

    toPlot = [[] for i in range(len(bitstrings))]

    for bitstring in bitstrings:
        bitstring_index = bitstrings.index(bitstring)
        toPlot[bitstring_index] = plotData[f"{bitstring}prop"]

    fig, ax = plt.subplots()
    fig.set_figheight(7)
    pos = np.arange(len(toPlot)) + 1
    bp = ax.boxplot(toPlot, sym='k+', positions=pos, bootstrap=5000)

    ax.set_xlabel('bitstrings')
    ax.set_ylabel('probability')
    plt.title(f"backend = {metaData['backend']}, shots = {metaData['shots']}, rep = {metaData['repetitions']} \n "
              f"initial guess = {metaData['initial_guess']}", fontdict = {'fontsize' : 8})
    plt.figtext(0.0, 0.01, f"data: {filename}", fontdict={'fontsize': 8})
    plt.suptitle(plotname)
    plt.xticks(range(1, len(bitstrings)+1), bitstrings, rotation=70)
    plt.setp(bp['whiskers'], color='k', linestyle='-')
    plt.setp(bp['fliers'], markersize=2.0)
    #plt.show()
    plt.savefig(f"plots/BP_{savename}.png")


def plotBoxplotBest(filename: str, plotname: str, savename: str, cut: float = 0.5,
                    directory: str = "results_qaoa_sweep/"):
    """
    Creates a boxplot of the probability of all possible bitstrings for the best cut of repetitions.
    Args:
        filename: (str) filename of dateset to be plotted.
        plotname: (str) title of the plot.
        savename: (str) the name to be used to add to "Scatter_" as the filename of the png
        cut: (float) the percentage of best repetitions to be plotted: Default: 0.5
        directory: (str) The folder in which the file is located. Default: "results_qaoa_sweep/"

    Returns:
        Saves the generated plot, with the name "BBP_{plotname}.png" to the subfolder 'plots'
    """
    bitstrings, plotData, plotDataFull, labels, metaData = extractPlotData(filename=filename, directory=directory)

    cutoffDictKeys = list(range(1, metaData["repetitions"] + 1))
    cutoff = dict(zip(cutoffDictKeys, plotData["cf"]))
    cutoff = dict(sorted(cutoff.items(), key=lambda item: item[1]))

    toPlot = [[] for i in range(len(bitstrings))]

    for i in range(int(len(cutoff)*cut)):#only plot the specified cut
        key = list(cutoff.keys())[i]
        for bitstring in bitstrings:
            bitstring_index = bitstrings.index(bitstring)
            toPlot[bitstring_index].append(plotData[f"{bitstring}prop"][key-1])

    fig, ax = plt.subplots()
    fig.set_figheight(7)
    pos = np.arange(2*len(toPlot),step=2) + 1
    bp = ax.boxplot(toPlot, sym='k+', positions=pos, bootstrap=5000)

    ax.set_xlabel('bitstrings')
    ax.set_ylabel('probability')
    plt.title(f"backend = {metaData['backend']}, shots = {metaData['shots']}, rep = {metaData['repetitions']} \n "
              f"initial guess = {metaData['initial_guess']}", fontdict={'fontsize': 8})
    plt.figtext(0.0, 0.01, f"data: {filename}", fontdict={'fontsize': 8})
    plt.suptitle(plotname)
    plt.xticks(range(1, 2*len(bitstrings) + 1,2), bitstrings, rotation=70)
    plt.setp(bp['whiskers'], color='k', linestyle='-')
    plt.setp(bp['fliers'], markersize=2.0)
    #plt.show()
    plt.savefig(f"plots/BBP_{savename}.png")


def plotCFoptimizationDouble(filename: str, plotname: str, savename: str, directory: str = "results_qaoa_sweep/"):
    """
    Creates two plot showing the evolution of the cost function, and all betas and gammas of two random repetitions.
    Args:
        filename: (str) filename of dateset to be plotted.
        plotname: (str) title of the plot.
        savename: (str) the name to be used to add to "Scatter_" as the filename of the png
        directory: (str) The folder in which the file is located. Default: "results_qaoa_sweep/"

    Returns:
        Saves the generated plot, with the name "CF_{plotname}.png" to the subfolder 'plots'
    """
    bitstrings, plotData, plotDataFull, labels, metaData = extractPlotData(filename=filename, directory=directory)

    fig, axs = plt.subplots(2, sharex=True, sharey=True)
    fig.set_figheight(7)

    random_list = list(range(1, metaData["repetitions"]))
    random.shuffle(random_list)

    for i in range(len(axs)):
        if i >= len(random_list):
            print("Index Error, skip iteration")
            continue
        rep = int(random_list[i])
        indexBegin = plotDataFull["filename"].index(plotData["filename"][rep])
        plotDataFull["filename"].reverse()
        indexEnd = plotDataFull["filename"].index(plotData["filename"][rep])
        plotDataFull["filename"].reverse()
        indexEnd = len(plotDataFull["filename"]) - indexEnd
        hpReps = int(len(metaData["initial_guess"]) / 2)

        xData = list(range(0, indexEnd - indexBegin))
        leg = []

        for j in range(1, hpReps + 1):
            axs[i].plot(xData, plotDataFull[f"beta{j}"][indexBegin:indexEnd], color=(0, 0, (1 - ((j - 1) / hpReps)), 1),
                        label=labels[f"beta{j}"])
            leg.append(labels[f"beta{j}"])
            axs[i].plot(xData, plotDataFull[f"gamma{j}"][indexBegin:indexEnd], color=((1 - ((j - 1) / hpReps)), 0, 0, 1),
                        label=labels[f"gamma{j}"])
            leg.append(labels[f"gamma{j}"])
        axs[i].plot(xData, plotDataFull["cf"][indexBegin:indexEnd], "g-", label=labels["cf"])
        leg.append(labels["cf"])
        axs[i].set_xlabel('iteration')
        axs[i].set_ylabel('value')
        axs[i].label_outer()
        axs[i].text(0.22, 0.02, f"data: {plotData['filename'][rep]}", transform=axs[i].transAxes, fontdict={'fontsize': 6})
        axs[i].set_title(f"rep {rep}", fontdict={'fontsize': 8})

    fig.suptitle(plotname)
    fig.legend(leg, loc="upper right")
    plt.figtext(0.0, 0.01, f"data: {filename}", fontdict={'fontsize': 8})
    #plt.show()
    plt.savefig(f"plots/CF_{savename}.png")


def plotCFoptimizationSingle(filename: str, repetition: str, plotname: str, savename: str, directory: str = "results_qaoa_sweep/"):
    """
    Creates two plot showing the evolution of the cost function, and all betas and gammas of a chosen repetition.
    Args:
        filename: (str) filename of dateset to be plotted.
        plotname: (str) title of the plot.
        savename: (str) the name to be used to add to "Scatter_" as the filename of the png
        directory: (str) The folder in which the file is located. Default: "results_qaoa_sweep/"

    Returns:
        Saves the generated plot, with the name "CFSingle_{plotname}.png" to the subfolder 'plots'
    """
    bitstrings, plotData, plotDataFull, labels, metaData = extractPlotData(filename=filename, directory=directory)

    minCF = min(plotData["cf"])
    index_minCF = plotData["cf"].index(minCF)

    fig, ax = plt.subplots()
    fig.set_figheight(7)

    rep = int(repetition) - 1
    indexBegin = plotDataFull["filename"].index(plotData["filename"][rep])
    plotDataFull["filename"].reverse()
    indexEnd = plotDataFull["filename"].index(plotData["filename"][rep])
    plotDataFull["filename"].reverse()
    indexEnd = len(plotDataFull["filename"]) - indexEnd
    hpReps = int(len(metaData["initial_guess"]) / 2)

    xData = list(range(0, indexEnd - indexBegin))
    leg = []

    for j in range(1, hpReps + 1):
        ax.plot(xData, plotDataFull[f"beta{j}"][indexBegin:indexEnd], color=(0, 0, (1 - ((j - 1) / hpReps)), 1),
                label=labels[f"beta{j}"])
        leg.append(labels[f"beta{j}"])
        ax.plot(xData, plotDataFull[f"gamma{j}"][indexBegin:indexEnd], color=((1 - ((j - 1) / hpReps)), 0, 0, 1),
                label=labels[f"gamma{j}"])
        leg.append(labels[f"gamma{j}"])
    ax.plot(xData, plotDataFull["cf"][indexBegin:indexEnd], "g-", label=labels["cf"])
    leg.append(labels["cf"])
    ax.set_xlabel('iteration')
    ax.set_ylabel('value')
    ax.label_outer()
    #ax.set_title(f"rep {repetition}", fontdict={'fontsize': 8})

    fig.suptitle(plotname)
    fig.legend(leg, loc="upper right")
    plt.figtext(0.0, 0.01, f"data: {plotData['filename'][rep]}", fontdict={'fontsize': 8})
    #plt.show()
    plt.savefig(f"plots/CFSingle_{savename}.png")


def plotBPandCF(filename: str, extraPlotInfo:str, savename: str, cut: float = 0.5,
                directory: str = "results_qaoa_sweep/"):
    """
    Generated three plots.
        1. A boxplot of the probability of all possible bitstrings;
        2. A boxplot of the probability of all possible bitstrings for the best cut of repetitions;
        3. A plot showing the evolution of the cost function, and all betas and gammas of two random repetitions.
    Args:
        filename: (str) filename of dateset to be plotted.
        extraPlotInfo: (str) extra information used to generate the title of the plot.
        savename: (str) the name to be used to add to "Scatter_" as the filename of the png
        cut: (float) the percentage of best repetitions to be plotted: Default: 0.5
        directory: (str) The folder in which the file is located. Default: "results_qaoa_sweep/"

    Returns:
        Saves the generated plot to the subfolder 'plots' with the following names:
            1. BP_{plotname}.png
            2. BBP_{plotname}.png
            3. CF_{plotname}.png
    """
    dataAll = openFile(filename=filename, directory=directory)

    if dataAll["config"]["QaoaBackend"]["simulate"]:
        if dataAll["config"]["QaoaBackend"]["noise"]:
            noise = "with noise"
        else:
            noise = "without noise"
    else:
        noise = "on QPU"
    optimizer = dataAll["config"]["QaoaBackend"]["classical_optimizer"]
    maxiter = dataAll["config"]["QaoaBackend"]["max_iter"]
    plotnameBP = f"{optimizer} {noise} - maxiter {maxiter} \n {extraPlotInfo}"
    plotnameBPB = f"{optimizer} {noise} - maxiter {maxiter}, best 50% \n {extraPlotInfo}"
    plotnameCF = f"{optimizer} CF evolution {noise} - maxiter {maxiter} \n {extraPlotInfo}"

    plotBoxplot(filename=filename, plotname=plotnameBP, savename=savename)
    plotBoxplotBest(filename=filename, plotname=plotnameBPB, savename=savename, cut=cut)
    if len(dataAll["results"]) > 1:
        plotCFoptimizationDouble(filename=filename, plotname=plotnameCF, savename=savename)


def buildKirchLabels(filename: str, directory: str = "results_qaoa_sweep/", kirchLabels: int = None) -> list:
    """
    Builds labels consisting of the bitstring and the kirchhoff costs for this bitstring, spanning over two lines.
    Args:
        filename: (str) filename of dateset to be used for the kirchhoff cost extraction.
        directory: (str) The folder in which the file is located. Default: "results_qaoa_sweep/"
        kirchLabels: (int) if None, no kirchhoff costs will be added to the labels. Otherwise the kirchhoff costs will
                           be extracted from the given file and added to the labels.

    Returns:
        labels: (list) A list with the labels consisting of the bitstrings and (possibly) the kirchhoff costs.
    """
    dataAll = openFile(filename=filename, directory=directory)
    nBits = len(list(dataAll["results"]["1"]["counts"].keys())[0])
    bitstrings = getBitstrings(nBits=nBits)

    labels = []
    if kirchLabels == None:
        labels = bitstrings
    else:
        for bitstring in bitstrings:
            kirchValue = dataAll["kirchhoff"][bitstring]["total"]
            labels.append(f"{bitstring}\nc={kirchValue}")

    return labels


def plotBitstringBoxCompare(filenames: list, labels: list, colors: list, savename: str,
                            directory: str = "results_qaoa_sweep/", title: str = "Comparision Boxplot",
                            cut: float = 0.5, kirchLabels: int = None):
    """
    Creates boxplots of the probability of all possible bitstrings for the best cut of repetitions of multiple datasets
    next to each other.
    Args:
        filenames: (list) filenames of datesets to be plotted.
        labels: (list) labels for datasets to be plotted.
        colors: (list) colors to be used for datasets.
        savename: (str) the name to be used to add to "Scatter_" as the filename of the png
        directory: (str) The folder in which the file is located. Default: "results_qaoa_sweep/"
        title: (str) title of the plot.
        cut: (float) the percentage of best repetitions to be plotted: Default: 0.5
        kirchLabels: (int) indicates which element of the list filenames is to be used to extract the kirchhoff costs.
                           If None, no kirchhoff costs will be added to the bitstring labels.

    Returns:
        Saves the generated plot, with the name "BPcomp_{savename}_cut{cut}.png" to the subfolder 'plots'
    """
    cutoff = {}
    toPlot = {}
    if kirchLabels:
        xlabels = buildKirchLabels(filename=filenames[kirchLabels], directory=directory, kirchLabels=kirchLabels)
    else:
        xlabels = buildKirchLabels(filename=filenames[0], directory=directory, kirchLabels=kirchLabels)
    for i in range(len(filenames)):
        bitstrings, plotData, plotDataFull, labelsExtract, metaData = extractPlotData(filename=filenames[i],
                                                                                      directory=directory)

        cutoffDictKeys = list(range(1, metaData["repetitions"] + 1))
        cutoff[i] = dict(zip(cutoffDictKeys, plotData["cf"]))
        cutoff[i] = dict(sorted(cutoff[i].items(), key=lambda item: item[1]))

        toPlot[i] = [[] for j in range(len(bitstrings))]
        for bitstring in bitstrings:
            bitstring_index = bitstrings.index(bitstring)
            for j in range(int(len(cutoff[i]) * cut)):  # only plot best cut
                key = list(cutoff[i].keys())[j]
                toPlot[i][bitstring_index].append(plotData[f"{bitstring}prop"][key-1])

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)
        plt.setp(bp['fliers'], markersize=5.0, marker="+", markeredgecolor=color)

    fig = plt.figure(figsize=(13,5))

    nPlots = len(filenames)
    if nPlots == 1:
        boxWidth = 0.6
        boxDistance = [0]
    elif nPlots == 2:
        boxWidth = 0.6
        boxDistance = [-0.4, 0.4]
    elif nPlots == 3:
        boxWidth = 0.5
        boxDistance = [-0.75, 0, 0.75]
    elif nPlots == 4:
        boxWidth = 0.4
        boxDistance = [-0.9, -0.3, 0.3, 0.9]

    for i in range(nPlots):
        bp = plt.boxplot(x=toPlot[i], positions=np.array(range(len(toPlot[i]))) * nPlots + boxDistance[i],
                         widths=boxWidth)
        set_box_color(bp, colors[i])
        plt.plot([], c=colors[i], label=labels[i])

    plt.legend()

    plt.title(title)
    plt.xticks(range(0, len(bitstrings) * nPlots, nPlots), xlabels)
    plt.xlim(-1.5, len(bitstrings) * nPlots + 0.5)
    plt.tight_layout()
    plt.xlabel('bitstrings')
    plt.ylabel('probability')
    fig.set_figheight(7)
    fig.set_figwidth(15)
    #plt.show()
    plt.savefig(f"plots/BPcomp_{savename}_cut{cut}.png")


def plotScatter(file1: str, file2: str, title: str, g1title: str, g2title: str, savename: str, mode: str, x: str,
                y: str, directory: str = "results_qaoa_sweep/"):
    """
    Creates two scatter plots of the same features for two different datasets.
    Args:
        file1: (str) filename of dateset 1.
        file2: (str) filename of dateset 2.
        title: (str) title of the plot.
        g1title: (str) title of the dataset 1 subplot.
        g2title: (str) title of the dataset 2 subplot.
        savename: (str) the name to be used to add to "Scatter_" as the filename of the png
        mode: (str) the mode for the data retrival. Choose from "opt", use only the optimized data or "full", use the
                    data from each optimiziation iteration
        x: (str) feature to be plotted on x-axis.
        y: (str) feature to be plotted on y-axis.
        directory: (str) The folder in which the file is located. Default: "results_qaoa_sweep/"
        Features to choose from for the x- and y-axis:
            - "cf" (cost function);
            - "init_beta{i}", where i is the number of beta, e.g. beta1, beta2, ...;
            - "beta{i}", where i is the number of beta, e.g. beta1, beta2, ...;
            - "init_gamma{i}", where i is the number of gamma, e.g. gamma1, gamma2, ...;
            - "gamma{i}", where i is the number of gamma, e.g. gamma1, gamma2, ...;
            - "{bitstring}prop" (probability of the chosen bistring);
            - "{bitstring}shots" (number of shots of the chosen bitstring);
            - "duration" (only available if mode is set to "opt").

    Returns:
        Saves the generated plot, with the name "Scatter_{plotname}.png" to the subfolder 'plots'
    """
    bitstrings, plotData1, plotDataFull1, labels, metaData1 = extractPlotData(filename=file1, directory=directory)
    bitstrings, plotData2, plotDataFull2, labels, metaData2 = extractPlotData(filename=file2, directory=directory)

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    fig.set_figwidth(15)

    if mode == "full":
        plotx1 = plotDataFull1[x]
        ploty1 = plotDataFull1[y]
        plotx2 = plotDataFull2[x]
        ploty2 = plotDataFull2[y]
    elif mode == "opt":
        plotx1 = plotData1[x]
        ploty1 = plotData1[y]
        plotx2 = plotData2[x]
        ploty2 = plotData2[y]

    ax1.scatter(x=plotx1, y=ploty1)
    ax2.scatter(x=plotx2, y=ploty2)
    ax1.set_xlabel(labels[x])
    ax1.set_ylabel(labels[y])
    ax2.set_xlabel(labels[x])
    ax2.set_ylabel(labels[y])
    ax1.set_title(g1title)
    ax2.set_title(g2title)
    plt.suptitle(title, y= 1, fontsize="xx-large")
    #plt.show()
    plt.savefig(f"plots/Scatter_{savename}.png")


def plotEigenvalues(filename: str, plotname:str, savename: str):
    hamiltonian = extractHamiltonian(filename=filename)
    eigenvalues, eigenvectors = linalg.eig(hamiltonian)

    plt.hist(eigenvalues)
    plt.xlabel("eigenvalues")
    plt.title(plotname)
    #plt.show()
    plt.savefig(f"plots/Hist_{savename}.png")


def meanOfInitGuess(filename: str):
    bitstrings, plotData, plotDataFull, labels, metaData = extractPlotData(filename=filename)
    betaMean = mean(plotData["init_beta1"])
    betaMedian = median(plotData["init_beta1"])
    gammaMean = mean(plotData["init_gamma1"])
    gammaMedian = median(plotData["init_gamma1"])
    minCFindex, minCF = min(enumerate(plotData["cf"]), key=operator.itemgetter(1))


    print(filename)
    print(f"minCF = {minCF}, at index {minCFindex} with beta = {plotData['beta1'][minCFindex]} and gamma = {plotData['gamma1'][minCFindex]}")
    print(f"Beta: mean = {betaMean}, median = {betaMedian}")
    print(f"Gamma: mean = {gammaMean}, median = {gammaMedian}")


def main():
    blueDark = "#003C50"
    blueMedium = "#005C7B"
    blueLight = "#008DBB"
    orangeDark = "#B45E00"
    orangeMedium = "#F07D00"
    orangeLight = "#FFB15D"

    #meanOfInitGuess(filename="infoNocost_testNetwork4QubitIsing_2_3_20.nc_30_1_2022-03-02_11-51-19_config.yaml")

    colors = [blueLight, blueDark, orangeLight, orangeDark]

    filenames = ["infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-04-04_15-27-33_config_90.yaml",
                 "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-04-04_17-07-10_config_91.yaml",
                 "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-04-04_17-07-10_config_92.yaml",
                 "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-04-04_17-07-10_config_93.yaml"]
    labels = ["1 Layer", "2 Layers", "4 Layers", "1 Layer mit schlechten Ausgangswerten"]
    title = "Einfluss des Initialwertes und der Anzahl der Layer"
    colors = [blueLight, blueDark, orangeLight, orangeDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_diffLayers_testNetwork4QubitIsing_2_0_20", title=title, cut=1.0,
                            kirchLabels=0)

    return

    plotCFoptimizationSingle(
        filename="infoNocost_testNetwork4QubitIsing_2_3_20.nc_30_1_2022-03-03_22-26-51_config_82.yaml",
        repetition="49", plotname="test1", savename="test_1")

    plotCFoptimizationSingle(filename="infoNocost_testNetwork4QubitIsing_2_3_20.nc_30_1_2022-03-03_22-26-51_config_82.yaml_rand",
                             repetition="20", plotname="test1", savename="test_1")

    return

    filenames = ["infoNocost_testNetwork4QubitIsing_2_3_20.nc_30_1_2022-03-03_22-26-51_config_80.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_3_20.nc_30_1_2022-03-03_22-26-51_config_81.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_3_20.nc_30_1_2022-03-03_22-26-51_config_82.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-03-03_16-43-45_config_83.yaml"]
    labels = ["single Hp + min cf rand", "single Hp + 1;1", "double Hp + min cf rand", "double Hp + 1;1"]
    title = "Network 3 evaluation - with noise"
    colors = [blueLight, blueDark, orangeLight, orangeDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_init-rand_2Hp_NOISE_testNetwork4QubitIsing_2_3_20", title=title, cut=1.0,
                            kirchLabels=0)

    return

    filenames = [
#    "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-14_14-17-31_config.yaml",
            "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-14_14-46-59_config.yaml",
#            "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-14_15-00-02_config.yaml",
#            "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-14_15-25-23_config.yaml",
#            "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-14_15-38-27_config.yaml",
#            "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-14_16-00-02_config.yaml", #inverted
#            "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-14_16-26-53_config.yaml",
#            "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-14_17-46-37_config.yaml",
#            "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-14_16-42-21_config.yaml",
#            "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-14_18-05-23_config.yaml",
#            "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-14_18-18-44_config.yaml",
#            "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-14_19-10-36_config.yaml",
#            "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-14_20-07-27_config.yaml",
#            "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-14_20-17-39_config.yaml",
#            "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-14_20-30-01_config.yaml",
#            "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-14_20-41-50_config.yaml",
#            "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-14_20-52-09_config.yaml",
#            "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-15_10-02-33_config.yaml",
#            "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-15_10-25-01_config.yaml",
#            "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-15_10-37-45_config.yaml",
#            "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-15_10-46-19_config.yaml",
#            "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-15_10-58-49_config.yaml",
#            "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-15_11-08-17_config.yaml",
#            "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-15_11-18-44_config.yaml",
#            "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-15_15-06-16_config.yaml",
#            "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-15_15-13-15_config.yaml",
#            "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-15_15-36-26_config.yaml",
#            "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-15_16-45-31_config.yaml",
#            "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-15_16-54-32_config.yaml",
            "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-15_17-06-09_config.yaml",
#            "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-15_18-17-20_configGreedy.yaml",
#            "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-15_18-21-07_configGreedy2.yaml",
            "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-15_18-25-54_configGreedy3.yaml",
            "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-15_18-36-10_configSingleton.yaml",
    ]

    meanOfInitGuess(filename=filenames[3])
#    plotBitstringBoxCompare(filenames=filenames, colors=colors,
#                labels=["randomHalf",
#                        "AdamLinearDoubleParam",
#                        "AdamReasonableIteration",
#                        "AdamSingleton"
#                        ],
#                savename="CompareGuessFactors", cut=1.0,
#                kirchLabels=0)

#    plotBPandCF(filenames[0],
#            extraPlotInfo="fixedInitGuess",
#            savename="FixedGUess" + filenames[0])
#    plotBPandCF(filenames[0],
#            extraPlotInfo="randomHalfGuess",
#            savename="randomHalfGuess" + filenames[0])
#    plotBPandCF(filenames[2],
#            extraPlotInfo="randomTenthGuess",
#            savename="randomTenthGuess" + filenames[2])
#    plotBPandCF(filenames[3],
#            extraPlotInfo="randomhundredGuessEqualRand",
#            savename="randomhundredGuessEqualRand" + filenames[3])
#    plotBPandCF(filenames[3],
#            extraPlotInfo="GuessIsLastBestResult",
#            savename="GuessIsLastBestResult" + filenames[3])
#    plotBPandCF(filenames[3],
#            extraPlotInfo="scaledCost",
#            savename="scaledCost" + filenames[3])
#    plotBPandCF(filenames[3],
#            extraPlotInfo="scaledCost",
#            savename="scaledCost" + filenames[3])
#    plotBPandCF(filenames[2],
#            extraPlotInfo="SPSA",
#            savename="SPSA" + filenames[2])
#    plotBPandCF(filenames[2],
#            extraPlotInfo="COBRandomSign",
#            savename="COBRandomSign" + filenames[2])
#    plotBPandCF(filenames[3],
#            extraPlotInfo="COBFixedSign",
#            savename="COBFixedSign" + filenames[3])
#    plotBPandCF(filenames[1],
#            extraPlotInfo="LinScaledCost",
#            savename="LinScaledCost" + filenames[1])
#    plotBPandCF(filenames[3],
#            extraPlotInfo="RootOfLinearKirchhoff",
#            savename="RootOfLinearKirchhoff" + filenames[3])
#    plotBPandCF(filenames[3],
#            extraPlotInfo="InverseHammingCost",
#            savename="InverseHammingCost" + filenames[3])
#    plotBPandCF(filenames[3],
#            extraPlotInfo="HammingCost",
#            savename="HammingCost" + filenames[3])
#    plotBPandCF(filenames[3],
#            extraPlotInfo="IsingHamming",
#            savename="IsingHamming" + filenames[3])
#    plotBPandCF(filenames[0],
#            extraPlotInfo="DoubleDiagonalIsingHamming",
#            savename="DoubleDiagonalIsingHamming" + filenames[2])
#    plotBPandCF(filenames[3],
#            extraPlotInfo="HalfDiagonalIsingHamming",
#            savename="HalfDiagonalIsingHamming" + filenames[3])
#    plotBPandCF(filenames[3],
#            extraPlotInfo="HalfDiagonalIsingRootLoss",
#            savename="HalfDiagonalIsingRootLoss" + filenames[3])
#    plotBPandCF(filenames[3],
#            extraPlotInfo="AdamSquaredKirch",
#            savename="AdamSquaredKirch" + filenames[3])
#    plotBPandCF(filenames[1],
#            extraPlotInfo="AdamSquaredKirchNoDiagonal",
#            savename="AdamSquaredKirchNoDiagonal" + filenames[1])
#    plotBPandCF(filenames[2],
#            extraPlotInfo="AdamSquaredKirchNegativeDiagonal",
#            savename="AdamSquaredKirchNegativeDiagonal" + filenames[2])
#    plotBPandCF(filenames[3],
#            extraPlotInfo="AdamSquaredKirchNegative4thDiagonal",
#            savename="AdamSquaredKirchNegativet4thDiagonal" + filenames[3])
#    plotBPandCF(filenames[3],
#            extraPlotInfo="AdamLinearNoDiagonal",
#            savename="AdamLinearNoDiagonal"+ filenames[3])
#    plotBPandCF(filenames[3],
#            extraPlotInfo="AdamRootNoDiagonalPreviousBestGuess",
#            savename="AdamRootNoDiagonalPreviousBestGuess"+ filenames[3])
#    plotBPandCF(filenames[3],
#            extraPlotInfo="AdamSquaredNoDemandTermIsingFixedGuess",
#            savename="AdamSquaredNoDemandTermIsingFixedGuess"+ filenames[3])
#    plotBPandCF(filenames[3],
#            extraPlotInfo="AdamSquaredNegativeDiagonalDoubleParam",
#            savename="AdamSquaredNegativeDiagonalDoubleParam" + filenames[3])
#    plotBPandCF(filenames[3],
#            extraPlotInfo="AdamLineardNegativeDiagonalDoubleParam",
#            savename="AdamLineardNegativeDiagonalDoubleParam" + filenames[3])
#    plotBPandCF(filenames[1],
#            extraPlotInfo="AdamLinearDoubleParam",
#            savename="AdamLinearDoubleParam" + filenames[1])
##    plotBPandCF(filenames[2],
##            extraPlotInfo="AdamLinearGreedy",
##            savename="AdamLinearGreedy" + filenames[2])
#    plotBPandCF(filenames[2],
#            extraPlotInfo="AdamReasonableIteration",
#            savename="AdamReasonableIteration " + filenames[2])
#    plotBPandCF(filenames[3],
#            extraPlotInfo="AdamSingleton",
#            savename="AdamSingleton" + filenames[3])
    

    SomeFile = "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-18_18-15-47_configIter1.yaml"
    plotBPandCF(SomeFile,
            extraPlotInfo="COBYLAOptimalGuess",
            savename="COBYLAOptimalGuess" + SomeFile)
    SomeFile = "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-21_09-41-28_configIter1.yamlrand"
    plotBPandCF(SomeFile,
            extraPlotInfo="COBYLAFullsplit4Param",
            savename="COBYLAFullsplit4Param" + SomeFile)
    SomeFile = "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-21_10-08-08_configIter1.yamlrand"
    plotBPandCF(SomeFile,
            extraPlotInfo="COBYLAFullsplit4ParamEfficientKirchhoff",
            savename="COBYLAFullsplit4ParamEfficientKirchhoff" + SomeFile)
    SomeFile = "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-21_10-48-46_configIter1.yamlrand"
    plotBPandCF(SomeFile,
            extraPlotInfo="COBYLAFullsplit4ParamBiggerSweep",
            savename="COBYLAFullsplit4ParamBiggerSweep" + SomeFile)

    return
    SomeFile = "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-18_17-42-36_configIter1.yamlrand"
    plotBPandCF(SomeFile,
            extraPlotInfo="CobylaBinarySplitIsingTest",
            savename="CobylaBinarySplitIsingTest" + SomeFile)

    SomeFile = "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-18_10-21-53_configIter1.yamlrand"
    plotBPandCF(SomeFile,
            extraPlotInfo="CobylaFullSplit4paramHighShotQuadraticLoss",
            savename="CobylaFullSplit4paramHighShotQuadraticLoss" + SomeFile)
    SomeFile = "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-18_10-33-48_configIter1.yamlrand"
    plotBPandCF(SomeFile,
            extraPlotInfo="CobylaFullSplit8paramHighShotQuadraticLoss",
            savename="CobylaFullSplit8paramHighShotQuadraticLoss" + SomeFile)
    SomeFile = "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-18_10-38-00_configIter1.yamlrand"
    plotBPandCF(SomeFile,
            extraPlotInfo="AdamDullsplit8ParamHighShotQuadraticLoss",
            savename="AdamDullsplit8ParamHighShotQuadraticLoss" + SomeFile)

    return

    SomeFile = "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-18_09-42-06_configIter1.yamlrand"
    plotBPandCF(SomeFile,
            extraPlotInfo="CobylaFullSplit2param",
            savename="CobylaFullSplit2param" + SomeFile)
    SomeFile = "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-18_09-59-04_configIter1.yamlrand"
    plotBPandCF(SomeFile,
            extraPlotInfo="CobylaFullSplit4param",
            savename="CobylaFullSplit4param" + SomeFile)
    SomeFile = "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-18_10-07-31_configIter1.yamlrand"
    plotBPandCF(SomeFile,
            extraPlotInfo="CobylaFullSplit4paramBinaryCost",
            savename="CobylaFullSplit4paramBinaryCost" + SomeFile)
    SomeFile = "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-18_10-09-22_configIter1.yamlrand"
    plotBPandCF(SomeFile,
            extraPlotInfo="CobylaFullSplit4paramQuadraticCost",
            savename="CobylaFullSplit4paramQuadraticCost" + SomeFile)
    SomeFile = "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-18_10-12-00_configIter1.yamlrand"
    plotBPandCF(SomeFile,
            extraPlotInfo="CobylaFullSplit4paramScaledLinearQuadraticCost",
            savename="CobylaFullSplit4paramScaledLinearQuadraticCost" + SomeFile)

    return
    SomeFile = "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-17_18-20-07_configIter1.yamlrand"
    plotBPandCF(SomeFile,
            extraPlotInfo="CobylaGuess8Param",
            savename="CobylaGuess8Param" + SomeFile)
    SomeFile = "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-17_18-28-10_configIter1.yaml"
    plotBPandCF(SomeFile,
            extraPlotInfo="CobylaGuess4x3-3with8Param",
            savename="CobylaGuess4x3-3with8Param" + SomeFile)
    SomeFile = "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-17_18-30-03_configIter1.yaml"
    plotBPandCF(SomeFile,
            extraPlotInfo="CobylaGuess4x3-3with8ParamLongIterate",
            savename="CobylaGuess4x3-3with8ParamLongIterate" + SomeFile)
    SomeFile = "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-17_18-32-52_configIter1.yaml"
    plotBPandCF(SomeFile,
            extraPlotInfo="CobylaGuess4x3-3with8ParamLongIterateMoreRep",
            savename="CobylaGuess4x3-3with8ParamLongIterateMoreRep" + SomeFile)
    SomeFile = "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-17_18-38-50_configIter1.yaml"
    plotBPandCF(SomeFile,
            extraPlotInfo="CobylaGuess0Initwith8ParamLongIterateMoreRep",
            savename="CobylaGuess0Init8ParamLongIterateMoreRep" + SomeFile)
    SomeFile = "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-17_18-42-23_configIter1.yaml"
    plotBPandCF(SomeFile,
            extraPlotInfo="CobylaGuess0Initwith4ParamLongIterateMoreRep",
            savename="CobylaGuess0Initwith4ParamLongIterateMoreRep" + SomeFile)


    SomeFile = "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-15_20-49-50_configSingletonNegativeSign.yaml"
    plotBPandCF(SomeFile,
            extraPlotInfo="AdamSingletonNegativeStart",
            savename="AdamSingletonNegativeStart" + SomeFile)

    SomeFile = "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-15_20-55-28_configSingletonNegativeSign.yaml"
    plotBPandCF(SomeFile,
            extraPlotInfo="AdamSingletonPosBetaNegGamma",
            savename="AdamSingletonPosBetaNegGamme" + SomeFile)
 
    SomeFile = "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-16_09-05-38_config.yaml"
    plotBPandCF(SomeFile,
            extraPlotInfo="AdamSmallInitGuess2Param",
            savename="AdamSmallInitGuess2Param" + SomeFile)

    SomeFile = "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-16_09-12-59_config.yaml"
    plotBPandCF(SomeFile,
            extraPlotInfo="AdamSmallRepMoreShots",
            savename="AdamSmallResMoreShots" + SomeFile)

    SomeFile = "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-16_09-27-24_config.yaml"
    plotBPandCF(SomeFile,
            extraPlotInfo="AdamSmallRepOptimalGuess",
            savename="AdamSmallResOptimalGuess" + SomeFile)

    SomeFile = "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-17_16-46-14_configIter1.yaml"
    plotBPandCF(SomeFile,
            extraPlotInfo="AdamSmallRepOptimalGuessAndThird",
            savename="AdamSmallResOptimalGuessAndThird" + SomeFile)

    SomeFile = "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-17_17-04-41_configIter1.yaml"
    plotBPandCF(SomeFile,
            extraPlotInfo="AdamOptimalGuessAndThird",
            savename="AdamOptimalGuessAndThird" + SomeFile)

    SomeFile = "infoNocostFixed_testNetwork4QubitIsing_2_0_20.nc_60_1_2022-03-17_17-46-14_configIter1.yaml"
    plotBPandCF(SomeFile,
            extraPlotInfo="AdamOptimalGuess8Param",
            savename="AdamOptimalGuess8Param" + SomeFile)
    return

    filenames = ["infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-03-03_16-43-45_config_80.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-03-03_16-43-45_config_81.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-03-03_16-43-45_config_82.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_3_20.nc_30_1_2022-03-03_22-26-51_config_83.yaml"]
    labels = ["single Hp + min cf rand", "single Hp + 1;1", "double Hp + min cf rand", "double Hp + 1;1"]
    title = "Network 0 evaluation - with noise"
    colors = [blueLight, blueDark, orangeLight, orangeDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_init-rand_2Hp_NOISE_testNetwork4QubitIsing_2_0_20", title=title, cut=1.0,
                            kirchLabels=0)

    filenames = ["infoNocost_testNetwork4QubitIsing_2_1_20.nc_30_1_2022-03-03_16-43-45_config_80.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_1_20.nc_30_1_2022-03-03_16-43-45_config_81.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_1_20.nc_30_1_2022-03-03_16-43-45_config_82.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_2_20.nc_30_1_2022-03-03_22-26-51_config_83.yaml"]
    labels = ["single Hp + min cf rand", "single Hp + 1;1", "double Hp + min cf rand", "double Hp + 1;1"]
    title = "Network 1 evaluation - with noise"
    colors = [blueLight, blueDark, orangeLight, orangeDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_init-rand_2Hp_NOISE_testNetwork4QubitIsing_2_1_20", title=title, cut=1.0,
                            kirchLabels=0)

    filenames = ["infoNocost_testNetwork4QubitIsing_2_2_20.nc_30_1_2022-03-03_22-26-51_config_80.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_2_20.nc_30_1_2022-03-03_22-26-51_config_81.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_2_20.nc_30_1_2022-03-03_22-26-51_config_82.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_1_20.nc_30_1_2022-03-03_16-43-45_config_83.yaml"]
    labels = ["single Hp + min cf rand", "single Hp + 1;1", "double Hp + min cf rand", "double Hp + 1;1"]
    title = "Network 2 evaluation - with noise"
    colors = [blueLight, blueDark, orangeLight, orangeDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_init-rand_2Hp_NOISE_testNetwork4QubitIsing_2_2_20", title=title, cut=1.0,
                            kirchLabels=0)

    filenames = ["infoNocost_testNetwork4QubitIsing_2_3_20.nc_30_1_2022-03-03_22-26-51_config_80.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_3_20.nc_30_1_2022-03-03_22-26-51_config_81.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_3_20.nc_30_1_2022-03-03_22-26-51_config_82.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-03-03_16-43-45_config_83.yaml"]
    labels = ["single Hp + min cf rand", "single Hp + 1;1", "double Hp + min cf rand", "double Hp + 1;1"]
    title = "Network 3 evaluation - with noise"
    colors = [blueLight, blueDark, orangeLight, orangeDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_init-rand_2Hp_NOISE_testNetwork4QubitIsing_2_3_20", title=title, cut=1.0,
                            kirchLabels=0)

    return
    filenames = ["infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-03-01_07-19-44_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-03-02_11-51-19_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-03-02_13-34-27_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-03-02_16-16-15_config.yaml"]
    labels = ["initial guess [1, 1]", "initial guess random", "initial guess mean from random", "initial guess min cf from random"]
    title = "Network 0 evaluation"
    colors = [blueLight, orangeLight, orangeMedium, orangeDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_init-rand_testNetwork4QubitIsing_2_0_20", title=title, cut=1.0,
                            kirchLabels=0)

    filenames = ["infoNocost_testNetwork4QubitIsing_2_1_20.nc_30_1_2022-03-01_07-19-44_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_1_20.nc_30_1_2022-03-02_11-51-19_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_1_20.nc_30_1_2022-03-02_13-52-15_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_1_20.nc_30_1_2022-03-02_16-32-33_config.yaml"]
    labels = ["initial guess [1, 1]", "initial guess random", "initial guess mean from random", "initial guess min cf from random"]
    title = "Network 1 evaluation"
    colors = [blueLight, orangeLight, orangeMedium, orangeDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_init-rand_testNetwork4QubitIsing_2_1_20", title=title, cut=1.0,
                            kirchLabels=0)

    filenames = ["infoNocost_testNetwork4QubitIsing_2_2_20.nc_30_1_2022-03-01_07-19-44_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_2_20.nc_30_1_2022-03-02_11-51-19_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_2_20.nc_30_1_2022-03-02_15-40-13_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_2_20.nc_30_1_2022-03-02_16-50-16_config.yaml"]
    labels = ["initial guess [1, 1]", "initial guess random", "initial guess mean from random", "initial guess min cf from random"]
    title = "Network 2 evaluation"
    colors = [blueLight, orangeLight, orangeMedium, orangeDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_init-rand_testNetwork4QubitIsing_2_2_20", title=title, cut=1.0,
                            kirchLabels=0)

    filenames = ["infoNocost_testNetwork4QubitIsing_2_3_20.nc_30_1_2022-03-01_07-19-44_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_3_20.nc_30_1_2022-03-02_11-51-19_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_3_20.nc_30_1_2022-03-02_15-58-23_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_3_20.nc_30_1_2022-03-02_17-21-13_config.yaml"]
    labels = ["initial guess [1, 1]", "initial guess random", "initial guess mean from random", "initial guess min cf from random"]
    title = "Network 3 evaluation"
    colors = [blueLight, orangeLight, orangeMedium, orangeDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_init-rand_testNetwork4QubitIsing_2_3_20", title=title, cut=1.0,
                            kirchLabels=0)
    return

    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-03-02_11-45-08_config.yaml",
                extraPlotInfo="number initial_guess after random initial_guess",
                savename="4qubit_COBYLA-numberINITafterRandomINIT_testNetwork4QubitIsing_2_0_20")
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-03-02_11-42-30_config.yaml",
                extraPlotInfo="random initial_guess",
                savename="4qubit_COBYLA-randomINIT_testNetwork4QubitIsing_2_0_20")

    return
    plotCFoptimizationDouble(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-03-02_08-32-15_config.yaml",
                             plotname="COBYLA init = 5,5",
                             savename="4qubit_COBYLA-init5-5_testNetwork4QubitIsing_2_0_20")
    filenames = ["infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-03-01_17-07-37_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-03-02_08-32-15_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-03-02_07-10-21_config.yaml"]
    labels = ["COBYLA init = 1,1", "COBYLA init = 5,5", "COBYLA init = 10,10"]
    title = "classical optimizer feature try out"
    colors = [blueLight, blueMedium, blueDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_COBYLA-init-test2_testNetwork4QubitIsing_2_0_20", title=title, cut=1.0,
                            kirchLabels=0)

    return
    plotCFoptimizationDouble(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-03-02_08-02-58_config.yaml",
                             plotname="COBYLA init = 1,1",
                             savename="4qubit_COBYLA-init1-1_testNetwork4QubitIsing_2_0_20")
    return
    plotCFoptimizationDouble(filename="infoNocost_testNetwork4QubitIsing_2_3_20.nc_30_1_2022-03-02_08-16-36_config.yaml",
                             plotname="COBYLA tol = 0.01",
                             savename="4qubit_COBYLA-tol0.01_testNetwork4QubitIsing_2_3_20")
    plotCFoptimizationDouble(filename="infoNocost_testNetwork4QubitIsing_2_3_20.nc_30_1_2022-03-02_08-18-44_config.yaml",
                             plotname="COBYLA tol = None",
                             savename="4qubit_COBYLA-tol-None_testNetwork4QubitIsing_2_3_20")

    filenames = ["infoNocost_testNetwork4QubitIsing_2_3_20.nc_30_1_2022-03-02_08-12-21_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_3_20.nc_30_1_2022-03-02_08-16-36_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_3_20.nc_30_1_2022-03-02_08-18-44_config.yaml"]
    labels = ["COBYLA tol = 0.0001", "COBYLA tol = 0.01", "COBYLA tol = None"]
    title = "Network 3 - classical optimizer feature try out"
    colors = [blueLight, blueMedium, blueDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_COBYLA-tolTests_testNetwork4QubitIsing_2_3_20", title=title, cut=1.0,
                            kirchLabels=0)

    return
    plotCFoptimizationDouble(filename="infoNocost_testNetwork4QubitIsing_2_3_20.nc_30_1_2022-03-02_08-14-49_config.yaml",
                             plotname="COBYLA init = 10,10",
                             savename="4qubit_COBYLA-init10-10_testNetwork4QubitIsing_2_3_20")

    filenames = ["infoNocost_testNetwork4QubitIsing_2_3_20.nc_30_1_2022-03-02_08-12-21_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_3_20.nc_30_1_2022-03-02_08-14-49_config.yaml"]
    labels = ["COBYLA init = 1,1", "COBYLA init = 10,10"]
    title = "Network 3 - classical optimizer feature try out"
    colors = [blueLight, blueDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_COBYLA-int10-10_testNetwork4QubitIsing_2_3_20", title=title, cut=1.0,
                            kirchLabels=0)

    return
    filenames = ["infoNocost_testNetwork4QubitIsing_2_3_20.nc_30_1_2022-03-02_08-12-21_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_3_20.nc_30_1_2022-03-02_08-10-29_config.yaml"]
    labels = ["COBYLA optimize", "COBYLA minimize"]
    title = "Network 3 - classical optimizer feature try out"
    colors = [blueLight, blueDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_COBYLA-optVSmin_testNetwork4QubitIsing_2_3_20", title=title, cut=1.0,
                            kirchLabels=0)
    return

    plotCFoptimizationDouble(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-03-02_08-02-58_config.yaml",
                             plotname="COBYLA minimize",
                             savename="4qubit_COBYLA-minimize_testNetwork4QubitIsing_2_0_20")

    filenames = ["infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-03-01_17-07-37_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-03-02_08-02-58_config.yaml"]
    labels = ["COBYLA optimize", "COBYLA minimize"]
    title = "classical optimizer feature try out"
    colors = [blueLight, blueDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_COBYLA-optVSmin_testNetwork4QubitIsing_2_0_20", title=title, cut=1.0,
                            kirchLabels=0)
    return
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-03-02_07-54-11_config.yaml",
                extraPlotInfo="COBYLA constraints test",
                savename="4qubit_COBYLA-constraints_testNetwork4QubitIsing_2_0_20")
    return
    plotCFoptimizationDouble(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-03-02_07-21-06_config.yaml",
                             plotname="COBYLA init = 10,10; rhobeg = 5",
                             savename="4qubit_COBYLA-init10-10_rhobeg5_testNetwork4QubitIsing_2_0_20")
    plotCFoptimizationDouble(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-03-02_07-18-58_config.yaml",
                             plotname="COBYLA init = 1,1; rhobeg = 5",
                             savename="4qubit_COBYLA-init1-1_rhobeg5_testNetwork4QubitIsing_2_0_20")

    filenames = ["infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-03-01_17-07-37_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-03-02_07-18-58_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-03-02_07-21-06_config.yaml"]
    labels = ["COBYLA init = 1,1", "COBYLA init = 1,1; rhobeg = 5", "COBYLA init = 10,10; rhobeg = 5"]
    title = "classical optimizer feature try out"
    colors = [blueLight, blueMedium, blueDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_COBYLA-rhobeg-test_testNetwork4QubitIsing_2_0_20", title=title, cut=1.0,
                            kirchLabels=0)
    return

    plotCFoptimizationDouble(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-03-02_07-10-21_config.yaml",
                             plotname="COBYLA init = 10,10",
                             savename="4qubit_COBYLA-init10-10_testNetwork4QubitIsing_2_0_20")

    filenames = ["infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-03-01_17-07-37_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-03-02_07-10-21_config.yaml"]
    labels = ["COBYLA init = 1,1", "COBYLA init = 10,10"]
    title = "classical optimizer feature try out"
    colors = [blueLight, blueDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_COBYLA-init-test_testNetwork4QubitIsing_2_0_20", title=title, cut=1.0,
                            kirchLabels=0)
    return
    filenames = ["infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-03-01_17-07-37_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-03-02_07-04-26_config.yaml"]
    labels = ["COBYLA tol = 0.0001", "COBYLA tol = 0.01"]
    title = "classical optimizer feature try out"
    colors = [blueLight, blueDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_COBYLA-tol-test_testNetwork4QubitIsing_2_0_20", title=title, cut=1.0,
                            kirchLabels=0)

    plotCFoptimizationDouble(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-03-01_17-07-37_config.yaml",
                             plotname="COBYLA tol = 0.0001",
                             savename="4qubit_COBYLA-tol0.0001_testNetwork4QubitIsing_2_0_20")

    plotCFoptimizationDouble(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-03-02_07-04-26_config.yaml",
                             plotname="COBYLA tol = 0.01",
                             savename="4qubit_COBYLA-tol0.01_testNetwork4QubitIsing_2_0_20")
    return
    filenames = ["infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-03-01_17-07-37_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-03-01_16-48-59_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-03-01_16-31-10_config.yaml"]
    labels = ["COBYLA", "SPSA blocking = False", "SPSA blocking = True"]
    title = "classical optimizer feature try out"
    colors = [blueLight, orangeLight, orangeDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_SPSA-new-features_testNetwork4QubitIsing_2_0_20", title=title, cut=1.0,
                            kirchLabels=0)
    return
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-03-01_16-31-10_config.yaml",
                extraPlotInfo="SPSA blocking = True",
                savename="4qubit_SPSA-blocking-True_testNetwork4QubitIsing_2_0_20")

    return

    filenames = ["infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-25_10-50-04_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-03-01_07-19-44_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-03-01_11-57-09_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-03-01_10-14-01_config.yaml"]
    labels = ["IterMatrix - unscaled", "IterMatrix - scaled", "IterMatrix - 2Hp", "Ising - scaled"]
    title = "Network 0 evaluation"
    colors = [blueLight, orangeLight, orangeMedium, orangeDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_scaled-unscaled_testNetwork4QubitIsing_2_0_20", title=title, cut=1.0,
                            kirchLabels=0)

    filenames = ["infoNocost_testNetwork4QubitIsing_2_1_20.nc_30_1_2022-02-25_10-14-41_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_1_20.nc_30_1_2022-03-01_07-19-44_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_1_20.nc_30_1_2022-03-01_11-57-09_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_1_20.nc_30_1_2022-03-01_10-14-01_config.yaml"]
    labels = ["IterMatrix - unscaled", "IterMatrix - scaled", "IterMatrix - 2Hp", "Ising - scaled"]
    title = "Network 1 evaluation"
    colors = [blueLight, orangeLight, orangeMedium, orangeDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_scaled-unscaled_testNetwork4QubitIsing_2_1_20", title=title, cut=1.0,
                            kirchLabels=0)

    filenames = ["infoNocost_testNetwork4QubitIsing_2_2_20.nc_30_1_2022-02-25_10-14-41_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_2_20.nc_30_1_2022-03-01_07-19-44_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_2_20.nc_30_1_2022-03-01_11-57-09_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_2_20.nc_30_1_2022-03-01_10-14-01_config.yaml"]
    labels = ["IterMatrix - unscaled", "IterMatrix - scaled", "IterMatrix - 2Hp", "Ising - scaled"]
    title = "Network 2 evaluation"
    colors = [blueLight, orangeLight, orangeMedium, orangeDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_scaled-unscaled_testNetwork4QubitIsing_2_2_20", title=title, cut=1.0,
                            kirchLabels=0)

    filenames = ["infoNocost_testNetwork4QubitIsing_2_3_20.nc_30_1_2022-02-25_11-06-22_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_3_20.nc_30_1_2022-03-01_07-19-44_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_3_20.nc_30_1_2022-03-01_11-57-09_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_3_20.nc_30_1_2022-03-01_10-14-01_config.yaml"]
    labels = ["IterMatrix - unscaled", "IterMatrix - scaled", "IterMatrix - 2Hp", "Ising - scaled"]
    title = "Network 3 evaluation"
    colors = [blueLight, orangeLight, orangeMedium, orangeDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_scaled-unscaled_testNetwork4QubitIsing_2_3_20", title=title, cut=1.0,
                            kirchLabels=0)

    return
    filenames = ["infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-25_10-50-04_config.yaml"]
    labels = ["network 0"]
    title = "Network 0 evaluation"
    colors = [blueLight]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_testNetwork4QubitIsing_2_0_20", title=title, cut=1.0,
                            kirchLabels=0)
    plt.clf()
    plotEigenvalues(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-25_10-50-04_config.yaml",
                    plotname="Eigenvalues of the hamiltonian matrix of network 0",
                    savename="4qubit_testNetwork4QubitIsing_2_0_20")

    filenames = ["infoNocost_testNetwork4QubitIsing_2_1_20.nc_30_1_2022-02-25_10-14-41_config.yaml"]
    labels = ["network 1"]
    title = "Network 1 evaluation"
    colors = [blueDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_testNetwork4QubitIsing_2_1_20", title=title, cut=1.0,
                            kirchLabels=0)
    plt.clf()
    plotEigenvalues(filename="infoNocost_testNetwork4QubitIsing_2_1_20.nc_30_1_2022-02-25_10-14-41_config.yaml",
                    plotname="Eigenvalues of the hamiltonian matrix of network 1",
                    savename="4qubit_testNetwork4QubitIsing_2_1_20")

    filenames = ["infoNocost_testNetwork4QubitIsing_2_2_20.nc_30_1_2022-02-25_10-14-41_config.yaml"]
    labels = ["network 2"]
    title = "Network 2 evaluation"
    colors = [orangeLight]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_testNetwork4QubitIsing_2_2_20", title=title, cut=1.0,
                            kirchLabels=0)
    plt.clf()
    plotEigenvalues(filename="infoNocost_testNetwork4QubitIsing_2_2_20.nc_30_1_2022-02-25_10-14-41_config.yaml",
                    plotname="Eigenvalues of the hamiltonian matrix of network 2",
                    savename="4qubit_testNetwork4QubitIsing_2_2_20")

    filenames = ["infoNocost_testNetwork4QubitIsing_2_3_20.nc_30_1_2022-02-25_11-06-22_config.yaml"]
    labels = ["network 3"]
    title = "Network 3 evaluation"
    colors = [orangeDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_testNetwork4QubitIsing_2_3_20", title=title, cut=1.0,
                            kirchLabels=0)
    plt.clf()
    plotEigenvalues(filename="infoNocost_testNetwork4QubitIsing_2_3_20.nc_30_1_2022-02-25_11-06-22_config.yaml",
                    plotname="Eigenvalues of the hamiltonian matrix of network 3",
                    savename="4qubit_testNetwork4QubitIsing_2_3_20")

    return

    plotEigenvalues(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-25_11-23-47_config.yaml",
                    plotname="Eigenvalues of the hamiltonian matrix",
                    savename="4qubit_after-bugfix_testNetwork4QubitIsing_2_0_20")

    return

    filenames = ["infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-25_10-50-04_config.yaml"]
    labels = ["after bugfix"]
    title = "check bugfix"
    colors = [orangeMedium]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_after-bugfix_testNetwork4QubitIsing_2_0_20", title=title, cut=1.0,
                            kirchLabels=0)

    filenames = ["infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-25_11-23-47_config.yaml"]
    labels = ["before bugfix"]
    title = "check bugfix"
    colors = [blueMedium]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_before-bugfix_testNetwork4QubitIsing_2_0_20", title=title, cut=1.0,
                            kirchLabels=0)

    return

    filenames = ["infoNocost_testNetwork4QubitIsing_2_1_20.nc_30_1_2022-02-25_12-04-27_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_1_20.nc_30_1_2022-02-25_10-14-41_config.yaml"]
    labels = ["before bugfix", "after bugfux"]
    title = "check bugfix"
    colors = [blueMedium, orangeMedium]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_before-after-bugfix_testNetwork4QubitIsing_2_1_20", title=title, cut=0.5,
                            kirchLabels=1)
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_before-after-bugfix_testNetwork4QubitIsing_2_1_20", title=title, cut=1.0,
                            kirchLabels=1)

    filenames = ["infoNocost_testNetwork4QubitIsing_2_2_20.nc_30_1_2022-02-25_12-04-27_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_2_20.nc_30_1_2022-02-25_10-14-41_config.yaml"]
    labels = ["before bugfix", "after bigfix"]
    title = "check bugfix"
    colors = [blueMedium, orangeMedium]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_before-after-bugfix_testNetwork4QubitIsing_2_2_20", title=title, cut=0.5,
                            kirchLabels=1)
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_before-after-bugfix_testNetwork4QubitIsing_2_2_20", title=title, cut=1.0,
                            kirchLabels=1)

    filenames = ["infoNocost_testNetwork4QubitIsing_2_3_20.nc_30_1_2022-02-25_12-04-27_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_3_20.nc_30_1_2022-02-25_11-06-22_config.yaml"]
    labels = ["before bugfix", "after bugfix"]
    title = "check bugfix"
    colors = [blueMedium, orangeMedium]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_before-after-bugfix_testNetwork4QubitIsing_2_3_20", title=title, cut=0.5,
                            kirchLabels=1)
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_before-after-bugfix_testNetwork4QubitIsing_2_3_20", title=title, cut=1.0,
                            kirchLabels=1)

    return

    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-25_10-50-04_config.yaml",
                extraPlotInfo="after bugfix",
                savename="AER_4QubitIsing_2_0_20.nc_after-bugfix")
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-25_11-23-47_config.yaml",
                extraPlotInfo="before bugfix",
                savename="AER_4QubitIsing_2_0_20.nc_before-bugfix")
    filenames = ["infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-25_11-23-47_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-25_10-50-04_config.yaml"]
    labels = ["before bugfix", "after bugfix"]
    title = "check bugfix"
    colors = [blueMedium, orangeMedium]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_before-after-bugfix", title=title, cut=0.5)
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_before-after-bugfix", title=title, cut=1.0)

    return
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_3_20.nc_30_1_2022-02-25_11-06-22_config.yaml",
                extraPlotInfo="new 4Qubit network",
                savename="AER_4QubitIsing_2_3_20.nc_IterMatrix_COBYLA_maxiter100_shots20000_rep100")
    return
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-25_10-50-04_config.yaml",
                extraPlotInfo="",
                savename="AER_4QubitIsing_2_0_20.nc_IterMatrix_COBYLA_maxiter100_shots20000_rep100")
    return
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_2_20.nc_30_1_2022-02-25_10-14-41_config.yaml",
                extraPlotInfo="new 4Qubit network",
                savename="AER_4QubitIsing_2_2_20.nc_IterMatrix_COBYLA_maxiter100_shots20000_rep100")
    return
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_1_20.nc_30_1_2022-02-25_10-14-41_config.yaml",
                extraPlotInfo="new 4Qubit network",
                savename="AER_4QubitIsing_2_1_20.nc_IterMatrix_COBYLA_maxiter100_shots20000_rep100")

    return
    plotScatter(file1="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_65.yaml",
                file2="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_71.yaml",
                title="cost function vs ideal solution probability of COBYLA and SPSA100 with noise",
                g1title="COBYLA", g2title="SPSA100",
                savename="COBYLA-SPSA100_yesNoise_CFvsTIME",
                mode="opt", x="cf", y="duration")

    plotScatter(file1="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_65.yaml",
                file2="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_71.yaml",
                title="cost function vs ideal solution probability of COBYLA and SPSA100 with noise",
                g1title="COBYLA", g2title="SPSA100",
                savename="COBYLA-SPSA100_yesNoise_CFvsPROP",
                mode="opt", x="cf", y="0101prop")

    return
    #test v2
    filenames = ["infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_72.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_73.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_74.yaml"]
    labels = ["1024", "4096", "20,000"]
    title = "SPSA200 without noise with different number of shots"
    colors = [orangeLight, orangeMedium, orangeDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_SPSA200_noNoise_1024_4096_20000", title=title, cut=0.5)
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_SPSA200_noNoise_1024_4096_20000", title=title, cut=1.0)

    filenames = ["infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_75.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_76.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_77.yaml"]
    labels = ["1024", "4096", "20,000"]
    title = "SPSA200 with noise with different number of shots"
    colors = [blueLight, blueMedium, blueDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_SPSA200_yesNoise_1024_4096_20000", title=title, cut=0.5)
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_SPSA200_yesNoise_1024_4096_20000", title=title, cut=1.0)

    return
    filenames = ["infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_63.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_69.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_75.yaml"]
    labels = ["COBYLA", "SPSA100", "SPSA200"]
    title = "classical optimizer comparision with noise"
    colors = [blueLight, blueMedium, blueDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_COBYLA_SPSA100_SPSA200_yesNoise_1024", title=title, cut=0.5)
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_COBYLA_SPSA100_SPSA200_yesNoise_1024", title=title, cut=1.0)

    filenames = ["infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_64.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_70.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_76.yaml"]
    labels = ["COBYLA", "SPSA100", "SPSA200"]
    title = "classical optimizer comparision with noise"
    colors = [blueLight, blueMedium, blueDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_COBYLA_SPSA100_SPSA200_yesNoise_4096", title=title, cut=0.5)
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_COBYLA_SPSA100_SPSA200_yesNoise_4096", title=title, cut=1.0)

    filenames = ["infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_65.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_71.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_77.yaml"]
    labels = ["COBYLA", "SPSA100", "SPSA200"]
    title = "classical optimizer comparision with noise"
    colors = [blueLight, blueMedium, blueDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_COBYLA_SPSA100_SPSA200_yesNoise_20000", title=title, cut=0.5)
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_COBYLA_SPSA100_SPSA200_yesNoise_20000", title=title, cut=1.0)

    return
    filenames = ["infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_60.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_61.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_62.yaml"]
    labels = ["1024", "4096", "20,000"]
    title = "COBYLA without noise with different number of shots"
    colors = [orangeLight, orangeMedium, orangeDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_COBYLA_noNoise_1024_4096_20000", title=title, cut=0.5)
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_COBYLA_noNoise_1024_4096_20000", title=title, cut=1.0)

    filenames = ["infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_63.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_64.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_65.yaml"]
    labels = ["1024", "4096", "20,000"]
    title = "COBYLA with noise with different number of shots"
    colors = [blueLight, blueMedium, blueDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_COBYLA_yesNoise_1024_4096_20000", title=title, cut=0.5)
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_COBYLA_yesNoise_1024_4096_20000", title=title, cut=1.0)

    filenames = ["infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_66.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_67.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_68.yaml"]
    labels = ["1024", "4096", "20,000"]
    title = "SPSA100 without noise with different number of shots"
    colors = [orangeLight, orangeMedium, orangeDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_SPSA100_noNoise_1024_4096_20000", title=title, cut=0.5)
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_SPSA100_noNoise_1024_4096_20000", title=title, cut=1.0)

    filenames = ["infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_69.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_70.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_71.yaml"]
    labels = ["1024", "4096", "20,000"]
    title = "SPSA100 with noise with different number of shots"
    colors = [blueLight, blueMedium, blueDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_SPSA100_yesNoise_1024_4096_20000", title=title, cut=0.5)
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_SPSA100_yesNoise_1024_4096_20000", title=title, cut=1.0)

    return

    filenames = ["infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_60.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_66.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_72.yaml"]
    labels = ["COBYLA", "SPSA100", "SPSA200"]
    title = "classical optimizer comparision without noise"
    colors = [orangeLight, orangeMedium, orangeDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_COBYLA_SPSA100_SPSA200_noNoise_1024", title=title, cut=0.5)
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_COBYLA_SPSA100_SPSA200_noNoise_1024", title=title, cut=1.0)

    filenames = ["infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_61.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_67.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_73.yaml"]
    labels = ["COBYLA", "SPSA100", "SPSA200"]
    title = "classical optimizer comparision without noise"
    colors = [orangeLight, orangeMedium, orangeDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_COBYLA_SPSA100_SPSA200_noNoise_4096", title=title, cut=0.5)
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_COBYLA_SPSA100_SPSA200_noNoise_4096", title=title, cut=1.0)

    filenames = ["infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_62.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_68.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_74.yaml"]
    labels = ["COBYLA", "SPSA100", "SPSA200"]
    title = "classical optimizer comparision without noise"
    colors = [orangeLight, orangeMedium, orangeDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_COBYLA_SPSA100_SPSA200_noNoise_20000", title=title, cut=0.5)
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_COBYLA_SPSA100_SPSA200_noNoise_20000", title=title, cut=1.0)

    return
    filenames = ["infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-17_17-21-18_config_17.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-17_17-21-18_config_19.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-17_17-21-18_config_21.yaml"]
    labels = ["COBYLA", "SPSA100", "SPSA200"]
    title = "classical optimizer comparision with noise"
    colors = [blueLight, blueMedium, blueDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_COBYLA_SPSA100_SPSA200_yesNoise", title=title)

    filenames = ["infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-17_17-21-18_config_16.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-17_17-21-18_config_18.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-17_17-21-18_config_20.yaml"]
    labels = ["COBYLA", "SPSA100", "SPSA200"]
    title = "classical optimizer comparision without noise"
    colors = [orangeLight, orangeMedium, orangeDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_COBYLA_SPSA100_SPSA200_noNoise", title=title)

    return
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-18_17-42-55_config_44.yaml",
                extraPlotInfo="",
                savename="QPU_4qubit_IterMatrix_COBYLA_g1-1_g2-3_maxiter50_shots20000_rep10")
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-18_17-42-55_config_45.yaml",
                extraPlotInfo="",
                savename="QPU_4qubit_IterMatrix_SPSA_g1-1_g2-3_maxiter50_shots20000_rep10")
    return
    filenames = ["infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-17_17-21-18_config_16.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-17_17-21-18_config_18.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-17_17-21-18_config_20.yaml"]
    plotOptTime_CF(filenames=filenames, savename="4qubit_optTime_CF")

    return
    filenamesNnoise = ["infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-17_17-21-18_config_08.yaml",
                       "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-17_17-21-18_config_09.yaml",
                       "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-17_17-21-18_config_10.yaml",
                       "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-17_17-21-18_config_11.yaml",
                       "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-17_17-21-18_config_12.yaml",
                       "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-17_17-21-18_config_13.yaml",
                       "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-17_17-21-18_config_14.yaml",
                       "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-18_13-47-31_config_40.yaml",
                       "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-18_13-47-31_config_42.yaml"]
    filenamesYnoise = ["infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_50.yaml",
                       "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_51.yaml",
                       "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_52.yaml",
                       "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_53.yaml",
                       "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_54.yaml",
                       "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_55.yaml",
                       "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_56.yaml",
                       "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_57.yaml",
                       "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_58.yaml"]
    plotShots_CF(filenamesNnoise=filenamesNnoise, filenamesYnoise=filenamesYnoise, savename="4qubit_shots_CF")
    return
    filenames = ["infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-18_08-34-00_config_23.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-18_08-34-00_config_22.yaml"]
    labels = ["QASM", "AER"]
    title = "Simulator comparision with noise"
    colors = [blueLight, blueDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_QASM_AER_yesNoise", title=title)

    filenames = ["infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-17_17-21-18_config_01.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-17_17-21-18_config_06.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-17_17-21-18_config_05.yaml"]
    labels = ["QASM", "Statevector", "AER - Statevector"]
    title = "Simulator comparision without noise"
    colors = [orangeLight, orangeMedium, orangeDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_AerState_State_QASM_noNoise", title=title)

    filenames = ["infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-17_17-21-18_config_01.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-17_17-21-18_config_03.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-17_17-21-18_config_02.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-17_17-21-18_config_04.yaml"]
    labels = ["Matrix w/o noise", "Iteration w/o noise", "Matrix w/ noise", "Iteration w/ noise"]
    colors = [orangeLight, blueLight, orangeDark, blueDark]
    title = "Comparision between iteration QC and matrix QC"
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_Iteration_vs_Matrix", title=title)

    return
    # optimizer tests
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-23_16-43-37_config.yaml",
                extraPlotInfo="CRS",
                savename="CRS_4qubit_rep10")
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-23_11-31-00_config.yaml",
                extraPlotInfo="COBYLA new",
                savename="COBYLA_new_4qubit_rep10")
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-23_11-08-27_config.yaml",
                extraPlotInfo="COBYLA old",
                savename="COBYLA_old_4qubit_rep10")
    filenames = ["infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-23_11-08-27_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-23_11-31-00_config.yaml"]
    labels = ["COBYLA old", "COBYLA new"]
    title = "COBYLA comparision with noise"
    colors = [blueLight, blueMedium]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_COBYLAold-COBYLAnew_yesNoise", title=title, cut=0.5)
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_COBYLAold-COBYLAnew_yesNoise", title=title, cut=1.0)
    return
    #tests v1
    filenames = ["infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-15_16-13-00_config_02.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-15_16-13-00_config_10.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-15_16-13-00_config_12.yaml"]
    labels = ["COBYLA w/ noise", "SPSA100 w/ noise", "SPSA200 w/ noise"]
    title = ""
    colors = [blueLight, blueMedium, blueDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors, savename="4qubit_COBYLA_SPSA100_200_yesNoise")

    filenames = ["infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-15_16-13-00_config_01.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-15_16-13-00_config_09.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-15_16-13-00_config_11.yaml"]
    labels = ["COBYLA w/o noise", "SPSA100 w/o noise", "SPSA200 w/o noise"]
    colors = [orangeLight, orangeMedium, orangeDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors, savename="4qubit_COBYLA_SPSA100_200_noNoise")

    filenames = ["infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-15_16-13-00_config_02.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-15_16-13-00_config_06.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-15_16-13-00_config_08.yaml"]
    labels = ["AER-simulator w/ noise", "QASM-simulater w/ noise", "statevector-simulator w/ noise"]
    colors = [blueLight, blueMedium, blueDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors, savename="4qubit_aer_qasm_state_yesNoise")

    filenames = ["infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-15_16-13-00_config_01.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-15_16-13-00_config_05.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-15_16-13-00_config_07.yaml"]
    labels = ["AER-simulator w/o noise", "QASM-simulater w/o noise", "statevector-simulator w/o noise"]
    colors = [orangeLight, orangeMedium, orangeDark]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors, savename="4qubit_aer_qasm_state_noNoise")

    return
    # initial guess + Hp, Hb vs 2Hp, 2Hb
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-15_15-50-37_config_01.yaml",
                extraPlotInfo="g1=1, g2=3, IterMatrix QC - 2Hp & 2Hb",
                savename="aer_4qubit_IterMatrix_2Hp-2Hb_COBYLA_g1-1_g2-3_noNoise_maxiter50_shots4096_rep10_initial1")
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-15_15-50-37_config_02.yaml",
                extraPlotInfo="g1=1, g2=3, IterMatrix QC - 2Hp & 2Hb",
                savename="aer_4qubit_IterMatrix_2Hp-2Hb_COBYLA_g1-1_g2-3_noNoise_maxiter50_shots4096_rep10_initial2")
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-15_15-50-37_config_03.yaml",
                extraPlotInfo="g1=1, g2=3, IterMatrix QC - 2Hp & 2Hb",
                savename="aer_4qubit_IterMatrix_2Hp-2Hb_COBYLA_g1-1_g2-3_noNoise_maxiter50_shots4096_rep10_initial3")
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-15_15-50-37_config_04.yaml",
                extraPlotInfo="g1=1, g2=3, IterMatrix QC - 2Hp & 2Hb",
                savename="aer_4qubit_IterMatrix_2Hp-2Hb_COBYLA_g1-1_g2-3_noNoise_maxiter50_shots4096_rep10_initial4")

    return
    #initial guess
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-15_15-39-59_config_01.yaml",
                extraPlotInfo="g1=1, g2=3, IterMatrix QC",
                savename="aer_4qubit_IterMatrix_COBYLA_g1-1_g2-3_noNoise_maxiter50_shots4096_rep10_initial1")
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-15_15-39-59_config_02.yaml",
                extraPlotInfo="g1=1, g2=3, IterMatrix QC",
                savename="aer_4qubit_IterMatrix_COBYLA_g1-1_g2-3_noNoise_maxiter50_shots4096_rep10_initial2")
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-15_15-39-59_config_03.yaml",
                extraPlotInfo="g1=1, g2=3, IterMatrix QC",
                savename="aer_4qubit_IterMatrix_COBYLA_g1-1_g2-3_noNoise_maxiter50_shots4096_rep10_initial3")
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-15_15-39-59_config_04.yaml",
                extraPlotInfo="g1=1, g2=3, IterMatrix QC",
                savename="aer_4qubit_IterMatrix_COBYLA_g1-1_g2-3_noNoise_maxiter50_shots4096_rep10_initial4")

    return
    #Hp, Hb vs 2Hp, 2Hb
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-15_15-17-55_config_04.yaml",
                extraPlotInfo="g1=1, g2=3, IterMatrix QC - 2Hp & 2Hb",
                savename="aer_4qubit_IterMatrix_2Hp-2Hb_COBYLA_g1-1_g2-3_yesNoise_maxiter50_shots4096_rep10")
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-15_15-17-55_config_02.yaml",
                extraPlotInfo="g1=1, g2=3, IterMatrix QC - 2Hp & 2Hb",
                savename="aer_4qubit_IterMatrix_2Hp-2Hb_COBYLA_g1-1_g2-3_noNoise_maxiter50_shots4096_rep10")
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-15_15-17-55_config_03.yaml",
                extraPlotInfo="g1=1, g2=3, IterMatrix QC",
                savename="aer_4qubit_IterMatrix_COBYLA_g1-1_g2-3_yesNoise_maxiter50_shots4096_rep10")
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-15_15-17-55_config_01.yaml",
                extraPlotInfo="g1=1, g2=3, IterMatrix QC",
                savename="aer_4qubit_IterMatrix_COBYLA_g1-1_g2-3_noNoise_maxiter50_shots4096_rep10")

    return
    #state, qasm, aer
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-15_11-09-48",
                extraPlotInfo="g1=1, g2=3, IterMatrix QC - kirch^2",
                savename="aer_4qubit_IterMatrix-kirch^2_2Hp-2Hb_COBYLA_g1-1_g2-3_yesNoise_maxiter50_shots4096_rep10")
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-15_11-11-03",
                extraPlotInfo="g1=1, g2=3, IterMatrix QC - kirch^2",
                savename="aer_4qubit_IterMatrix-kirch^2_2Hp-2Hb_COBYLA_g1-1_g2-3_noNoise_maxiter50_shots4096_rep10")
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-15_11-05-52",
                extraPlotInfo="g1=1, g2=3, IterMatrix QC - kirch^2",
                savename="qasm_4qubit_IterMatrix-kirch^2_2Hp-2Hb_COBYLA_g1-1_g2-3_yesNoise_maxiter50_shots4096_rep10")
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-15_11-06-16",
                extraPlotInfo="g1=1, g2=3, IterMatrix QC - kirch^2",
                savename="qasm_4qubit_IterMatrix-kirch^2_2Hp-2Hb_COBYLA_g1-1_g2-3_noNoise_maxiter50_shots4096_rep10")
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-15_11-05-38",
                extraPlotInfo="g1=1, g2=3, IterMatrix QC - kirch^2",
                savename="state_4qubit_IterMatrix-kirch^2_2Hp-2Hb_COBYLA_g1-1_g2-3_yesNoise_maxiter50_shots4096_rep10")
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-15_11-06-38",
                extraPlotInfo="g1=1, g2=3, IterMatrix QC - kirch^2",
                savename="state_4qubit_IterMatrix-kirch^2_2Hp-2Hb_COBYLA_g1-1_g2-3_noNoise_maxiter50_shots4096_rep10")

    return
    #5qubit
    plotBPandCF(filename="infoNocost_testNetwork5QubitIsing_2_0_20.nc_30_1_2022-02-15_09-25-00",
                extraPlotInfo="g1=2, g2=4, g3=2, IterMatrix QC - 2Hp & 2Hb",
                savename="aer_5qubit_IterMatrix-kirch^2_2Hp-2Hb_COBYLA_g1-2_g2-4_g3-2_noNoise_maxiter50_shots4096_rep10")
    plotBPandCF(filename="infoNocost_testNetwork5QubitIsing_2_0_20.nc_30_1_2022-02-15_09-30-47",
                extraPlotInfo="g1=2, g2=4, g3=2, IterMatrix QC - 2Hp & 2Hb",
                savename="aer_5qubit_IterMatrix-kirch^2_2Hp-2Hb_COBYLA_g1-2_g2-4_g3-2_yesNoise_maxiter50_shots4096_rep10")
    plotBPandCF(filename="infoNocost_testNetwork5QubitIsing_2_0_20.nc_30_1_2022-02-15_09-30-19",
                extraPlotInfo="g1=2, g2=4, g3=2, IterMatrix QC - 2Hp & 2Hb",
                savename="aer_5qubit_IterMatrix-kirch^2_2Hp-2Hb_SPSA_g1-2_g2-4_g3-2_noNoise_maxiter50_shots4096_rep10")
    plotBPandCF(filename="infoNocost_testNetwork5QubitIsing_2_0_20.nc_30_1_2022-02-15_09-30-31",
                extraPlotInfo="g1=2, g2=4, g3=2, IterMatrix QC - 2Hp & 2Hb",
                savename="aer_5qubit_IterMatrix-kirch^2_2Hp-2Hb_SPSA_g1-2_g2-4_g3-2_yesNoise_maxiter50_shots4096_rep10")

    return
    plotHistCF(docker=True, filename="infoNocost_testNetwork5QubitIsing_2_0_20.nc_30_1_2022-02-15_09-25-00",
               plotname="COBYLA no noise \n g1=2, g2=4, g3=2, IterMatrix QC - 2Hp & 2Hb",
               savename="aer_5qubit_IterMatrix-kirch^2_2Hp-2Hb_COBYLA_g1-2_g2-4_g3-2_noNoise_maxiter50_shots4096_rep10")
    plotHistCF(docker=True, filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-14_17-53-54",
               plotname="SPSA with noise \n g1=1, g2=3, IterMatrix QC - kirch^2",
               savename="aer_4qubit_IterMatrix-kirch^2_SPSA_g1-1_g2-3_yesNoise_maxiter50_shots4096_rep100")
    plotBoxplotBest(docker=True, filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-14_17-53-54",
                    plotname="SPSA with noise \n g1=1, g2=3, IterMatrix QC - kirch^2",
                    savename="aer_4qubit_IterMatrix-kirch^2_SPSA_g1-1_g2-3_yesNoise_maxiter50_shots4096_rep100")
    return

    plotBPandCF(filename="infoNocost_testNetwork5QubitIsing_2_0_20.nc_30_1_2022-02-15_08-27-39",
                extraPlotInfo="g1=2, g2=4, g3=2, IterMatrix QC - kirch^2",
                savename="aer_5qubit_IterMatrix-kirch^2_SPSA_g1-2_g2-4_g3-2_yesNoise_maxiter50_shots4096_rep10")
    plotBPandCF(filename="infoNocost_testNetwork5QubitIsing_2_0_20.nc_30_1_2022-02-15_08-27-29",
                extraPlotInfo="g1=2, g2=4, g3=2, IterMatrix QC - kirch^2",
                savename="aer_5qubit_IterMatrix-kirch^2_SPSA_g1-2_g2-4_g3-2_noNoise_maxiter50_shots4096_rep10")
    plotBPandCF(filename="infoNocost_testNetwork5QubitIsing_2_0_20.nc_30_1_2022-02-15_08-28-00",
                extraPlotInfo="g1=2, g2=4, g3=2, IterMatrix QC - kirch^2",
                savename="aer_5qubit_IterMatrix-kirch^2_COBYLA_g1-2_g2-4_g3-2_yesNoise_maxiter50_shots4096_rep10")
    plotBPandCF(filename="infoNocost_testNetwork5QubitIsing_2_0_20.nc_30_1_2022-02-15_08-28-21",
                extraPlotInfo="g1=2, g2=4, g3=2, IterMatrix QC - kirch^2",
                savename="aer_5qubit_IterMatrix-kirch^2_COBYLA_g1-2_g2-4_g3-2_noNoise_maxiter50_shots4096_rep10")

    return
    #Iteration vs IterastionMatrix
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-14_17-53-54",
                extraPlotInfo="g1=1, g2=3, IterMatrix QC - kirch^2",
                savename="aer_4qubit_IterMatrix-kirch^2_SPSA_g1-1_g2-3_yesNoise_maxiter50_shots4096_rep100")
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-14_17-51-06",
                extraPlotInfo="g1=1, g2=3, IterMatrix QC - kirch^2",
                savename="aer_4qubit_IterMatrix-kirch^2_SPSA_g1-1_g2-3_noNoise_maxiter50_shots4096_rep100")
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-14_17-54-17",
                extraPlotInfo="g1=1, g2=3, Iter QC - kirch^2",
                savename="aer_4qubit_Iter-kirch^2_SPSA_g1-1_g2-3_yesNoise_maxiter50_shots4096_rep100")
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-14_17-55-00",
                extraPlotInfo="g1=1, g2=3, Iter QC - kirch^2",
                savename="aer_4qubit_Iter-kirch^2_SPSA_g1-1_g2-3_noNoise_maxiter50_shots4096_rep100")

    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-14_15-33-38",
                extraPlotInfo="g1=1, g2=3, IterMatrix QC - kirch^2",
                savename="aer_4qubit_IterMatrix-kirch^2_COBYLA_g1-1_g2-3_yesNoise_maxiter50_shots4096_rep100")
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-14_15-33-27",
                extraPlotInfo="g1=1, g2=3, IterMatrix QC - kirch^2",
                savename="aer_4qubit_IterMatrix-kirch^2_COBYLA_g1-1_g2-3_noNoise_maxiter50_shots4096_rep100")
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-14_15-34-18",
                extraPlotInfo="g1=1, g2=3, Iter QC - kirch^2",
                savename="aer_4qubit_Iter-kirch^2_COBYLA_g1-1_g2-3_yesNoise_maxiter50_shots4096_rep100")
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-14_15-33-53",
                extraPlotInfo="g1=1, g2=3, Iter QC - kirch^2",
                savename="aer_4qubit_Iter-kirch^2_COBYLA_g1-1_g2-3_noNoise_maxiter50_shots4096_rep100")

    return
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-14_15-18-35",
                extraPlotInfo="g1=1, g2=3, Iter QC - kirch^2",
                savename="aer_4qubit_IterMatrix-kirch^2_g1-1_g2-3_yesNoise_maxiter50_shots4096_rep10")
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-14_15-18-25",
                extraPlotInfo="g1=1, g2=3, Iter QC - kirch^2",
                savename="aer_4qubit_IterMatrix-kirch^2_g1-1_g2-3_noNoise_maxiter50_shots4096_rep10")

    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-14_14-37-58",
                extraPlotInfo="g1=1, g2=3, Ising QC - kirch^2",
                savename="aer_4qubit_Ising-kirch^2_SPSA_g1-1_g2-3_yesNoise_maxiter50_shots4096_rep10")
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-14_14-31-52",
                extraPlotInfo="g1=1, g2=3, Ising QC - kirch^2",
                savename="aer_4qubit_Ising-kirch^2_SPSA_g1-1_g2-3_noNoise_maxiter50_shots4096_rep10")

    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-14_14-20-07",
                extraPlotInfo="g1=1, g2=3, Iter QC - kirch^2",
                savename="aer_4qubit_Iter-kirch^2_SPSA_g1-1_g2-3_yesNoise_maxiter50_shots4096_rep10")
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-14_14-19-54",
                extraPlotInfo="g1=1, g2=3, Iter QC - kirch^2",
                savename="aer_4qubit_Iter-kirch^2_SPSA_g1-1_g2-3_noNoise_maxiter50_shots4096_rep10")

    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-14_12-36-35",
                extraPlotInfo="g1=1, g2=3, Iter QC - kirch^2",
                savename="aer_4qubit_Iter-kirch^2_g1-1_g2-3_yesNoise_maxiter50_shots4096_rep100")
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-14_12-36-52",
                extraPlotInfo="g1=1, g2=3, Iter QC - kirch^2",
                savename="aer_4qubit_Iter-kirch^2_g1-1_g2-3_noNoise_maxiter50_shots4096_rep100")

    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-14_12-33-40",
                extraPlotInfo="g1=1, g2=3, Iter QC - kirch^2",
                savename="aer_4qubit_Iter-kirch^2_g1-1_g2-3_yesNoise_maxiter50_shots4096_rep10")
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-14_12-29-11",
                extraPlotInfo="g1=1, g2=3, Iter QC - kirch^2",
                savename="aer_4qubit_Iter-kirch^2_g1-1_g2-3_noNoise_maxiter50_shots4096_rep10")

    return
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-14_09-41-18",
                extraPlotInfo="g1=1, g2=3, Ising QC - +Hamiltonian",
                savename="aer_4qubit_Ising-posHam_g1-1_g2-3_noNoise_fixedQC_maxiter50_shots4096_rep10")
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-14_09-45-47",
                extraPlotInfo="g1=1, g2=3, Ising QC - -Hamiltonian",
                savename="aer_4qubit_Ising-negHam_g1-1_g2-3_noNoise_fixedQC_maxiter50_shots4096_rep10")
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-14_10-26-15",
                extraPlotInfo="g1=1, g2=3, Ising QC - -Hamiltonian, [2][3] = 4",
                savename="aer_4qubit_Ising-negHam-4_g1-1_g2-3_noNoise_fixedQC_maxiter50_shots4096_rep10")
    return
    # real run
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-11_17-02-31",
                extraPlotInfo="g1=1, g2=3",
                savename="QPU_4qubit_Ising_g1-1_g2-3_maxiter50_shots4096_rep100")
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-11_17-02-00",
                extraPlotInfo="g1=1, g2=3",
                savename="QPU_4qubit_Ising_g1-1_g2-3_maxiter50_shots4096_rep10")
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-11_17-01-18",
                extraPlotInfo="g1=1, g2=3",
                savename="QPU_4qubit_Ising_g1-1_g2-3_maxiter50_shots4096_rep1")
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-11_17-04-24",
                extraPlotInfo="g1=1, g2=3",
                savename="QPU_4qubit_Iteration_g1-1_g2-3_maxiter50_shots4096_rep100")
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-11_17-03-57",
                extraPlotInfo="g1=1, g2=3",
                savename="QPU_4qubit_Iteration_g1-1_g2-3_maxiter50_shots4096_rep10")
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-11_17-03-28",
                extraPlotInfo="g1=1, g2=3",
                savename="QPU_4qubit_Iteration_g1-1_g2-3_maxiter50_shots4096_rep1")

    return

    # Ising vs Iteration
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-11_16-43-28",
                extraPlotInfo="g1=1, g2=3, Ising QC",
                savename="aer_4qubit_Ising_g1-1_g2-3_noNoise_fixedQC_maxiter50_shots4096_rep100")
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-11_16-43-39",
                extraPlotInfo="g1=1, g2=3, Ising QC",
                savename="aer_4qubit_Ising_g1-1_g2-3_yesNoise_fixedQC_maxiter50_shots4096_rep100")
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-11_14-07-23",
                extraPlotInfo="g1=1, g2=3, Iteration QC",
                savename="aer_4qubit_Iteration_g1-1_g2-3_noNoise_fixedQC_maxiter50_shots4096_rep100")
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-11_14-07-33",
                extraPlotInfo="g1=1, g2=3, Iteration QC",
                savename="aer_4qubit_Iteration_g1-1_g2-3_yesNoise_fixedQC_maxiter50_shots4096_rep100")

    return
    # used Ising network
    plotBPandCF(filename="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-11_14-01-21",
                extraPlotInfo="g1=1, g2=3, Ising network",
                savename="aer_4qubit_IsingNet_g1-1_g2-3_noNoise_fixedQC_maxiter50_shots4096_rep10")

    return

    # fixed QC
    plotBPandCF(filename="info_testNetwork4Qubit_2_0_20.nc_30_1_2022-02-10_11-30-03",
                extraPlotInfo="g1=1, g2=3, fixed QC",
                savename="aer_4qubit_g1-1_g2-3_noNoise_fixedQC_maxiter50_shots4096_rep10")
    plotBPandCF(filename="info_testNetwork4Qubit_2_0_20.nc_30_1_2022-02-10_11-40-01",
                extraPlotInfo="g1=1, g2=3, fixed QC",
                savename="aer_4qubit_g1-1_g2-3_yesNoise_fixedQC_maxiter50_shots4096_rep10")
    plotBPandCF(filename="info_testNetwork4Qubit_2_0_20.nc_30_1_2022-02-10_11-53-34",
                extraPlotInfo="g1=1, g2=3, fixed QC",
                savename="aer_4qubit_g1-1_g2-3_noNoise_fixedQC_initial-2_maxiter50_shots4096_rep10")
    plotBPandCF(filename="info_testNetwork4Qubit_2_0_20.nc_30_1_2022-02-10_12-02-05",
                extraPlotInfo="g1=1, g2=3, fixed QC",
                savename="aer_4qubit_g1-1_g2-3_noNoise_COBYLA_fixedQC_initial-2_maxiter50_shots4096_rep10")
    plotBPandCF(filename="info_testNetwork4Qubit_2_0_20.nc_30_1_2022-02-10_12-10-36",
                extraPlotInfo="g1=1, g2=3, fixed QC",
                savename="aer_4qubit_g1-1_g2-3_yesNoise_fixedQC_initial-2_maxiter50_shots4096_rep10")
    plotBPandCF(filename="info_testNetwork4Qubit_2_0_20.nc_30_1_2022-02-10_14-01-39",
                extraPlotInfo="g1=1, g2=3, fixed QC",
                savename="aer_4qubit_g1-1_g2-3_yesNoise_COBYLA_fixedQC_initial-2_maxiter50_shots4096_rep10")
    plotBPandCF(filename="info_testNetwork4Qubit_2_0_20.nc_30_1_2022-02-10_14-04-09",
                extraPlotInfo="g1=1, g2=3, fixed QC",
                savename="aer_4qubit_g1-1_g2-3_yesNoise_fixedQC_initial-2_maxiter200_shots4096_rep10")
    plotBPandCF(filename="info_testNetwork4Qubit_2_0_20.nc_30_1_2022-02-10_15-34-44",
                extraPlotInfo="g1=1, g2=3, fixed QC",
                savename="aer_4qubit_g1-1_g2-3_noNoise_fixedQC_initial-2_maxiter200_shots4096_rep10")

    return
    # COBYLA
    # without noise
    plotBPandCF(filename="info_testNetwork4Qubit_2_0_20.nc_30_1_2022-02-09_17-32-05",
                extraPlotInfo="g1=1, g2=3",
                savename="state_4qubit_g1-1_g2-3_noNoise_COBYLA_maxiter50_shots4096_rep10")
    plotBPandCF(filename="info_testNetwork4Qubit_2_0_20.nc_30_1_2022-02-09_17-32-18",
                extraPlotInfo="g1=1, g2=3",
                savename="qasm_4qubit_g1-1_g2-3_noNoise_COBYLA_maxiter50_shots4096_rep10")
    plotBPandCF(filename="info_testNetwork4Qubit_2_0_20.nc_30_1_2022-02-09_17-32-32",
                extraPlotInfo="g1=1, g2=3",
                savename="aer_4qubit_g1-1_g2-3_noNoise_COBYLA_maxiter50_shots4096_rep10")

    return
    # COBYLA
    # without noise
    # separate plots
    plotBoxplot(docker=True, filename="info_testNetwork4Qubit_2_0_20.nc_30_1_2022-02-09_15-31-59",
                plotname="COBYLA without noise - maxiter 50 \n g1=1, g2=3",
                savename="state_4qubit_g1-1_g2-3_noNoise_COBYLA_maxiter50_shots4096_rep10")
    plotBoxplot(docker=True, filename="info_testNetwork4Qubit_2_0_20.nc_30_1_2022-02-09_15-31-49",
                plotname="COBYLA without noise - maxiter 50 \n g1=1, g2=3",
                savename="qasm_4qubit_g1-1_g2-3_noNoise_COBYLA_maxiter50_shots4096_rep10")
    plotBoxplot(docker=True, filename="info_testNetwork4Qubit_2_0_20.nc_30_1_2022-02-09_15-24-45",
                plotname="COBYLA without noise - maxiter 50 \n g1=1, g2=3",
                savename="aer_4qubit_g1-1_g2-3_noNoise_COBYLA_maxiter50_shots4096_rep10")

    return
    # 4bit kirch^2
    # without noise
    plotBoxplot(docker=True, filename="info_testNetwork4Qubit_2_0_20.nc_30_1_2022-02-09_14-48-42",
                plotname="SPSA without noise - maxiter 50 \n g1=1, g2=3, kirch^2",
                savename="state_4qubit_g1-1_g2-3_kirch^2_noNoise_maxiter50_shots4096_rep10")
    plotBoxplot(docker=True, filename="info_testNetwork4Qubit_2_0_20.nc_30_1_2022-02-09_14-48-51",
                plotname="SPSA without noise - maxiter 50 \n g1=1, g2=3, kirch^2",
                savename="qasm_4qubit_g1-1_g2-3_kirch^2_noNoise_maxiter50_shots4096_rep10")
    plotBoxplot(docker=True, filename="info_testNetwork4Qubit_2_0_20.nc_30_1_2022-02-09_14-49-00",
                plotname="SPSA without noise - maxiter 50 \n g1=1, g2=3, kirch^2",
                savename="aer_4qubit_g1-1_g2-3_kirch^2_noNoise_maxiter50_shots4096_rep10")
    plotCFoptimizationDouble(docker=True, filename="info_testNetwork4Qubit_2_0_20.nc_30_1_2022-02-09_14-48-42",
                             plotname="SPSA evolution without noise - maxiter 50 \n g1=1, g2=3, kirch^2, statevector",
                             savename="state_4qubit_g1-1_g2-3_kirch^2_noNoise_maxiter50_shots4096_rep10")
    plotCFoptimizationDouble(docker=True, filename="info_testNetwork4Qubit_2_0_20.nc_30_1_2022-02-09_14-48-51",
                             plotname="SPSA evolution without noise - maxiter 50 \n g1=1, g2=3, kirch^2, qasm",
                             savename="qasm_4qubit_g1-1_g2-3_kirch^2_noNoise_maxiter50_shots4096_rep10")
    plotCFoptimizationDouble(docker=True, filename="info_testNetwork4Qubit_2_0_20.nc_30_1_2022-02-09_14-49-00",
                             plotname="SPSA evolution without noise - maxiter 50 \n g1=1, g2=3, kirch^2, aer",
                             savename="aer_4qubit_g1-1_g2-3_kirch^2_noNoise_maxiter50_shots4096_rep10")

    # 4bit gInverse kirch^2
    # without noise
    plotBoxplot(docker=True, filename="info_testNetwork4Qubit-gInverse_2_0_20.nc_30_1_2022-02-09_14-26-31",
                plotname="SPSA without noise - maxiter 50 \n g1=3, g2=1, kirch^2",
                savename="state_4qubit_g1-3_g2-1_kirch^2_noNoise_maxiter50_shots4096_rep10")
    plotBoxplot(docker=True, filename="info_testNetwork4Qubit-gInverse_2_0_20.nc_30_1_2022-02-09_14-26-22",
                plotname="SPSA without noise - maxiter 50 \n g1=3, g2=1, kirch^2",
                savename="qasm_4qubit_g1-3_g2-1_kirch^2_noNoise_maxiter50_shots4096_rep10")
    plotBoxplot(docker=True, filename="info_testNetwork4Qubit-gInverse_2_0_20.nc_30_1_2022-02-09_14-26-12",
                plotname="SPSA without noise - maxiter 50 \n g1=3, g2=1, kirch^2",
                savename="aer_4qubit_g1-3_g2-1_kirch^2_noNoise_maxiter50_shots4096_rep10")
    plotCFoptimizationDouble(docker=True, filename="info_testNetwork4Qubit-gInverse_2_0_20.nc_30_1_2022-02-09_14-26-31",
                             plotname="SPSA evolution without noise - maxiter 50 \n g1=3, g2=1, kirch^2, statevector",
                             savename="state_4qubit_g1-3_g2-1_kirch^2_noNoise_maxiter50_shots4096_rep10")
    plotCFoptimizationDouble(docker=True, filename="info_testNetwork4Qubit-gInverse_2_0_20.nc_30_1_2022-02-09_14-26-22",
                             plotname="SPSA evolution without noise - maxiter 50 \n g1=3, g2=1, kirch^2, qasm",
                             savename="qasm_4qubit_g1-3_g2-1_kirch^2_noNoise_maxiter50_shots4096_rep10")
    plotCFoptimizationDouble(docker=True, filename="info_testNetwork4Qubit-gInverse_2_0_20.nc_30_1_2022-02-09_14-26-12",
                             plotname="SPSA evolution without noise - maxiter 50 \n g1=3, g2=1, kirch^2, aer",
                             savename="aer_4qubit_g1-3_g2-1_kirch^2_noNoise_maxiter50_shots4096_rep10")

    # 4bit gInverse
    # without noise
    plotBoxplot(docker=True, filename="info_testNetwork4Qubit-gInverse_2_0_20.nc_30_1_2022-02-09_13-35-57",
                plotname="SPSA without noise - maxiter 50 \n g1=3, g2=1",
                savename="state_4qubit_g1-3_g2-1_noNoise_maxiter50_shots4096_rep10")
    plotBoxplot(docker=True, filename="info_testNetwork4Qubit-gInverse_2_0_20.nc_30_1_2022-02-09_13-36-07",
                plotname="SPSA without noise - maxiter 50 \n g1=3, g2=1",
                savename="qasm_4qubit_g1-3_g2-1_noNoise_maxiter50_shots4096_rep10")
    plotBoxplot(docker=True, filename="info_testNetwork4Qubit-gInverse_2_0_20.nc_30_1_2022-02-09_13-36-20",
                plotname="SPSA without noise - maxiter 50 \n g1=3, g2=1",
                savename="aer_4qubit_g1-3_g2-1_noNoise_maxiter50_shots4096_rep10")
    plotCFoptimizationDouble(docker=True, filename="info_testNetwork4Qubit-gInverse_2_0_20.nc_30_1_2022-02-09_13-35-57",
                             plotname="SPSA evolution without noise - maxiter 50 \n g1=3, g2=1, statevector",
                             savename="state_4qubit_g1-3_g2-1_noNoise_maxiter50_shots4096_rep10")
    plotCFoptimizationDouble(docker=True, filename="info_testNetwork4Qubit-gInverse_2_0_20.nc_30_1_2022-02-09_13-36-07",
                             plotname="SPSA evolution without noise - maxiter 50 \n g1=3, g2=1, qasm",
                             savename="qasm_4qubit_g1-3_g2-1_noNoise_maxiter50_shots4096_rep10")
    plotCFoptimizationDouble(docker=True, filename="info_testNetwork4Qubit-gInverse_2_0_20.nc_30_1_2022-02-09_13-36-20",
                             plotname="SPSA evolution without noise - maxiter 50 \n g1=3, g2=1, aer",
                             savename="aer_4qubit_g1-3_g2-1_noNoise_maxiter50_shots4096_rep10")

    # 4bit g3 & g12
    # without noise
    plotBoxplot(docker=True, filename="info_testNetwork4Qubit-g3g12_2_0_20.nc_30_1_2022-02-09_13-20-35",
                plotname="SPSA without noise - maxiter 50 \n g1=3, g2=12",
                savename="state_4qubit_g1-3_g2-12_noNoise_maxiter50_shots4096_rep10")
    plotBoxplot(docker=True, filename="info_testNetwork4Qubit-g3g12_2_0_20.nc_30_1_2022-02-09_13-10-21",
                plotname="SPSA without noise - maxiter 50 \n g1=3, g2=12",
                savename="qasm_4qubit_g1-3_g2-12_noNoise_maxiter50_shots4096_rep10")
    plotBoxplot(docker=True, filename="info_testNetwork4Qubit-g3g12_2_0_20.nc_30_1_2022-02-09_13-10-51",
                plotname="SPSA without noise - maxiter 50 \n g1=3, g2=12",
                savename="aer_4qubit_g1-3_g2-12_noNoise_maxiter50_shots4096_rep10")
    plotCFoptimizationDouble(docker=True, filename="info_testNetwork4Qubit-g3g12_2_0_20.nc_30_1_2022-02-09_13-20-35",
                             plotname="SPSA evolution without noise - maxiter 50 \n g1=3, g2=12, statevector",
                             savename="state_4qubit_g1-3_g2-12_noNoise_maxiter50_shots4096_rep10")
    plotCFoptimizationDouble(docker=True, filename="info_testNetwork4Qubit-g3g12_2_0_20.nc_30_1_2022-02-09_13-10-21",
                             plotname="SPSA evolution without noise - maxiter 50 \n g1=3, g2=12, qasm",
                             savename="qasm_4qubit_g1-3_g2-12_noNoise_maxiter50_shots4096_rep10")
    plotCFoptimizationDouble(docker=True, filename="info_testNetwork4Qubit-g3g12_2_0_20.nc_30_1_2022-02-09_13-10-51",
                             plotname="SPSA evolution without noise - maxiter 50 \n g1=3, g2=12, aer",
                             savename="aer_4qubit_g1-3_g2-12_noNoise_maxiter50_shots4096_rep10")

    # 5bit
    # without noise
    plotBoxplot(docker=True, filename="info_testNetwork5Qubit_2_0_20.nc_30_1_2022-02-09_11-33-18",
                plotname="SPSA without noise - maxiter 50 \n g1=2, g2=4, g3=2",
                savename="state_5qubit_g1-2_g2-4_g3-2_noNoise_maxiter50_shots4096_rep10")
    plotBoxplot(docker=True, filename="info_testNetwork5Qubit_2_0_20.nc_30_1_2022-02-09_12-03-53",
                plotname="SPSA without noise - maxiter 50 \n g1=2, g2=4, g3=2",
                savename="qasm_5qubit_g1-2_g2-4_g3-2_noNoise_maxiter50_shots4096_rep10")
    plotBoxplot(docker=True, filename="info_testNetwork5Qubit_2_0_20.nc_30_1_2022-02-09_12-03-37",
                plotname="SPSA without noise - maxiter 50 \n g1=2, g2=4, g3=2",
                savename="aer_5qubit_g1-2_g2-4_g3-2_noNoise_maxiter50_shots4096_rep10")
    plotCFoptimizationDouble(docker=True, filename="info_testNetwork5Qubit_2_0_20.nc_30_1_2022-02-09_11-33-18",
                             plotname="SPSA evolution without noise - maxiter 50 \n g1=2, g2=4, g3=2, statevector",
                             savename="state_5qubit_g1-2_g2-4_g3-2_noNoise_maxiter50_shots4096_rep10")
    plotCFoptimizationDouble(docker=True, filename="info_testNetwork5Qubit_2_0_20.nc_30_1_2022-02-09_12-03-53",
                             plotname="SPSA evolution without noise - maxiter 50 \n g1=2, g2=4, g3=2, qasm",
                             savename="qasm_5qubit_g1-2_g2-4_g3-2_noNoise_maxiter50_shots4096_rep10")
    plotCFoptimizationDouble(docker=True, filename="info_testNetwork5Qubit_2_0_20.nc_30_1_2022-02-09_12-03-37",
                             plotname="SPSA evolution without noise - maxiter 50 \n g1=2, g2=4, g3=2, aer",
                             savename="aer_5qubit_g1-2_g2-4_g3-2_noNoise_maxiter50_shots4096_rep10")

    # 4bit
    # without noise
    plotBoxplot(docker=True, filename="info_testNetwork4Qubit_2_0_20.nc_30_1_2022-02-09_09-59-35",
                plotname="SPSA without noise - maxiter 50 \n g1=1, g2=3",
                savename="state_4qubit_g1-1_g2-3_noNoise_maxiter50_shots4096_rep10")
    plotBoxplot(docker=True, filename="info_testNetwork4Qubit_2_0_20.nc_30_1_2022-02-09_10-16-10",
                plotname="SPSA without noise - maxiter 50 \n g1=1, g2=3",
                savename="qasm_4qubit_g1-1_g2-3_noNoise_maxiter50_shots4096_rep10")
    plotBoxplot(docker=True, filename="info_testNetwork4Qubit_2_0_20.nc_30_1_2022-02-09_10-16-57",
                plotname="SPSA without noise - maxiter 50 \n g1=1, g2=3",
                savename="aer_4qubit_g1-1_g2-3_noNoise_maxiter50_shots4096_rep10")
    plotCFoptimizationDouble(docker=True, filename="info_testNetwork4Qubit_2_0_20.nc_30_1_2022-02-09_09-59-35",
                             plotname="SPSA evolution without noise - maxiter 50 \n g1=1, g2=3, statevector",
                             savename="state_4qubit_g1-1_g2-3_noNoise_maxiter50_shots4096_rep10")
    plotCFoptimizationDouble(docker=True, filename="info_testNetwork4Qubit_2_0_20.nc_30_1_2022-02-09_10-16-10",
                             plotname="SPSA evolution without noise - maxiter 50 \n g1=1, g2=3, qasm",
                             savename="qasm_4qubit_g1-1_g2-3_noNoise_maxiter50_shots4096_rep10")
    plotCFoptimizationDouble(docker=True, filename="info_testNetwork4Qubit_2_0_20.nc_30_1_2022-02-09_10-16-57",
                             plotname="SPSA evolution without noise - maxiter 50 \n g1=1, g2=3, aer",
                             savename="aer_4qubit_g1-1_g2-3_noNoise_maxiter50_shots4096_rep10")

    # 4bit
    # with noise
    plotBoxplot(docker=True, filename="info_testNetwork4Qubit_2_0_20.nc_30_1_2022-02-09_10-44-43",
                plotname="SPSA with noise - maxiter 50 \n g1=1, g2=3",
                savename="state_4qubit_g1-1_g2-3_yesNoise_maxiter50_shots4096_rep10")
    plotBoxplot(docker=True, filename="info_testNetwork4Qubit_2_0_20.nc_30_1_2022-02-09_10-32-41",
                plotname="SPSA with noise - maxiter 50 \n g1=1, g2=3",
                savename="qasm_4qubit_g1-1_g2-3_yesNoise_maxiter50_shots4096_rep10")
    plotBoxplot(docker=True, filename="info_testNetwork4Qubit_2_0_20.nc_30_1_2022-02-09_10-32-34",
                plotname="SPSA with noise - maxiter 50 \n g1=1, g2=3",
                savename="aer_4qubit_g1-1_g2-3_yesNoise_maxiter50_shots4096_rep10")
    plotCFoptimizationDouble(docker=True, filename="info_testNetwork4Qubit_2_0_20.nc_30_1_2022-02-09_10-44-43",
                             plotname="SPSA evolution with noise - maxiter 50 \n g1=1, g2=3, statevector",
                             savename="state_4qubit_g1-1_g2-3_yesNoise_maxiter50_shots4096_rep10")
    plotCFoptimizationDouble(docker=True, filename="info_testNetwork4Qubit_2_0_20.nc_30_1_2022-02-09_10-32-41",
                             plotname="SPSA evolution with noise - maxiter 50 \n g1=1, g2=3, qasm",
                             savename="qasm_4qubit_g1-1_g2-3_yesNoise_maxiter50_shots4096_rep10")
    plotCFoptimizationDouble(docker=True, filename="info_testNetwork4Qubit_2_0_20.nc_30_1_2022-02-09_10-32-34",
                             plotname="SPSA evolution with noise - maxiter 50 \n g1=1, g2=3, aer",
                             savename="aer_4qubit_g1-1_g2-3_yesNoise_maxiter50_shots4096_rep10")
    return

    #plotBoxplot(filename="QaoaCompare_2022-2-4_15-22-53_606565",
    #            plotname="simulator with noise using SPSA - maxiter 200 \n")
    #plotCFoptimization(filename="QaoaCompare_2022-2-4_15-22-53_606565",
    #                   plotname="SPSA evolution with noise - maxiter 200 \n")
    plotPropVsCost(filename="QaoaCompare_2022-2-4_15-22-53_606565",
                   plotname="probability of costs - maxiter 50 \n ")

    #plotBoxplot(filename="QaoaCompare_2022-2-4_14-36-14_624819",
    #            plotname="simulator with noise using SPSA - maxiter 50 \n optimize")
    #plotCFoptimization(filename="QaoaCompare_2022-2-4_14-36-14_624819",
    #                   plotname="SPSA evolution with noise - maxiter 50 \n optimize")

    #plotBoxplot(filename="QaoaCompare_2022-2-4_11-28-27_274990",
    #            plotname="simulator with noise using SPSA - maxiter 50 \n minimize")
    #plotCFoptimization(filename="QaoaCompare_2022-2-4_11-28-27_274990",
    #                   plotname="SPSA evolution with noise - maxiter 50 \n minimize")
    # plotPropVsCost4Qubit(filename="QaoaCompare_2022-2-3_18-27-55_714984",
    #               plotname="probability of costs - maxiter 50 \n kirchhoff einfach")

    #plotBoxplot(filename="QaoaCompare_2022-2-3_16-33-32_116043",
    #            plotname="simulator with noise using SPSA - maxiter 50 \n kirchhoff einfach, complete Hb after each Hp")
    #plotCFoptimization(filename="QaoaCompare_2022-2-3_16-33-32_116043",
    #                   plotname="SPSA evolution with noise - maxiter 50 \n kirchhoff einfach, complete Hb after each Hp")

    #plotBoxplot(filename="QaoaCompare_2022-2-3_15-29-4_469482",
    #            plotname="simulator with noise using SPSA - maxiter 50 \n kirchhoff quadriert")
    #plotCFoptimization(filename="QaoaCompare_2022-2-3_15-29-4_469482",
    #                   plotname="SPSA evolution with noise - maxiter 50 \n kirchhoff quadriert")

    #plotBoxplot(filename="QaoaCompare_2022-2-3_15-40-21_628234",
    #            plotname="simulator with noise using SPSA - maxiter 50 \n kirchhoff +5 & quadriert")
    #plotCFoptimization(filename="QaoaCompare_2022-2-3_15-40-21_628234",
    #                   plotname="SPSA evolution with noise - maxiter 50 \n kirchhoff +5 & quadriert")



    return
    plotBoxplot(filename="QaoaCompare_2022-2-3_10-25-6_709496",
                plotname="simulator with noise using SPSA - maxiter 50 - kirchhoff quadriert, 2 Hp")

    return

    plotBoxplot(filename="QaoaCompare_2022-2-3_10-9-50_335597",
                plotname="simulator with noise using SPSA - maxiter 50 - kirchhoff quadriert")

    return

    plotBoxplot(filename="QaoaCompare_2022-2-3_9-38-29_848235",
                plotname="simulator with noise using SPSA - maxiter 50")
    #plotCFoptimization(filename="QaoaCompare_2022-2-2_19-43-50_611326",
    #                   plotname="SPSA evolution with noise - maxiter 50")

    return

    plotBoxplot(filename="QaoaCompare_2022-2-2_9-57-52_224944",
                plotname="simulator with noise using SPSA - maxiter 100")
    plotCFoptimizationDouble(filename="QaoaCompare_2022-2-2_9-57-52_224944",
                             plotname="SPSA evolution with noise - maxiter 100")

    return

    plotBoxplot(filename="QaoaCompare_2022-2-1_17-39-28_26095",
                plotname="simulator no noise using SPSA - maxiter 25")
    plotBoxplot(filename="QaoaCompare_2022-2-1_18-16-50_858006",
                plotname="simulator with noise using SPSA - maxiter 25")
    plotCFoptimizationDouble(filename="QaoaCompare_2022-2-1_18-16-50_858006",
                             plotname="SPSA evolution with noise - maxiter 25")
    plotCFoptimizationDouble(filename="QaoaCompare_2022-2-1_17-39-28_26095",
                             plotname="SPSA evolution without noise - maxiter 25")

    return

    plotBoxplot(filename="QaoaCompare_2022-2-1_17-14-20_973918",
                plotname="simulator no noise using SPSA - maxiter 10")
    plotBoxplot(filename="QaoaCompare_2022-2-1_17-3-6_698242",
                plotname="simulator with noise using SPSA - maxiter 10")
    plotCFoptimizationDouble(filename="QaoaCompare_2022-2-1_17-3-6_698242",
                             plotname="SPSA evolution with noise - maxiter 10")
    plotCFoptimizationDouble(filename="QaoaCompare_2022-2-1_17-14-20_973918",
                             plotname="SPSA evolution without noise - maxiter 10")


    plotCFoptimizationDouble(filename="QaoaCompare_2022-2-1_15-22-52_137642",
                             plotname="SPSA evolution with noise - maxiter 50")
    plotCFoptimizationDouble(filename="QaoaCompare_2022-2-1_15-52-0_342208",
                             plotname="SPSA evolution without noise - maxiter 50")
    plotBoxplot(filename="QaoaCompare_2022-2-1_15-52-0_342208",
                plotname="simulator no noise using SPSA - maxiter 50")
    plotBoxplot(filename="QaoaCompare_2022-2-1_15-22-52_137642",
                plotname="simulator with noise using SPSA - maxiter 50")

    return
    plotCFoptimizationDouble(filename="QaoaCompare_2022-1-31_13-11-6_987313", plotname="optimization evolution with noise")
    plotCFoptimizationDouble(filename="QaoaCompare_2022-1-31_12-35-19_479895", plotname="optimization evolution without noise")

    plotBoxplot(filename="QaoaCompare_2022-1-31_13-11-6_987313", plotname="aer_simulator with noise")


if __name__ == '__main__':
    main()
