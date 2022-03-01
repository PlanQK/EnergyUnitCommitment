import itertools

import matplotlib.pyplot as plt
import numpy
from matplotlib import rcParams
import numpy as np
import json
import random

from numpy import median, linalg

from statistics import mean
from scipy.optimize import curve_fit


def openFile(filename: str, directory: str) -> dict:
    """
    Opens the given json file and returns its content as a dictionary.
    Args:
        filename: (str) The name of the file to be opened.
        directory: (str) The folder in which the file is located.

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
    bitstrings = getBitstrings(nBits=4)

    plotData = {"cf": [],
                "beta": [],
                "gamma": [],
                "duration": []}
    plotDataFull = {"cf": [],
                    "beta": [],
                    "gamma": []}
    labels = {"cf": "cost function",
                "beta": "beta",
                "gamma": "gamma",
                "duration": "time"}
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
        plotData["beta"].append(tempData["optimizeResults"]["x"][0])
        plotData["gamma"].append(tempData["optimizeResults"]["x"][1])
        plotData["duration"].append(tempData["duration"])
        for bitstring in bitstrings:
            if bitstring in data[key]["counts"]:
                plotData[f"{bitstring}shots"].append(data[key]["counts"][bitstring])
                plotData[f"{bitstring}prop"].append(data[key]["counts"][bitstring] / metaData["shots"])
            else:
                plotData[f"{bitstring}shots"].append(0)
                plotData[f"{bitstring}prop"].append(0)

        for i in range(1, tempData["iter_count"] + 1):
            plotDataFull["cf"].append(tempData[f"rep{i}"]["return"])
            plotDataFull["beta"].append(tempData[f"rep{i}"]["theta"][0])
            plotDataFull["gamma"].append(tempData[f"rep{i}"]["theta"][1])
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


def plotBoxplotBest(docker: bool, filename: str, plotname: str, savename: str):
    if docker:
        dataAll = openFile(filename=filename, directory="results_qaoa_sweep/")
        data = dataAll["results"]
    else:
        data = openFile(filename=filename, directory="results_qaoa/qaoaCompare/")

    cutoff = {}
    for key in data:
        if docker:
            subdata = openFile(filename=data[key]["filename"], directory="results_qaoa_sweep/")
        else:
            subdata = openFile(filename=data[key]["filename"], directory="results_qaoa/")
        cutoff[key] = subdata["optimizeResults"]["fun"]

    cutoff = dict(sorted(cutoff.items(), key=lambda item: item[1]))

    bitstrings = list(data["1"]["counts"].keys())
    bitstrings.sort()
    # bitstrings = ["0000", "1000", "0100", "1100", "0010", "1010", "0110", "1110",
    #              "0001", "1001", "0101", "1101", "0011", "1011", "0111", "1111"]
    shots = dataAll["config"]["QaoaBackend"]["shots"]
    backend = dataAll["qaoaBackend"]["backend_name"]
    initial_guess = dataAll["config"]["QaoaBackend"]["initial_guess"]
    toPlot = [[] for i in range(len(bitstrings))]

    for i in range(int(len(cutoff)*0.5)):#only plot best 50%
        key = list(cutoff.keys())[i]
        for bitstring in bitstrings:
            bitstring_index = bitstrings.index(bitstring)
            if bitstring in data[key]["counts"]:
                appendData = data[key]["counts"][bitstring]
            else:
                appendData = 0
            toPlot[bitstring_index].append(appendData / shots)

    fig, ax = plt.subplots()
    fig.set_figheight(7)
    pos = np.arange(len(toPlot)) + 1
    bp = ax.boxplot(toPlot, sym='k+', positions=pos, bootstrap=5000)

    ax.set_xlabel('bitstrings')
    ax.set_ylabel('probability')
    plt.title(f"backend = {backend}, shots = {shots}, rep = {len(data)} \n initial guess = {initial_guess}",
              fontdict={'fontsize': 8})
    plt.figtext(0.0, 0.01, f"data: {filename}", fontdict={'fontsize': 8})
    plt.suptitle(plotname)
    plt.xticks(range(1, len(bitstrings) + 1), bitstrings, rotation=70)
    plt.setp(bp['whiskers'], color='k', linestyle='-')
    plt.setp(bp['fliers'], markersize=2.0)
    # plt.show()
    plt.savefig(f"plots/BPB_{savename}.png")


def plotCFoptimization(docker: bool, filename: str, plotname:str, savename: str):
    if docker:
        data = openFile(filename=filename, directory="results_qaoa_sweep/")
        data = data["results"]
    else:
        data = openFile(filename=filename, directory="results_qaoa/qaoaCompare/")
    fig, axs = plt.subplots(2, sharex=True, sharey=True)
    fig.set_figheight(7)

    random_list = list(range(1, len(data)+1))
    random.shuffle(random_list)

    for i in range(len(axs)):
        rep = random_list[i]
        subfile = data[str(rep)]["filename"]
        if docker:
            subdata = openFile(filename=subfile, directory="results_qaoa_sweep/")
        else:
            subdata = openFile(filename=subfile, directory="results_qaoa/")

        l_theta = len(subdata["rep1"]["theta"])

        yData = [[] for j in range(l_theta+1)]
        xData = []
        leg = []
        for j in range(subdata["iter_count"]):
            if (l_theta % 2) != 0:
                yData[0].append(subdata[f"rep{j + 1}"]["theta"][0])
                for k in range(1, l_theta):
                    yData[k].append(subdata[f"rep{j + 1}"]["theta"][k])
            elif (l_theta % 2) == 0:
                for k in range(int(l_theta/2)):
                    yData[2*k].append(subdata[f"rep{j + 1}"]["theta"][2*k])
                    yData[2*k+1].append(subdata[f"rep{j + 1}"]["theta"][2*k+1])
            yData[l_theta].append(subdata[f"rep{j + 1}"]["return"])
            xData.append(j + 1)

        if (l_theta % 2) != 0:
            axs[i].plot(xData, yData[0], "b-", label="beta")
            leg.append("beta")
            for k in range(1, l_theta):
                axs[i].plot(xData, yData[k], color=((k / l_theta), 0, 0, 1),
                            label=f"gamma{k}")
                leg.append(f"gamma{k}")
        elif (l_theta % 2) == 0:
            for k in range(int(l_theta / 2)):
                axs[i].plot(xData, yData[2*k],  color=(0, 0, (1 - (k / l_theta)), 1),
                            label=f"beta{k}")
                leg.append(f"beta{k}")
                axs[i].plot(xData, yData[2*k+1], color=((1 - (k / l_theta)), 0, 0, 1),
                            label=f"gamma{k}")
                leg.append(f"gamma{k}")
        axs[i].plot(xData, yData[l_theta], "g-", label="cost function")
        leg.append("cost function")
        axs[i].set_xlabel('iteration')
        axs[i].set_ylabel('value')
        axs[i].label_outer()
        axs[i].text(0.53, 0.02, f"data: {subfile}", transform=axs[i].transAxes, fontdict={'fontsize': 8})
        axs[i].set_title(f"rep {rep}", fontdict={'fontsize': 8})

    fig.suptitle(plotname)
    fig.legend(leg, loc="upper right")
    plt.figtext(0.0, 0.01, f"data: {filename}", fontdict={'fontsize': 8})
    #plt.show()
    plt.savefig(f"plots/CF_{savename}.png")


def plotBPandCF(filename: str, extraPlotInfo:str, savename: str):
    dataAll = openFile(filename=filename, directory="results_qaoa_sweep/")

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

    plotBoxplot(docker=True, filename=filename, plotname=plotnameBP, savename=savename)
    plotBoxplotBest(docker=True, filename=filename, plotname=plotnameBPB, savename=savename)
    if len(dataAll["results"]) > 1:
        plotCFoptimization(docker=True, filename=filename, plotname=plotnameCF, savename=savename)


def getCFvalue(filename: str, directory: str) -> float:
    subdata = openFile(filename=filename, directory=directory)
    cfValue = subdata["optimizeResults"]["fun"]

    return cfValue


def buildLabels(bitstrings: list, added: list) -> list:
    labels = []
    for i in range(len(bitstrings)):
        labels.append(f"{bitstrings[i]}\n{added[i]}")

    return labels


def plotBitstringBoxCompare(filenames: list, labels: list, colors: list, savename: str,
                            title: str = "Comparision Boxplot", cut: float = 0.5, kirchLabels: int = None):
    directory = "results_qaoa_sweep/"
    cutoff = {}
    toPlot = {}
    if kirchLabels:
        dataAll = openFile(filename=filenames[kirchLabels], directory=directory)
        nBits = len(list(dataAll["results"]["1"]["counts"].keys())[0])
        bitstrings = getBitstrings(nBits=nBits)
        labelAdd = []
        for bitstring in bitstrings:
            kirchValue = dataAll["kirchhoff"][bitstring]["total"]
            labelAdd.append(f"k={kirchValue}")
        xlabels = buildLabels(bitstrings=bitstrings, added=labelAdd)
    elif kirchLabels == 0:
        dataAll = openFile(filename=filenames[kirchLabels], directory=directory)
        nBits = len(list(dataAll["results"]["1"]["counts"].keys())[0])
        bitstrings = getBitstrings(nBits=nBits)
        labelAdd = []
        for bitstring in bitstrings:
            kirchValue = dataAll["kirchhoff"][bitstring]["total"]
            labelAdd.append(f"k={kirchValue}")
        xlabels = buildLabels(bitstrings=bitstrings, added=labelAdd)
    else:
        dataAll = openFile(filename=filenames[0], directory=directory)
        nBits = len(list(dataAll["results"]["1"]["counts"].keys())[0])
        bitstrings = getBitstrings(nBits=nBits)
        xlabels = bitstrings

    for i in range(len(filenames)):
        dataAll = openFile(filename=filenames[i], directory=directory)
        data = dataAll["results"]

        shots = dataAll["config"]["QaoaBackend"]["shots"]

        cutoff[i] = {}

        for key in data:
            cutoff[i][key] = getCFvalue(filename=data[key]["filename"], directory=directory)

        cutoff[i] = dict(sorted(cutoff[i].items(), key=lambda item: item[1]))

        toPlot[i] = [[] for j in range(len(bitstrings))]
        for bitstring in bitstrings:
            bitstring_index = bitstrings.index(bitstring)
            for j in range(int(len(cutoff[i]) * cut)):  # only plot best cut
                key = list(cutoff[i].keys())[j]
                if bitstring in data[key]["counts"]:
                    appendData = data[key]["counts"][bitstring]
                else:
                    appendData = 0
                toPlot[i][bitstring_index].append(appendData / shots)

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)
        plt.setp(bp['fliers'], color=color, markersize=2.0)

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
        bp = plt.boxplot(toPlot[i], positions=np.array(range(len(toPlot[i]))) * nPlots + boxDistance[i], sym='', widths=boxWidth)
        set_box_color(bp, colors[i])
        plt.plot([], c=colors[i], label=labels[i])

    plt.legend()

    plt.title(title)
    plt.xticks(range(0, len(bitstrings) * nPlots, nPlots), xlabels)
    plt.xlim(-1, len(bitstrings) * nPlots + 0.5)
    plt.tight_layout()
    plt.xlabel('bitstrings')
    plt.ylabel('probability')
    fig.set_figheight(7)
    fig.set_figwidth(15)
    #plt.show()
    plt.savefig(f"plots/BP_{savename}_cut{cut}.png")


def plotScatter(file1: str, file2: str, title: str, g1title: str, g2title: str, savename: str, mode: str, x: str, y: str):
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
        Features to choose from for the x- and y-axis:
            - "cf" (cost function);
            - "beta";
            - "gamma";
            - "{bitstring}prop" (probability of the chosen bistring);
            - "{bitstring}shots" (number of shots of the chosen bitstring);
            - "duration" (only available if mode is set to "opt").

    Returns:
        Saves the generated plot, with the name "Scatter_{plotname}.png" to the subfolder 'plots'
    """
    bitstrings, plotData1, plotDataFull1, labels, metaData1 = extractPlotData(filename=file1)
    bitstrings, plotData2, plotDataFull2, labels, metaData2 = extractPlotData(filename=file2)

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

def main():
    blueDark = "#003C50"
    blueMedium = "#005C7B"
    blueLight = "#008DBB"
    orangeDark = "#B45E00"
    orangeMedium = "#F07D00"
    orangeLight = "#FFB15D"

    plotScatter(file1="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_65.yaml",
                file2="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_71.yaml",
                title="cost function vs ideal solution probability of COBYLA and SPSA100 with noise",
                g1title="COBYLA", g2title="SPSA100",
                savename="COBYLA-SPSA100_yesNoise_CFvsTIME",
                mode="opt", x="cf", y="duration")

    filenames = ["infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-25_10-50-04_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-03-01_07-19-44_config.yaml"]
    labels = ["unscaled", "scaled"]
    title = "Network 0 evaluation"
    colors = [blueMedium, orangeMedium]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_scaled-unscaled_testNetwork4QubitIsing_2_0_20", title=title, cut=1.0,
                            kirchLabels=0)

    filenames = ["infoNocost_testNetwork4QubitIsing_2_1_20.nc_30_1_2022-02-25_10-14-41_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_1_20.nc_30_1_2022-03-01_07-19-44_config.yaml"]
    labels = ["unscaled", "scaled"]
    title = "Network 1 evaluation"
    colors = [blueMedium, orangeMedium]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_scaled-unscaled_testNetwork4QubitIsing_2_1_20", title=title, cut=1.0,
                            kirchLabels=0)

    filenames = ["infoNocost_testNetwork4QubitIsing_2_2_20.nc_30_1_2022-02-25_10-14-41_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_2_20.nc_30_1_2022-03-01_07-19-44_config.yaml"]
    labels = ["unscaled", "scaled"]
    title = "Network 2 evaluation"
    colors = [blueMedium, orangeMedium]
    plotBitstringBoxCompare(filenames=filenames, labels=labels, colors=colors,
                            savename="4qubit_scaled-unscaled_testNetwork4QubitIsing_2_2_20", title=title, cut=1.0,
                            kirchLabels=0)

    filenames = ["infoNocost_testNetwork4QubitIsing_2_3_20.nc_30_1_2022-02-25_11-06-22_config.yaml",
                 "infoNocost_testNetwork4QubitIsing_2_3_20.nc_30_1_2022-03-01_07-19-44_config.yaml"]
    labels = ["unscaled", "scaled"]
    title = "Network 3 evaluation"
    colors = [blueMedium, orangeMedium]
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
    plotCFoptimization(docker=True, filename="info_testNetwork4Qubit_2_0_20.nc_30_1_2022-02-09_14-48-42",
                       plotname="SPSA evolution without noise - maxiter 50 \n g1=1, g2=3, kirch^2, statevector",
                       savename="state_4qubit_g1-1_g2-3_kirch^2_noNoise_maxiter50_shots4096_rep10")
    plotCFoptimization(docker=True, filename="info_testNetwork4Qubit_2_0_20.nc_30_1_2022-02-09_14-48-51",
                       plotname="SPSA evolution without noise - maxiter 50 \n g1=1, g2=3, kirch^2, qasm",
                       savename="qasm_4qubit_g1-1_g2-3_kirch^2_noNoise_maxiter50_shots4096_rep10")
    plotCFoptimization(docker=True, filename="info_testNetwork4Qubit_2_0_20.nc_30_1_2022-02-09_14-49-00",
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
    plotCFoptimization(docker=True, filename="info_testNetwork4Qubit-gInverse_2_0_20.nc_30_1_2022-02-09_14-26-31",
                       plotname="SPSA evolution without noise - maxiter 50 \n g1=3, g2=1, kirch^2, statevector",
                       savename="state_4qubit_g1-3_g2-1_kirch^2_noNoise_maxiter50_shots4096_rep10")
    plotCFoptimization(docker=True, filename="info_testNetwork4Qubit-gInverse_2_0_20.nc_30_1_2022-02-09_14-26-22",
                       plotname="SPSA evolution without noise - maxiter 50 \n g1=3, g2=1, kirch^2, qasm",
                       savename="qasm_4qubit_g1-3_g2-1_kirch^2_noNoise_maxiter50_shots4096_rep10")
    plotCFoptimization(docker=True, filename="info_testNetwork4Qubit-gInverse_2_0_20.nc_30_1_2022-02-09_14-26-12",
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
    plotCFoptimization(docker=True, filename="info_testNetwork4Qubit-gInverse_2_0_20.nc_30_1_2022-02-09_13-35-57",
                       plotname="SPSA evolution without noise - maxiter 50 \n g1=3, g2=1, statevector",
                       savename="state_4qubit_g1-3_g2-1_noNoise_maxiter50_shots4096_rep10")
    plotCFoptimization(docker=True, filename="info_testNetwork4Qubit-gInverse_2_0_20.nc_30_1_2022-02-09_13-36-07",
                       plotname="SPSA evolution without noise - maxiter 50 \n g1=3, g2=1, qasm",
                       savename="qasm_4qubit_g1-3_g2-1_noNoise_maxiter50_shots4096_rep10")
    plotCFoptimization(docker=True, filename="info_testNetwork4Qubit-gInverse_2_0_20.nc_30_1_2022-02-09_13-36-20",
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
    plotCFoptimization(docker=True, filename="info_testNetwork4Qubit-g3g12_2_0_20.nc_30_1_2022-02-09_13-20-35",
                       plotname="SPSA evolution without noise - maxiter 50 \n g1=3, g2=12, statevector",
                       savename="state_4qubit_g1-3_g2-12_noNoise_maxiter50_shots4096_rep10")
    plotCFoptimization(docker=True, filename="info_testNetwork4Qubit-g3g12_2_0_20.nc_30_1_2022-02-09_13-10-21",
                       plotname="SPSA evolution without noise - maxiter 50 \n g1=3, g2=12, qasm",
                       savename="qasm_4qubit_g1-3_g2-12_noNoise_maxiter50_shots4096_rep10")
    plotCFoptimization(docker=True, filename="info_testNetwork4Qubit-g3g12_2_0_20.nc_30_1_2022-02-09_13-10-51",
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
    plotCFoptimization(docker=True, filename="info_testNetwork5Qubit_2_0_20.nc_30_1_2022-02-09_11-33-18",
                       plotname="SPSA evolution without noise - maxiter 50 \n g1=2, g2=4, g3=2, statevector",
                       savename="state_5qubit_g1-2_g2-4_g3-2_noNoise_maxiter50_shots4096_rep10")
    plotCFoptimization(docker=True, filename="info_testNetwork5Qubit_2_0_20.nc_30_1_2022-02-09_12-03-53",
                       plotname="SPSA evolution without noise - maxiter 50 \n g1=2, g2=4, g3=2, qasm",
                       savename="qasm_5qubit_g1-2_g2-4_g3-2_noNoise_maxiter50_shots4096_rep10")
    plotCFoptimization(docker=True, filename="info_testNetwork5Qubit_2_0_20.nc_30_1_2022-02-09_12-03-37",
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
    plotCFoptimization(docker=True, filename="info_testNetwork4Qubit_2_0_20.nc_30_1_2022-02-09_09-59-35",
                       plotname="SPSA evolution without noise - maxiter 50 \n g1=1, g2=3, statevector",
                       savename="state_4qubit_g1-1_g2-3_noNoise_maxiter50_shots4096_rep10")
    plotCFoptimization(docker=True, filename="info_testNetwork4Qubit_2_0_20.nc_30_1_2022-02-09_10-16-10",
                       plotname="SPSA evolution without noise - maxiter 50 \n g1=1, g2=3, qasm",
                       savename="qasm_4qubit_g1-1_g2-3_noNoise_maxiter50_shots4096_rep10")
    plotCFoptimization(docker=True, filename="info_testNetwork4Qubit_2_0_20.nc_30_1_2022-02-09_10-16-57",
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
    plotCFoptimization(docker=True, filename="info_testNetwork4Qubit_2_0_20.nc_30_1_2022-02-09_10-44-43",
                       plotname="SPSA evolution with noise - maxiter 50 \n g1=1, g2=3, statevector",
                       savename="state_4qubit_g1-1_g2-3_yesNoise_maxiter50_shots4096_rep10")
    plotCFoptimization(docker=True, filename="info_testNetwork4Qubit_2_0_20.nc_30_1_2022-02-09_10-32-41",
                       plotname="SPSA evolution with noise - maxiter 50 \n g1=1, g2=3, qasm",
                       savename="qasm_4qubit_g1-1_g2-3_yesNoise_maxiter50_shots4096_rep10")
    plotCFoptimization(docker=True, filename="info_testNetwork4Qubit_2_0_20.nc_30_1_2022-02-09_10-32-34",
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
    plotCFoptimization(filename="QaoaCompare_2022-2-2_9-57-52_224944",
                       plotname="SPSA evolution with noise - maxiter 100")

    return

    plotBoxplot(filename="QaoaCompare_2022-2-1_17-39-28_26095",
                plotname="simulator no noise using SPSA - maxiter 25")
    plotBoxplot(filename="QaoaCompare_2022-2-1_18-16-50_858006",
                plotname="simulator with noise using SPSA - maxiter 25")
    plotCFoptimization(filename="QaoaCompare_2022-2-1_18-16-50_858006",
                       plotname="SPSA evolution with noise - maxiter 25")
    plotCFoptimization(filename="QaoaCompare_2022-2-1_17-39-28_26095",
                       plotname="SPSA evolution without noise - maxiter 25")

    return

    plotBoxplot(filename="QaoaCompare_2022-2-1_17-14-20_973918",
                plotname="simulator no noise using SPSA - maxiter 10")
    plotBoxplot(filename="QaoaCompare_2022-2-1_17-3-6_698242",
                plotname="simulator with noise using SPSA - maxiter 10")
    plotCFoptimization(filename="QaoaCompare_2022-2-1_17-3-6_698242",
                       plotname="SPSA evolution with noise - maxiter 10")
    plotCFoptimization(filename="QaoaCompare_2022-2-1_17-14-20_973918",
                       plotname="SPSA evolution without noise - maxiter 10")


    plotCFoptimization(filename="QaoaCompare_2022-2-1_15-22-52_137642",
                       plotname="SPSA evolution with noise - maxiter 50")
    plotCFoptimization(filename="QaoaCompare_2022-2-1_15-52-0_342208",
                       plotname="SPSA evolution without noise - maxiter 50")
    plotBoxplot(filename="QaoaCompare_2022-2-1_15-52-0_342208",
                plotname="simulator no noise using SPSA - maxiter 50")
    plotBoxplot(filename="QaoaCompare_2022-2-1_15-22-52_137642",
                plotname="simulator with noise using SPSA - maxiter 50")

    return
    plotCFoptimization(filename="QaoaCompare_2022-1-31_13-11-6_987313", plotname="optimization evolution with noise")
    plotCFoptimization(filename="QaoaCompare_2022-1-31_12-35-19_479895", plotname="optimization evolution without noise")

    plotBoxplot(filename="QaoaCompare_2022-1-31_13-11-6_987313", plotname="aer_simulator with noise")


if __name__ == '__main__':
    main()