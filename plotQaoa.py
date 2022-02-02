import matplotlib.pyplot as plt
import numpy as np
import json
import random

FILENAME = "QaoaCompare_2022-1-31_13-11-6_987313"

def openFile(filename: str, directory: str) -> dict:
    with open(f"{directory}{filename}.json") as json_file:
        data = json.load(json_file)

    return data


def plotBoxplot(filename: str, plotname: str):
    data = openFile(filename=filename, directory="results_qaoa/qaoaCompare/")

    bitstrings = list(data["1"]["counts"].keys())
    bitstrings.sort()
    shots = data["1"]["shots"]
    subfile = data["1"]["filename"][:-5]
    subdata = openFile(filename=subfile, directory="results_qaoa/")
    backend = subdata[f"rep{subdata['iter_count']}"]["backend"]["backend_name"]
    initial_guess = data["1"]["initial_guess"]
    toPlot = [[] for i in range(len(bitstrings))]

    for key in data:
        for bitstring in bitstrings:
            bitstring_index = bitstrings.index(bitstring)
            if bitstring in data[key]["counts"]:
                appendData = data[key]["counts"][bitstring]
            else:
                appendData = 0
            toPlot[bitstring_index].append(appendData/shots)

    fig, ax = plt.subplots()
    fig.set_figheight(7)
    pos = np.arange(len(toPlot)) + 1
    bp = ax.boxplot(toPlot, sym='k+', positions=pos, bootstrap=5000)

    ax.set_xlabel('bitstrings')
    ax.set_ylabel('probability')
    plt.title(f"backend = {backend}, shots = {shots}, rep = {len(data)} \n initial guess = {initial_guess}", fontdict = {'fontsize' : 8})
    plt.figtext(0.5, 0.01, f"data: {filename}", fontdict={'fontsize': 8})
    plt.suptitle(plotname)
    plt.xticks(range(1, len(bitstrings)+1), bitstrings, rotation=70)
    plt.setp(bp['whiskers'], color='k', linestyle='-')
    plt.setp(bp['fliers'], markersize=2.0)
    #plt.show()
    plt.savefig(f"plots/{filename}_boxplot_{plotname}.png")


def plotCFoptimization(filename: str, plotname:str):
    data = openFile(filename=filename, directory="results_qaoa/qaoaCompare/")
    fig, axs = plt.subplots(2, sharex=True, sharey=True)
    fig.set_figheight(7)

    random_list = list(range(1, len(data)+1))
    random.shuffle(random_list)

    for i in range(len(axs)):
        rep = random_list[i]
        subfile = data[str(rep)]["filename"][:-5]
        subdata = openFile(filename=subfile, directory="results_qaoa/")

        yData = [[], [], []]
        xData = []
        for j in range(subdata["iter_count"]):
            yData[0].append(subdata[f"rep{j + 1}"]["beta"])
            yData[1].append(subdata[f"rep{j + 1}"]["gamma"])
            yData[2].append(subdata[f"rep{j + 1}"]["return"])
            xData.append(j + 1)

        axs[i].plot(xData, yData[0], "b-", label="beta")
        axs[i].plot(xData, yData[1], "r-", label="gamma")
        axs[i].plot(xData, yData[2], "g-", label="cost function")
        axs[i].set_xlabel('iteration')
        axs[i].set_ylabel('value')
        axs[i].label_outer()
        axs[i].text(0.53, 0.02, f"data: {subfile}", transform=axs[i].transAxes, fontdict={'fontsize': 8})
        axs[i].set_title(f"rep {rep}", fontdict={'fontsize': 8})

    fig.suptitle(plotname)
    fig.legend(["beta", "gamma", "cost function"], loc="upper right")
    plt.figtext(0.55, 0.01, f"data: {filename}", fontdict={'fontsize': 8})
    #plt.show()
    plt.savefig(f"plots/{filename}_CF_optimization_{plotname}.png")



def main():
    plotBoxplot(filename="QaoaCompare_2022-2-2_18-49-45_869532",
                plotname="simulator with noise using SPSA - maxiter 50")
    #plotCFoptimization(filename="QaoaCompare_2022-2-2_17-40-14_868831",
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