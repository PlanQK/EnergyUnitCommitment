import matplotlib.pyplot as plt
import numpy as np
import json
import random
from statistics import mean

FILENAME = "QaoaCompare_2022-1-31_13-11-6_987313"

def openFile(filename: str, directory: str) -> dict:
    with open(f"{directory}{filename}") as json_file:
        data = json.load(json_file)

    return data


def plotPropVsCost(docker: bool, filename: str, plotname: str):
    if docker:
        data = openFile(filename=filename, directory="results_qaoa_sweep/")
        data = data["results"]
    else:
        data = openFile(filename=filename, directory="results_qaoa/qaoaCompare/")

    bitstrings = list(data["1"]["counts"].keys())
    bitstrings.sort()
    shots = data["1"]["shots"]
    subfile = data["1"]["filename"]
    if docker:
        subdata = openFile(filename=subfile, directory="results_qaoa_sweep/")
    else:
        subdata = openFile(filename=subfile, directory="results_qaoa/")
    backend = subdata[f"rep{subdata['iter_count']}"]["backend"]["backend_name"]
    initial_guess = data["1"]["initial_guess"]
    kirchhoffFileName = "kirchhoff" + data["1"]["filename"][4:]
    kirchhoffData = openFile(filename=kirchhoffFileName, directory="results_qaoa/")
    bitstring_costs = kirchhoffData["rep1"]
    toPlot = [[] for i in range(100)]
    #bitstring_costs = {"0000": 3, "0001": 4, "0010": 5, "0011": 8, "0100": 3, "0101": 0, "0110": 3, "0111": 4,
    #                   "1000": 2, "1001": 3, "1010": 4, "1011": 7, "1100": 4, "1101": 1, "1110": 2, "1111": 3}

    for key in data:
        for bitstring in bitstrings:
            bitstring_index = int(bitstring_costs[bitstring]["total"])
            if bitstring in data[key]["counts"]:
                appendData = data[key]["counts"][bitstring]
            else:
                appendData = 0
            toPlot[bitstring_index].append(appendData / shots)
    yData = []
    xData = []
    for i in range(len(toPlot)):
        if toPlot[i]:  # if toPlot[i] is an empty list it returns False
            yData.append(mean(toPlot[i]))
            #yData.append(sum(toPlot[i]) / len(data))
            xData.append(i)

    fig, ax = plt.subplots()
    fig.set_figheight(7)
    ax.plot(xData, yData, "x-")

    ax.set_xlabel('cost function')
    ax.set_ylabel('probability')
    plt.title(f"backend = {backend}, shots = {shots}, rep = {len(data)} \n initial guess = {initial_guess}",
              fontdict={'fontsize': 8})
    plt.figtext(0.0, 0.01, f"data: {filename}", fontdict={'fontsize': 8})
    plt.suptitle(plotname)

    #plt.show()
    plt.savefig(f"plots/{filename}_PropVsCF1.png")


def plotPropVsCost4qubit(docker: bool, filename: str, plotname: str):
    if docker:
        data = openFile(filename=filename, directory="results_qaoa_sweep/")
        data = data["results"]
    else:
        data = openFile(filename=filename, directory="results_qaoa/qaoaCompare/")

    bitstrings = list(data["1"]["counts"].keys())
    bitstrings.sort()
    shots = data["1"]["shots"]
    subfile = data["1"]["filename"]
    if docker:
        subdata = openFile(filename=subfile, directory="results_qaoa_sweep/")
    else:
        subdata = openFile(filename=subfile, directory="results_qaoa/")
    backend = subdata[f"rep{subdata['iter_count']}"]["backend"]["backend_name"]
    initial_guess = data["1"]["initial_guess"]
    toPlot = [[] for i in range(9)]
    bitstring_costs = {"0000": 3, "0001": 4, "0010": 5, "0011": 8, "0100": 3, "0101": 0, "0110": 3, "0111": 4,
                       "1000": 2, "1001": 3, "1010": 4, "1011": 7, "1100": 4, "1101": 1, "1110": 2, "1111": 3}

    for key in data:
        for bitstring in bitstrings:
            bitstring_index = bitstring_costs[bitstring]
            if bitstring in data[key]["counts"]:
                appendData = data[key]["counts"][bitstring]
            else:
                appendData = 0
            toPlot[bitstring_index].append(appendData / shots)
    yData = []
    xData = []
    for i in range(len(toPlot)):
        if toPlot[i]:
            #yData.append(mean(toPlot[i]))
            yData.append(sum(toPlot[i]) / len(data))
            xData.append(i)

    fig, ax = plt.subplots()
    fig.set_figheight(7)
    ax.plot(xData, yData, "x-")

    ax.set_xlabel('cost function')
    ax.set_ylabel('probability')
    plt.title(f"backend = {backend}, shots = {shots}, rep = {len(data)} \n initial guess = {initial_guess}",
              fontdict={'fontsize': 8})
    plt.figtext(0.0, 0.01, f"data: {filename}", fontdict={'fontsize': 8})
    plt.suptitle(plotname)

    #plt.show()
    plt.savefig(f"plots/{filename}_PropVsCF2.png")


def plotBoxplot(docker: bool, filename: str, plotname: str):
    if docker:
        data = openFile(filename=filename, directory="results_qaoa_sweep/")
        data = data["results"]
    else:
        data = openFile(filename=filename, directory="results_qaoa/qaoaCompare/")

    bitstrings = list(data["1"]["counts"].keys())
    bitstrings.sort()
    shots = data["1"]["shots"]
    subfile = data["1"]["filename"]
    if docker:
        subdata = openFile(filename=subfile, directory="results_qaoa_sweep/")
    else:
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
    plt.figtext(0.0, 0.01, f"data: {filename}", fontdict={'fontsize': 8})
    plt.suptitle(plotname)
    plt.xticks(range(1, len(bitstrings)+1), bitstrings, rotation=70)
    plt.setp(bp['whiskers'], color='k', linestyle='-')
    plt.setp(bp['fliers'], markersize=2.0)
    #plt.show()
    plt.savefig(f"plots/{filename}_boxplot.png")


def plotCFoptimization(docker: bool, filename: str, plotname:str):
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
    plt.savefig(f"plots/{filename}_CF_optimization.png")



def main():
    plotBoxplot(docker= True, filename="info_testNetwork4Qubit_2_0_20.nc_30_1_2022-02-07_17-54-30",
                plotname="simulator with noise using SPSA - maxiter 50 \n docker run")
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