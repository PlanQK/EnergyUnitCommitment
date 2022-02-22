import itertools
from datetime import datetime

import numpy as np
import json
import pandas as pd
from scipy import stats
from numpy import mean, std, median
import matplotlib.pyplot as plt


def openFile(filename: str, directory: str = "results_qaoa_sweep/") -> dict:
    with open(f"{directory}{filename}") as json_file:
        data = json.load(json_file)

    return data


def extractCFvalue(filename: str, directory: str = "results_qaoa_sweep/") -> list:
    cfValues = []
    data = openFile(filename=filename, directory=directory)
    data = data["results"]
    for key in data:
        filenameTemp = data[key]["filename"]
        dataTemp = openFile(filename=filenameTemp, directory=directory)
        cfValues.append(dataTemp["optimizeResults"]["fun"])

    return cfValues


def intTupleToString(a: tuple):
    string = ""
    for i in range(len(a)):
        string += str(a[i])

    return string


def getBitstrings(nBits: int):
    bitstrings = []
    tupleBits = list(itertools.product([0,1], repeat=nBits))
    for i in range(len(tupleBits)):
        bitstrings.append(intTupleToString(tupleBits[i]))

    return bitstrings


def extractP(filename: str, directory: str = "results_qaoa_sweep/") -> dict:
    pValues = {}
    data = openFile(filename=filename, directory=directory)
    data = data["results"]
    bitstrings = getBitstrings(len(list(data["1"]["counts"].keys())[0]))
    for bitstring in bitstrings:
        pValues[bitstring] = []
        for key in data:
            if bitstring in data[key]["counts"]:
                pValues[bitstring].append(data[key]["counts"][bitstring]/data[key]["shots"])
            else:
                pValues[bitstring].append(0)

    return pValues


def compareBitStringToRest(bitstring: str, pValues: dict) -> dict:
    pValues = pValues
    pValueKeys = list(pValues.keys())
    pValueKeys.remove(bitstring)
    results = {"bestBitstring": bitstring,
               "bitstrings": [],
               "stat": [],
               "p": []}
    for bitstring2 in pValueKeys:
        stat, p = stats.mannwhitneyu(x=pValues[bitstring], y=pValues[bitstring2], alternative="greater")
        results["bitstrings"].append(bitstring2)
        results["stat"].append(stat)
        results["p"].append(p)

    return results


def distributionWithinSample(alpha: float, pValues: dict) -> dict:
    bestKey = list(pValues.keys())[0]
    bestMedian = median(pValues[bestKey])

    for bitstring in pValues:
        if median(pValues[bitstring]) > bestMedian:
            bestMedian = median(pValues[bitstring])
            bestKey = bitstring
    results = compareBitStringToRest(bitstring=bestKey, pValues=pValues)
    results["bestMedian"] = bestMedian
    results["H0"] = {"bitstrings": [], "stat": [], "p": []}
    results["H1"] = {"bitstrings": [], "stat": [], "p": []}
    for i in range(len(results["bitstrings"])):
        if results["p"][i] > alpha:
            results["H0"]["bitstrings"].append(results["bitstrings"][i])
            results["H0"]["stat"].append(results["stat"][i])
            results["H0"]["p"].append(results["p"][i])
        else:
            results["H1"]["bitstrings"].append(results["bitstrings"][i])
            results["H1"]["stat"].append(results["stat"][i])
            results["H1"]["p"].append(results["p"][i])

    if results["H0"]["p"]:
        results["H0"]["minP"] = min(results["H0"]["p"])
        results["H0"]["maxP"] = max(results["H0"]["p"])

    if results["H1"]["p"]:
        results["H1"]["minP"] = min(results["H1"]["p"])
        results["H1"]["maxP"] = max(results["H1"]["p"])

    return results


def equalSamples(alpha: float, file1: str, file2: str, outfile: str, alternative: str = "two-sided") -> None:
    """
    Determine if two samples are equal using Mann Whitney U Test. Only performs test if both samples have the same best
    Bitstring. Checks as well if best bitstring can be found with certainty performing one-sided Mann Whitney U Tests
    between the best bitstring sample data and the sample data for each other non-ideal bitstring.
    Args:
        alpha: confidence of test
        file1: filename of results data 1
        file2: filename of results data 2
        outfile: filename of the output file, to be saved to the Folder "statistics"
        alternative: alternative to be used for Mann Whitney U Test.
                     "two-sided" -> H1: distributions are not equal;
                     "less" -> H1: distribution of file 1 is stochastically less than distribution of file 2
                     "greater" -> H1: distribution of file 1 is stochastically greater than distribution of file 2

    Returns:

    """
    pValues1 = extractP(filename=file1)
    distSample1 = distributionWithinSample(alpha=alpha, pValues=pValues1)

    pValues2 = extractP(filename=file2)
    distSample2 = distributionWithinSample(alpha=alpha, pValues=pValues2)

    if distSample1["bestBitstring"] == distSample2["bestBitstring"]:
        stat, p = stats.mannwhitneyu(x=pValues1[distSample1["bestBitstring"]], y=pValues2[distSample1["bestBitstring"]], alternative=alternative)

    out = {"alternative": alternative,
           "stat": stat,
           "p": p,
           "sample1": {"filename": file1,
                       "distribution": distSample1},
           "sample2": {"filename": file2,
                       "distribution": distSample2}}

    now = datetime.today()
    timeStamp = f"{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}"
    with open(f"statistics/{timeStamp}__{outfile}.json", "w") as write_file:
        json.dump(out, write_file, indent=2, default=str)

def main():
    alpha = 0.05
    equalSamples(alpha=alpha,
                 file1="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-18_08-34-00_config_23.yaml",
                 file2="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-18_08-34-00_config_22.yaml",
                 outfile="YesNoise_QASM-AER_two-sided", alternative="two-sided")
    return
    equalSamples(alpha=alpha,
                 file1="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-17_17-21-18_config_01.yaml",
                 file2="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-17_17-21-18_config_06.yaml",
                 outfile="NoNoise_QASM-State_two-sided", alternative="two-sided")
    equalSamples(alpha=alpha,
                 file1="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-17_17-21-18_config_01.yaml",
                 file2="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-17_17-21-18_config_05.yaml",
                 outfile="NoNoise_QASM-AERState_two-sided", alternative="two-sided")

    equalSamples(alpha=alpha,
                 file1="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-17_17-21-18_config_01.yaml",
                 file2="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-17_17-21-18_config_03.yaml",
                 outfile="NoNoise_matrix-Iter_two-sided", alternative="two-sided")
    equalSamples(alpha=alpha,
                 file1="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-17_17-21-18_config_02.yaml",
                 file2="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-17_17-21-18_config_04.yaml",
                 outfile="YesNoise_matrix-Iter_greater", alternative="greater")

    return


if __name__ == '__main__':
    main()