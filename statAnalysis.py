import itertools
from datetime import datetime

import numpy as np
import json
import pandas as pd
from scipy import stats
from numpy import mean, std, median, linalg
import matplotlib.pyplot as plt


def openFile(filename: str, directory: str = "results_qaoa_sweep/") -> dict:
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


def extractSortedCF(filename: str, directory: str = "results_qaoa_sweep/") -> dict:
    """
    Extracts the cost function values from the given file and returns them in a sorted dictionary.
    Args:
        filename: (str) The name of the file to be opened.
        directory: (str) The folder in which the file is located. Default: "results_qaoa_sweep/"

    Returns:
        cfValues (dict) The sorted cost function values.
    """
    cfValues = {}
    data = openFile(filename=filename, directory=directory)
    data = data["results"]
    for key in data:
        filenameTemp = data[key]["filename"]
        dataTemp = openFile(filename=filenameTemp, directory=directory)
        cfValues[key] = dataTemp["optimizeResults"]["fun"]

    cfValues = dict(sorted(cfValues.items(), key=lambda item: item[1]))

    return cfValues


def intTupleToString(a: tuple):
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


def getBitstrings(nBits: int):
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


def extractP(filename: str, directory: str = "results_qaoa_sweep/") -> dict:
    """
    Extracts the probability values for all bitstrings from the given file and returns them in a dictionary.
    Args:
        filename: (str) The name of the file to be opened.
        directory: (str) The folder in which the file is located. Default: "results_qaoa_sweep/"

    Returns:
        pValues (dict) The probability values.
    """
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


def extractSortedP(filename: str, cut: float, directory: str = "results_qaoa_sweep/") -> dict:
    """
        Extracts the probability values for all bitstrings from the given file and returns the best {cut} in a sorted
        dictionary.
        Args:
            filename: (str) The name of the file to be opened.
            cut: (float) the percentage of best repetitions to be plotted: Default: 0.5
            directory: (str) The folder in which the file is located. Default: "results_qaoa_sweep/"

        Returns:
            pValues (dict) The probability values.
        """
    pValues = extractP(filename=filename, directory=directory)
    cfValues = extractSortedCF(filename=filename, directory=directory)
    cfValues = dict(sorted(cfValues.items(), key=lambda item: item[1]))

    keys = []
    for i in range(int(len(cfValues) - 1), int(len(cfValues) * cut - 1), -1):
        keys.append(int(list(cfValues.keys())[i]) - 1)
    keys.sort(reverse=True)
    for key in keys:
        for bitstring in pValues:
            pValues[bitstring].pop(key)

    return pValues

def compareBitStringToRest(bitstring: str, pValues: dict) -> dict:
    """
    Determines if the probabilities of the given bitstring and all other bitstrings within pValues are equal using a
    one-sided Mann Whitney U Test. The results are stored in a dictionary and returned.
    Args:
        bitstring: (str) the bitstring to which all other bitstrings should be compared
        pValues: (dict) dictionary of the p-values of all bitstrings

    Returns:
        results (dict) dictionary containing the results of the Mann Whitney U Tests.
    """
    pValues = pValues
    pValueKeys = list(pValues.keys())
    pValueKeys.remove(bitstring)
    results = {"best_bitstring": bitstring,
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
    """
    Determines if the distribution within a sample is clear enough to determine an ideal bitstring. The bitstring with
    the highest median probability is determined and then compared to the probabilities of all other bitsttrings using
    a one-sided Mann Whitney U Test. The results are stored in a dictionary and returned.
    Args:
        alpha: (float) confidence of test.
        pValues: (dict) dictionary of the p-values of all bitstrings

    Returns:
        results (dict) dictionary containing all statistical data to determine if an ideal bitstring can be found.
    """
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


def equalSamples(alpha: float, file1: str, file2: str, outfile: str, cut: float = 0.5, alternative: str = "two-sided") -> None:
    """
    Determine if two samples are equal using Mann Whitney U Test. Only performs test if both samples have the same best
    Bitstring. Checks as well if best bitstring can be found with certainty performing one-sided Mann Whitney U Tests
    between the best bitstring sample data and the sample data for each other non-ideal bitstring.
    Args:
        alpha: (float) confidence of test
        file1: (str) filename of results data 1
        file2: (str) filename of results data 2
        outfile: (str) filename of the output file, to be saved to the Folder "statistics"
        alternative: (str) alternative to be used for Mann Whitney U Test.
                     "two-sided" -> H1: distributions are not equal;
                     "less" -> H1: distribution of file 1 is stochastically less than distribution of file 2
                     "greater" -> H1: distribution of file 1 is stochastically greater than distribution of file 2

    Returns:
        Saves the generated statistics, with the names "{timeStamp}__{outfile}_all.json" and
        "{timeStamp}__{outfile}_cut.json" in the subfolder 'statistics'
    """
    pValues1all = extractSortedP(filename=file1, cut=1.0)
    distSample1all = distributionWithinSample(alpha=alpha, pValues=pValues1all)
    pValues1cut = extractSortedP(filename=file1, cut=cut)
    distSample1cut = distributionWithinSample(alpha=alpha, pValues=pValues1cut)


    pValues2all = extractSortedP(filename=file2, cut=1.0)
    distSample2all = distributionWithinSample(alpha=alpha, pValues=pValues2all)
    pValues2cut = extractSortedP(filename=file2, cut=cut)
    distSample2cut = distributionWithinSample(alpha=alpha, pValues=pValues2cut)

    if distSample1all["best_bitstring"] == distSample2all["best_bitstring"]:
        stat, p = stats.mannwhitneyu(x=pValues1all[distSample1all["best_bitstring"]],
                                     y=pValues2all[distSample1all["best_bitstring"]],
                                     alternative=alternative)
    else:
        stat = None
        p = None
    outAll = {"alternative": alternative,
           "stat": stat,
           "p": p,
           "sample1": {"filename": file1,
                       "distribution": distSample1all},
           "sample2": {"filename": file2,
                       "distribution": distSample2all}}

    if distSample1cut["best_bitstring"] == distSample2cut["best_bitstring"]:
        stat, p = stats.mannwhitneyu(x=pValues1cut[distSample1cut["best_bitstring"]],
                                     y=pValues2cut[distSample1cut["best_bitstring"]],
                                     alternative=alternative)
    else:
        stat = None
        p = None
    outCut = {"alternative": alternative,
              "cut": cut,
           "stat": stat,
           "p": p,
           "sample1": {"filename": file1,
                       "distribution": distSample1cut},
           "sample2": {"filename": file2,
                       "distribution": distSample2cut}}

    now = datetime.today()
    timeStamp = f"{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}"
    with open(f"statistics/{timeStamp}__{outfile}_all.json", "w") as write_file:
        json.dump(outAll, write_file, indent=2, default=str)
    with open(f"statistics/{timeStamp}__{outfile}_cut.json", "w") as write_file:
        json.dump(outCut, write_file, indent=2, default=str)

def main():
    alpha = 0.05
    equalSamples(alpha=alpha,
                 file1="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_65.yaml",
                 file2="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_77.yaml",
                 outfile="YesNoise_20000_COBYLA-SPSA200_greater", cut=0.5, alternative="greater")
    equalSamples(alpha=alpha,
                 file1="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_65.yaml",
                 file2="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_71.yaml",
                 outfile="YesNoise_20000_COBYLA-SPSA100_greater", cut=0.5, alternative="greater")

    return
    equalSamples(alpha=alpha,
                 file1="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_71.yaml",
                 file2="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_77.yaml",
                 outfile="YesNoise_20000_SPSA100-SPSA200_greater", cut=0.5, alternative="greater")
    equalSamples(alpha=alpha,
                 file1="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_71.yaml",
                 file2="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_65.yaml",
                 outfile="YesNoise_20000_SPSA100-COBYLA_greater", cut=0.5, alternative="greater")

    return
    equalSamples(alpha=alpha,
                 file1="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_71.yaml",
                 file2="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_77.yaml",
                 outfile="YesNoise_20000_SPSA100-SPSA200_two-sided", cut=0.5, alternative="two-sided")
    equalSamples(alpha=alpha,
                 file1="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_65.yaml",
                 file2="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_71.yaml",
                 outfile="YesNoise_20000_COBYLA-SPSA100_two-sided", cut=0.5, alternative="two-sided")
    equalSamples(alpha=alpha,
                 file1="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_65.yaml",
                 file2="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_77.yaml",
                 outfile="YesNoise_20000_COBYLA-SPSA200_two-sided", cut=0.5, alternative="two-sided")

    return
    equalSamples(alpha=alpha,
                 file1="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_68.yaml",
                 file2="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_74.yaml",
                 outfile="NoNoise_20000_SPSA100-SPSA200_two-sided", alternative="two-sided")
    equalSamples(alpha=alpha,
                 file1="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_62.yaml",
                 file2="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_68.yaml",
                 outfile="NoNoise_20000_COBYLA-SPSA100_two-sided", alternative="two-sided")
    equalSamples(alpha=alpha,
                 file1="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_62.yaml",
                 file2="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_74.yaml",
                 outfile="NoNoise_20000_COBYLA-SPSA200_two-sided", alternative="two-sided")

    return
    equalSamples(alpha=alpha,
                 file1="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-23_11-08-27_config.yaml",
                 file2="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-23_11-31-00_config.yaml",
                 outfile="YesNoise_20000_COBYLA-old-new_two-sided", alternative="two-sided")
    return
    equalSamples(alpha=alpha,
                 file1="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_71.yaml",
                 file2="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_65.yaml",
                 outfile="YesNoise_20000_COBYLA-SPSA100_less", alternative="less")
    return
    equalSamples(alpha=alpha,
                 file1="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_62.yaml",
                 file2="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_61.yaml",
                 outfile="NoNoise_COBYLA_20000-4096_greater", alternative="greater")
    equalSamples(alpha=alpha,
                 file1="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_62.yaml",
                 file2="infoNocost_testNetwork4QubitIsing_2_0_20.nc_30_1_2022-02-22_17-57-05_config_60.yaml",
                 outfile="NoNoise_COBYLA_20000-1024_greater", alternative="greater")

    return
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