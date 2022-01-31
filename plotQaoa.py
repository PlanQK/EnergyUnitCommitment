import matplotlib.pyplot as plt
import numpy as np
import json

FILENAME = "QaoaCompare_2022-1-31_13-11-6_987313"

with open(f"results_qaoa/qaoaCompare/{FILENAME}.json") as json_file:
    data = json.load(json_file)

bitstrings = list(data["1"]["counts"].keys())
bitstrings.sort()
shots = data["1"]["shots"]
simulator = data["1"]["simulator"]
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
pos = np.arange(len(toPlot)) + 1
bp = ax.boxplot(toPlot, sym='k+', positions=pos, bootstrap=5000)

ax.set_xlabel('bitstrings')
ax.set_ylabel('probability')
plt.title(FILENAME, fontdict = {'fontsize' : 8})
plt.suptitle(f"simulator = {simulator}, shots = {shots}, rep = {len(data)}") # , seed number = 10
plt.xticks(range(1, len(bitstrings)+1), bitstrings, rotation=90)
plt.setp(bp['whiskers'], color='k', linestyle='-')
plt.setp(bp['fliers'], markersize=3.0)
#plt.show()
plt.savefig(f"results_qaoa/qaoaCompare/{FILENAME}.png")