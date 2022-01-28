import matplotlib.pyplot as plt
import numpy as np
import json

FILENAME = "QaoaCompare_2022-1-28_12-45-11_457427"

with open(f"results_qaoa/qaoaCompare/{FILENAME}.json") as json_file:
    data = json.load(json_file)

bitstrings = list(data["1"]["counts"].keys())
bitstrings.sort()
shots = sum(data["1"]["counts"].values())
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
plt.suptitle(f"rep = {len(data)}, shots = {shots}") # , seed number = 10
plt.xticks(range(1, 8), bitstrings)
plt.setp(bp['whiskers'], color='k', linestyle='-')
plt.setp(bp['fliers'], markersize=3.0)
#plt.show()
plt.savefig(f"results_qaoa/qaoaCompare/{FILENAME}.png")