import matplotlib.pyplot as plt
import numpy as np
import json

FILENAME = "QaoaCompare_2022-1-28_10-6-20_913307"

with open(f"results_qaoa/qaoaCompare/{FILENAME}.json") as json_file:
    data = json.load(json_file)

bitstrings = list(data["1"]["counts"].keys())
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
plt.title(FILENAME)
plt.xticks(range(1, 8), bitstrings)
plt.setp(bp['whiskers'], color='k', linestyle='-')
plt.setp(bp['fliers'], markersize=3.0)
#plt.show()
plt.savefig(f"results_qaoa/qaoaCompare/{FILENAME}.png")