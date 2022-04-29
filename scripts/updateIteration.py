import math
import sys

# Small script to convert EstimationFactors and absolute estimation values and get the next iteration step
# in the iterative procedure to obtain a good QUBO for the unit commitment problem with cost minimization

docstring = """The signature is: d_alt, c_alt, kirchhoff

    d_alt: absolute value if the last estimation used to build the qubo

    c_alt: absolute value of the marginal cost incurred by the last solution

    kirchhoff: Kirchhoff cost incurred by the last solution
"""


def main():
    if len(sys.argv) == 1 or  sys.argv[1] == "arg":
        print(docstring)
        return
    EstAt1 = 307.973
    print(f"Pol0: {EstAt1* 1.22717}")
    print(f"OPT_ABS::{377.8138072482}")
    print(f"OPT::{377.8138072482/EstAt1}")
    MonFakInverse = 100.0
    d_alt = float(sys.argv[1])
    c_alt = float(sys.argv[2])
    Kirchhoff = float(sys.argv[3])

#    new_d = d_alt + math.sqrt((d_alt-c_alt) ** 2 + MonFakInverse * Kirchhoff)
    new_d = d_alt + (c_alt - Kirchhoff) / MonFakInverse
    print("HERE")
    print(f"Neue Schatz:: {new_d}")
    print(f"Faktor :: {new_d / EstAt1}")


if __name__ == "__main__":
    main()

