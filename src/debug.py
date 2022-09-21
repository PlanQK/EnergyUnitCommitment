"""This file is meant as an entrypoint for debugging
"""
import pdb

from program import run

def main():
    network = "defaultnetwork.nc"
    params = "config-all.yaml"

    # pdb.set_trace()
    response = run(data=network, params=params)

if __name__ == "__main__":
    main()
