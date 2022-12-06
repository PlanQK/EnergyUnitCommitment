"""This file is meant as an entrypoint for debugging
"""
import pdb

from program import run


def main():
    """The main function used as an entrypoint for debugging"""
    network = "defaultnetwork.nc"
    params = "config-all.yaml"

    pdb.set_trace()
    run(data=network, params=params)


if __name__ == "__main__":
    main()
