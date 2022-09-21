"""
This file is an example how to use plot_results.py to generate plots of the optimization 
runs. This involves two step. First you have to load a data set. We can use that data to
make PlotAgent which can generate plots based on the data you provided to it.

There are two ways to load the data which happens when you make a PlotAgent. The first way
is to search the various results folders using glob expressions. You can also apply filters 
to the search results which will filter out all result files who don't have values specified 
you. If you have a lot of optimization runs, this will take a lot of time. Therefore, 
`plot_results` can also generate a PlotAgent by using a pandas dataframe. All plots will then 
be based on the data of that frame. If you use a glob to search the various json files, you 
can also save the pandas dataframe that you need to quickly reread the same data set. In the 
end all results will be saved in a pandas DataFrame.

After initializing a PlotAgent, you can then use that data to generate plots using 
`make_figure`. In general, this requires the name how you want to save the plot. More information 
on how to use it can be found in the documentation of that method.

You can then run the script `make_plots.py` by either running it directly, or making the recipe 
`plots` which activates the appropriate environment for you.
"""

from plot_results import *


def main():
    plt.style.use("seaborn")

    networkbatch = "defaultnetwork.nc_sqa"

    # In order to search the result folders, the plot_agent requires a dictionary when instantiating it.
    # The keys denote which solver result folder will be searched and the value is a list of glob expressions
    # that will be used in the corresponding result folder of the solver
    # The constraints argument is a dictionary. The result files will be filtered such that if a key of the constraints
    # is also the in the json of the result, the value in that json has to be in the value of that key.
    plot_agent = PlottingAgent.extract_from_json({
        "sqa": [networkbatch +"*"],
        # "pypsa_glpk": [networkbatch + "*[36592]0_?_20.nc*"],
        },
        constraints={"trotter_slices": [100,150,200]}
        )

    plot_agent.make_figure("first_figure",
                            x_field = "trotter_slices",
                            y_field_list = ["kirchhoff_cost", "marginal_cost"],
                            split_fields = []
                            )

    # dump pandas DataFrame as csv with git root as current directory
    plot_agent.to_csv("results_csv/some_results.csv")
    # remake plot_agent with the same data set by reading the dumped csv
    plot_agent.read_csv("results_csv/some_results.csv")
    plot_agent.make_figure("same_figure_from_same_csv",
                            x_field = "trotter_slices",
                            y_field_list = ["kirchhoff_cost", "marginal_cost"],
                            split_fields = []
                            )
    # Add additional plots
    return


if __name__ == "__main__":
    main()
