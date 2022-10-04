"""
This file is an example how to use plot_results.py to generate plots of the optimization
runs. This involves two step. First you have to load a data set. We can use that data to
make an instance of `PlotAgent` which can generate plots based on the data you provided to it.

There are two ways to load the data which happens when you make a PlotAgent. The first way
is to search the various results folders using glob expressions. You can also apply filters
to the search results which will filter out all result files who don't have values specified by
you. If you have a lot of optimization runs, this will take a lot of time. Therefore,
`plot_results` can also generate a PlotAgent by using a pandas dataframe. All plots will then
be based on that data frame. If you use a glob to search the various json files, you
can also save the pandas dataframe that you need to quickly reread the same data set.

After initializing a PlotAgent, you can then use that data to generate plots using
`make_figure`. In general, this requires the name how you want to save the plot. More information
on how to use it can be found in the documentation of that method.

You can then run the script `make_plots.py` by either running it directly, or making the recipe
`plots` which activates the appropriate environment for you.
"""

from plot_results import *


def main():
    plt.style.use("seaborn")

    # In order to search the result folders, the plot_agent requires a dictionary when instantiating it.
    # The keys denote which solver result folder will be searched and the value is a list of glob expressions
    # that will be used in the corresponding result folder of the solver
    # The constraints argument is a dictionary. The keys are names of result parameters and the values
    # are lists of parameter values. The result files will be filtered such that if the key of the constraints
    # is also the in the json of the result, the value in that json has to be in the value of that key.

    # The following code block searches all sqa results for files containing the `defaultnetwork.nc`
    # network in it's name with any additional ending. All sqa results which weren't run with the trotter
    # slices config config of either 100, 150 or 200 will be discarded.
    networkbatch = "defaultnetwork.nc"
    plot_agent = PlottingAgent.extract_from_json({
        "sqa": [networkbatch +"*"],
        },
        # constraints={"trotter_slices": [100,150,200]}
        )

    # This will create a plot nammed "first_figure" with the trotter slices on the x-axis and
    # two lines: one for the average kirchhoff cost and one for the average marginal costs
    # The empty `split_fields` means that the results aren't grouped by the value of a third
    # parameter
    plot_agent.make_figure("first_figure",
                            x_field = "trotter_slices",
                            y_field_list = ["kirchhoff_cost", "marginal_cost"],
                            split_fields = []
                            )

    # This dump the underlying pandas DataFrame as a csv file with the git root as the current directory
    plot_agent.to_csv("results_csv/some_results.csv")

    # We can remake a plot_agent with the same data set by reading the dumped csv
    plot_agend = plot_agent.read_csv("results_csv/some_results.csv")
    plot_agent.make_figure("same_figure_from_same_csv",
                            x_field = "trotter_slices",
                            y_field_list = ["kirchhoff_cost", "marginal_cost"],
                            split_fields = []
                            )
    return


if __name__ == "__main__":
    main()
