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


def __main__():
    plt.style.use("seaborn")

    networkbatch = "infoNocost_220124cost5input_"

    # In order to search the result folders, the plot_agent requires a dictionary when instantiating it.
    # The keys denote which solver result folder will be searched and the value is a list of glob expressions
    # that will be used in the corresponding result folder of the solver
    # The constraints argument is a dictionary. The result files will be filtered such that if a key of the constraints
    # is also the in the json of the result, the value in that json has to be in the value of that key.
    plot_agent = PlottingAgent.extract_from_json({
        "sqa": [networkbatch + "*[36592]0_?_20.nc*"],
        "pypsa_glpk": [networkbatch + "*[36592]0_?_20.nc*"],
        },
        constraints={
            "problemSize": [30, 60, 90, 120, 150],
            "trotterSlices": [500],
        })

    plot_agent.make_figure("large_problem_sqa_glpk",
                           split_fields=["solver"],
                           )
    plot_agent.make_figure("large_problem_time_sqa_glpk",
                           y_field_list=["optimization_time"],
                           split_fields=["solver"],
                           )
    plot_agent.make_figure("large_problem_time_and_cost_sqa_glpk",
                           y_field_list=["optimization_time", "total_cost"],
                           split_fields=["solver"],
                           )

    # dump pandas dataframe as csv with git root as current directory
    plot_agent.to_csv("results_csv/some_results.csv")
    # remake plot_agent with the same data set
    plot_agent.read_csv("some_results.csv")

    # Add additional plots
    return


if __name__ == "__main__":
    main()
