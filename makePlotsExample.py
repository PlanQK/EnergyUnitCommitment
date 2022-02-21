from plot_results import *

def __main__():

    plt.style.use("seaborn")

    regex = "*1[0-4]_[0-9]_20.nc_100_200_full*60_200_1"
    networkbatch = "infoNocost_220124cost5input_" 

    PlotAgent = PlottingAgent({
            "sqa" : [networkbatch+ "*[36592]0_?_20.nc*"],
            "pypsa_glpk" : [networkbatch+ "*[36592]0_?_20.nc*"],
    },
    constraints={
        "problemSize" : [30,60,90,120,150],
        "trotterSlices" : [500],
    })
    PlotAgent.makeFigure("large_problem_sqa_glpk",
            splitFields=["solver"], 
    )
    PlotAgent.makeFigure("large_problem_time_sqa_glpk",
            yFieldList = ["optimizationTime"],
            splitFields=["solver"], 
    )
    PlotAgent.makeFigure("large_problem_time_and_cost_sqa_glpk",
            yFieldList = ["optimizationTime", "totalCost"],
            splitFields=["solver"], 
    )

    # Add additional plots 
    return


if __name__ == "__main__":
    main()
