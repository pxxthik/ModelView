from algos import fit, plot
from taipy import Config


def configure():
    X = Config.configure_data_node("X")
    y = Config.configure_data_node("y")
    model_name = Config.configure_data_node("model_name", default_data="MLPClassifier")

    model = Config.configure_data_node("model")
    fit_task = Config.configure_task(
        id="fit", function=fit, input=[X, y, model_name], output=model, skippable=True
    )

    fig = Config.configure_data_node("fig")
    plot_task = Config.configure_task(
        id="plot", function=plot, input=[X, y, model], output=fig, skippable=True
    )

    scenario = Config.configure_scenario(id="scenario", task_configs=[fit_task, plot_task])
    Config.export("scenario.toml")
    return scenario
