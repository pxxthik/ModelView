import taipy as tp

from time import time
from config import configure
from interface import interface
from taipy import Core, Gui
from sklearn.datasets import make_moons

if __name__ == "__main__":
    core = Core()
    my_scenario = configure()
    core.run()

    start = time()

    dataset = make_moons(noise=0.3, random_state=42)
    for model_name in ["RandomForestClassifier", "SVC", 
                       "KNeighborsClassifier", "LogisticRegression",
                       "AdaBoostClassifier", "GradientBoostingClassifier",
                       "DecisionTreeClassifier", "GaussianNB"]:
        scenario = tp.create_scenario(my_scenario, name=model_name)

        scenario.X.write(dataset[0])
        scenario.y.write(dataset[1])
        scenario.model_name.write(model_name)

        tp.submit(scenario)
    
    print(f"Total time {time()-start}")

    # Instantiate, configure and run the GUI
    gui = Gui(pages={"/": interface})

    gui.run(dark_mode=False, port=3559, title="Taipy Scikit Demo App")
