from taipy.gui import Markdown

selected_scenario = None
figure = None
    
def on_change(state, var_name):
    if var_name == "selected_scenario":
        state.figure = state.selected_scenario.fig.read()

root_page = """
<|container|

# Decision region plots from Sklearn models 
*Dataset used: make_moon from sklearn.datasets*

<br/>

### Select a model:

<layout_scenario|layout|columns=1 2|

<|{selected_scenario}|scenario_selector|show_add_button=False|>

<scenario|part|render={selected_scenario}|

<|chart|figure={figure}|>

|scenario>

|layout_scenario>

|>
"""

interface = Markdown(root_page)
