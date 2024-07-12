import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

import plotly.express as px


def fit(X, y, model_name):

    if model_name == "RandomForestClassifier":
        model = RandomForestClassifier()
    elif model_name == "SVC":
        model = SVC()
    elif model_name == "KNeighborsClassifier":
        model = KNeighborsClassifier()
    elif model_name == "LogisticRegression":
        model = LogisticRegression()
    elif model_name == "AdaBoostClassifier":
        model = AdaBoostClassifier()
    elif model_name == "GradientBoostingClassifier":
        model = GradientBoostingClassifier()
    elif model_name == "GaussianNB":
        model = GaussianNB()
    elif model_name == "DecisionTreeClassifier":
        model = DecisionTreeClassifier()

    return make_pipeline(
        StandardScaler(),
        model,
    ).fit(X, y)


def plot(X, y, model):
    x1, x2 = X[:, 0], X[:, 1]

    # Create a mesh grid
    x1_range = np.arange(x1.min()-1, x1.max()+1, 0.01)
    x2_range = np.arange(x2.min()-1, x2.max()+1, 0.01)
    xx1, xx2 = np.meshgrid(x1_range, x2_range)

    # Predict on the mesh grid
    y_pred_mesh = model.predict(np.c_[xx1.ravel(), xx2.ravel()]).reshape(xx1.shape)

    # Plot the decision boundaries
    fig = px.scatter(x=x1, y=x2, color=y)
    fig.update_traces(marker={"size": 15})
    # Add decision boundaries
    fig.add_contour(x=x1_range, y=x2_range, z=y_pred_mesh, showscale=False, opacity=0.2)

    return fig
