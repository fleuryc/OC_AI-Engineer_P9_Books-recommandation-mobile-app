"""Helper functions, not project specific."""
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from numpy import exp
from numpy.core.fromnumeric import repeat, shape  # noqa: F401,W0611
from scipy.stats import f_oneway
from sklearn.base import ClassifierMixin, is_classifier
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

# Set the default theme
template = go.layout.Template()
template.layout = go.Layout(
    title_x=0.5,
    # border width and size
    margin=dict(l=2, r=2, b=2, t=30),
    height=400,
    # Interaction
    hovermode="closest",
    # axes
    xaxis_showline=True,
    xaxis_linewidth=2,
    yaxis_showline=True,
    yaxis_linewidth=2,
    # Pick a slightly different P.O.V from default
    # this avoids the extremities of the y and x axes
    # being cropped off
    scene_camera=dict(eye=dict(x=1.5, y=1.5, z=0.1)),
)

template.data.scatter = [go.Scatter(marker=dict(opacity=0.8))]
template.data.scatter3d = [go.Scatter3d(marker=dict(opacity=0.8))]
template.data.surface = [go.Surface()]
template.data.histogram = [go.Histogram(marker=dict(line=dict(width=1)))]
template.data.box = [go.Box(boxpoints="outliers", notched=False)]


pio.templates["custom_template"] = template
pio.templates.default = "plotly_white+custom_template"

# Trendline colors
# Take note that the text for this course often refers to colours explicitly
# such as "looking at the red line". Changing the variable below may result
# in this text being inconsistent
colours_trendline = px.colors.qualitative.Set1


def _to_human_readable(text: str):
    """
    Converts a label into a human readable form
    """
    return text.replace("_", " ")


def _prepare_labels(
    df: pd.DataFrame,
    labels: List[Optional[str]],
    replace_nones: Union[bool, List[bool]] = True,
):
    """
    Ensures labels are human readable.
    Automatically picks data if labels not provided explicitly
    """

    human_readable = {}

    if isinstance(replace_nones, bool):
        replace_nones = [replace_nones] * len(labels)

    for i, lab in enumerate(labels):
        lab = labels[i]
        if replace_nones[i] and (lab is None):
            lab = df.columns[i]
            labels[i] = lab

        # make human-readable
        if lab is not None:
            human_readable[lab] = _to_human_readable(lab)

    return labels, human_readable


def box_and_whisker(
    df: pd.DataFrame,
    label_x: Optional[str] = None,
    label_y: Optional[str] = None,
    label_x2: Optional[str] = None,
    title=None,
    show: bool = False,
):
    """
    Creates a box and whisker plot and optionally shows it.
    Returns the figure for that plot.

    Note that if calling this from jupyter notebooks and not capturing the
    output it will appear on screen as though `.show()` has been called

    df: The data
    label_x: What to group by. Defaults to None
    label_y: What to plot on the y axis. Defaults to count of df.columns[0]
    label_x2: If provided, splits boxplots into 2+ per x value, each with its own colour
    title: Plot title
    show:   appears on screen. NB that this is not needed if this is called from a
            notebook and the output is not captured

    """

    # Automatically pick columns if not specified
    selected_columns, axis_labels = _prepare_labels(
        df, [label_x, label_y, label_x2], replace_nones=[False, True, False]
    )

    fig = px.box(
        df,
        x=selected_columns[0],
        y=selected_columns[1],
        color=label_x2,
        labels=axis_labels,
        title=title,
    )

    # Show the plot, if requested
    if show:
        fig.show()

    # return the figure
    return fig


def histogram(
    df: pd.DataFrame,
    label_x: Optional[str] = None,
    label_y: Optional[str] = None,
    label_colour: Optional[str] = None,
    nbins: Optional[int] = None,
    title=None,
    include_boxplot=False,
    histfunc: Optional[str] = None,
    show: bool = False,
):
    """
    Creates a 2D histogram and optionally shows it.
    Returns the figure for that histogram.

    Note that if calling this from jupyter notebooks and not capturing the output
    it will appear on screen as though `.show()` has been called

    df: The data
    label_x: What to bin by. Defaults to df.columns[0]
    label_y: If provided, the sum of these numbers becomes the y axis.
        Defaults to count of label_x
    label_colour: If provided, creates a stacked histogram,
        splitting each bar by this column
    title: Plot title
    nbins: the number of bins to show. None for automatic
    histfunc: How to calculate y. See plotly for options
    show:   appears on screen. NB that this is not needed if this is called from a
            notebook and the output is not captured

    """

    # Automatically pick columns if not specified
    selected_columns, axis_labels = _prepare_labels(
        df, [label_x, label_y, label_colour], replace_nones=[True, False, False]
    )

    fig = px.histogram(
        df,
        x=selected_columns[0],
        y=selected_columns[1],
        nbins=nbins,
        color=label_colour,
        labels=axis_labels,
        title=title,
        marginal="box" if include_boxplot else None,
        histfunc=histfunc,
    )

    # Set the boxplot notches to False by default to deal with plotting bug
    # But only call this line if the user wants to include a boxplot
    if include_boxplot:
        fig.data[1].notched = False

    # Show the plot, if requested
    if show:
        fig.show()

    # return the figure
    return fig


def multiple_histogram(
    df: pd.DataFrame,
    label_x: str,
    label_group: str,
    label_y: Optional[str] = None,
    histfunc: str = "count",
    nbins: Optional[int] = None,
    title=None,
    show: bool = False,
):
    """
    Creates a 2D histogram and optionally shows it.
    Returns the figure for that histogram.

    Note that if calling this from jupyter notebooks and not capturing the output
    it will appear on screen as though `.show()` has been called

    df: The data
    label_x: What to bin by. Defaults to df.columns[0]
    label_y: If provided, the sum of these numbers becomes the y axis.
        Defaults to count of label_x
    title: Plot title
    nbins: the number of bins to show. None for automatic
    show:   appears on screen. NB that this is not needed if this is called from a
            notebook and the output is not captured

    """

    if histfunc == "count" and label_y is not None:
        raise ValueError("Set histfunc to a value such as sum or avg if using label_y")

    # Automatically pick columns if not specified
    selected_columns, axis_labels = _prepare_labels(
        df, [label_x, label_y, label_group], replace_nones=[True, False, False]
    )

    fig = go.Figure(
        layout=dict(
            title=title,
            xaxis_title_text=axis_labels[label_x],
            yaxis_title_text=histfunc
            if label_y is None
            else (histfunc + " of " + axis_labels[label_y]),
        )
    )

    group_values = sorted(set(df[label_group]))

    for group_value in group_values:
        dat = df[df[label_group] == group_value]
        x = dat[selected_columns[0]]

        if label_y is None:
            y = None
        else:
            y = dat[selected_columns[1]]

        fig.add_trace(
            go.Histogram(
                x=x,
                y=y,
                histfunc=histfunc,
                name=group_value,  # name used in legend and hover labels
                nbinsx=nbins,
            )
        )

    # Place legend title
    fig.update_layout(legend_title_text=label_group)

    # Show the plot, if requested
    if show:
        fig.show()

    # return the figure
    return fig


def line_2D(
    trendline: Union[
        Tuple[str, Callable], List[Tuple[str, Callable]], Dict[str, List[float]]
    ],
    x_range: List[float] = [0, 1],
    label_x: str = "x",
    label_y: str = "y",
    legend_title: str = "Line",
    title=None,
    show: bool = False,
):
    """
    Creates a 2D line plot *using functions* and optionally shows it.
    Returns the figure for that plot.
    If you simply want a line plot using data,
    call scatter_2D then write fig.update_traces(mode='lines')

    Note that if calling this from jupyter notebooks and not capturing the output
    it will appear on screen as though `.show()` has been called

    trendline:  (name, function) tuples. The functions accept X (a numpy array)
        and return Y (an iterable). Alternatively a dict of pre-calculated values
    x_range:    Sets the x-axis range. If this has more than three values,
        it is interpeted as each x-value to be graphed
    label_x:    The title for the x-axis
    label_y:    The title for the y-axis
    legend_title: The title for the legend
    title:      The plot title. If None and a single function is provided,
        the title is automatically set. Use "" to avoid
    show:   appears on screen. NB that this is not needed if this is called from a
            notebook and the output is not captured

    """

    if isinstance(trendline, tuple):
        trendline = [trendline]

    x = np.array([])
    y = np.array([])

    if len(x_range) == 2:
        x_vals = np.linspace(x_range[0], x_range[1], num=200)
    else:
        # X-range is interpreted as x_vals
        x_vals = np.array(x_range)
        x_vals.sort()

        # Rewrite x_range to actually be an x-axis range
        x_range = [x_vals[0], x_vals[-1]]

    names = []  # type: ignore

    if isinstance(trendline, dict):
        for cur in trendline.items():
            name = cur[0]
            x = np.concatenate([x, x_vals])
            names = names + ([name] * len(x_vals))
            y = np.concatenate([y, cur[1]])
    else:
        for cur in trendline:  # type: ignore
            name = cur[0]
            x = np.concatenate([x, x_vals])
            names = names + ([name] * len(x_vals))
            y = np.concatenate([y, cur[1](x=x_vals)])  # type: ignore

    data = {}
    data[label_x] = x
    data[label_y] = y
    data[legend_title] = names  # type: ignore

    df = pd.DataFrame(data)

    # Pick a title if none provided and we only have one function
    if (title is None) and (len(trendline) == 1):
        title = trendline[0][0]  # type: ignore

    # Create as a 2d scatter but with lines
    fig = scatter_2D(
        df, label_colour=legend_title, title=title, show=False, x_range=x_range
    )
    fig.update_traces(mode="lines")

    # Don't show a legend if we only have one function plotted
    if len(trendline) == 1:
        fig.update_layout(showlegend=False)

    if show:
        fig.show()

    return fig


def scatter_2D(
    df: pd.DataFrame,
    label_x: Optional[str] = None,
    label_y: Optional[str] = None,
    label_colour: Optional[str] = None,
    label_size: Optional[str] = None,
    size_multiplier: float = 1,
    title=None,
    show: bool = False,
    x_range: Optional[List[float]] = None,
    trendline: Union[Callable, List[Callable], None] = None,
):
    """
    Creates a 2D scatter plot and optionally shows it.
    Returns the figure for that scatter.

    Note that if calling this from jupyter notebooks and not capturing the output
    it will appear on screen as though `.show()` has been called

    df: The data
    label_x: The label to extract from df to plot on the x axis.
        Defaults to df.columns[0]
    label_y: The label to extract from df to plot on the y axis.
        Defaults to df.columns[1]
    label_colour: The label to extract from df to colour points by
    title: Plot title
    show:   appears on screen. NB that this is not needed if this is called from a
            notebook and the output is not captured
    x_range:    Overrides the x-axis range
    trendline:  A function that accepts X (a numpy array) and returns Y (an iterable)

    """

    # Automatically pick columns if not specified
    selected_columns, axis_labels = _prepare_labels(
        df, [label_x, label_y, label_colour], [True, True, False]
    )

    # Create the figure and plot
    fig = px.scatter(
        df,
        x=selected_columns[0],
        y=selected_columns[1],
        color=selected_columns[2],
        labels=axis_labels,
        hover_data=[label_size],
        title=title,
    )

    if label_size is None:
        # User a marker size inversely proportional to the number of points
        size = int(
            (round(22.0 - 19 / (1 + exp(-(df.shape[0] / 100 - 2)))) * size_multiplier)
        )
    else:
        # Set the size based on a label
        size = df[label_size] * size_multiplier

    fig.update_traces(marker={"size": size})

    if x_range is not None:
        fig.update_xaxes(range=[x_range[0], x_range[1]])

    # Create trendlines
    if trendline is not None:
        if isinstance(trendline, Callable):  # type: ignore
            trendline = [trendline]  # type: ignore
        x_min = min(df[selected_columns[0]]) if x_range is None else x_range[0]
        x_max = max(df[selected_columns[0]]) if x_range is None else x_range[1]
        evaluate_for = np.linspace(x_min, x_max, num=200)
        shapes = []
        for t, colour in zip(trendline, colours_trendline):  # type: ignore
            y_vals = t(evaluate_for)
            path = "M" + " L ".join(
                [str(c[0]) + " " + str(c[1]) for c in zip(evaluate_for, y_vals)]
            )
            shapes.append(
                dict(
                    type="path",
                    path=path,
                    line_color=colour,
                )
            )

        fig.update_layout(shapes=shapes)

    # Show the plot, if requested
    if show:
        fig.show()

    # return the figure
    return fig


def scatter_3D(
    df: pd.DataFrame,
    label_x: Optional[str] = None,
    label_y: Optional[str] = None,
    label_z: Optional[str] = None,
    label_colour: Optional[str] = None,
    title=None,
    show: bool = False,
):
    """
    Creates a 3D scatter plot and optionally shows it.
    Returns the figure for that scatter.

    Note that if calling this from jupyter notebooks and not capturing the output
    it will appear on screen as though `.show()` has been called

    df: The data
    label_x: The label to extract from df to plot on the x axis.
        Defaults to df.columns[0]
    label_y: The label to extract from df to plot on the y axis.
        Defaults to df.columns[1]
    label_z: The label to extract from df to plot on the z axis.
        Defaults to df.columns[2]
    label_colour: The label to extract from df to colour points by. Defaults to label_x
    title: Plot title
    show:   appears on screen. NB that this is not needed if this is called from a
            notebook and the output is not captured
    """

    # Automatically pick columns if not specified
    selected_columns, axis_labels = _prepare_labels(df, [label_x, label_y, label_z])

    if label_colour is None:
        # Colour by the Z dimension
        label_colour = selected_columns[2]
    else:
        axis_labels[label_colour] = _to_human_readable(label_colour)

    # Create the figure and plot
    fig = px.scatter_3d(
        df,
        x=selected_columns[0],
        y=selected_columns[1],
        z=selected_columns[2],
        color=label_colour,
        labels=axis_labels,
        title=title,
    )

    # Show the plot, if requested
    if show:
        fig.show()

    # return the figure
    return fig


def surface(
    x_values,
    y_values,
    calc_z: Callable,
    title=None,
    axis_title_x: Optional[str] = None,
    axis_title_y: Optional[str] = None,
    axis_title_z: Optional[str] = None,
    show: bool = False,
):
    """
    Creates a surface plot using a function. Returns the figure for that plot.

    Note that if calling this from jupyter notebooks and not capturing the output
    it will appear on screen as though `.show()` has been called

    x_value: A numpy array of x values
    y_value: A numpy array of y values
    calc_z: A function to calculate z, given an x and a y value
    title: Plot title
    axis_title_x: Title for the x axis
    axis_title_y: Title for the y axis
    axis_title_z: Title for the z axis
    show:   appears on screen. NB that this is not needed if this is called from a
            notebook and the output is not captured
    """

    # Check arguments
    if not len(x_values.shape) == 1:
        raise ValueError("x_values must be a 1D array")
    if not len(y_values.shape) == 1:
        raise ValueError("y_values must be a 1D array")

    # Calculate z for a range of x and y inputs
    # Note that z seems to be expected to be indexed [y,x] not [x,y]
    # though this appears to be counter to the documentation.
    # If z is indexed [x, y] the result is flipped.
    # Potentially there is a bug here somewhere causing this issue or in plotly itself
    z = np.zeros((y_values.shape[0], x_values.shape[0]))
    for i_x in range(x_values.shape[0]):
        for i_y in range(y_values.shape[0]):
            z[i_y, i_x] = calc_z(x_values[i_x], y_values[i_y])

    # Create a graph of cost
    fig = go.Figure(data=[go.Surface(x=x_values, y=y_values, z=z)])
    fig.update_layout(
        title=title,
        scene_xaxis_title=axis_title_x,
        scene_yaxis_title=axis_title_y,
        scene_zaxis_title=axis_title_z,
    )

    # Add z-axis as colourbar title
    fig.update_traces(colorbar_title_text=axis_title_z, selector=dict(type="surface"))

    # Show the plot, if requested
    if show:
        fig.show()

    # return the figure
    return fig


def model_to_surface_plot(model, plot_features: List[str], data: pd.DataFrame):
    """Plots two features of a model as a surface. Other values are set at their means

    model:          A model that accepts a dataframe for prediction
    plot_features:  Two features to plot
    data:           A dataframe the model was trained or tested on
    """

    # Give status as this can take several seconds to run
    print("Creating plot...")

    other_features = [f for f in data.columns if f not in plot_features]

    means = np.average(data[other_features], axis=0)
    mins = np.min(data[plot_features], axis=0)
    maxes = np.max(data[plot_features], axis=0)

    df = pd.DataFrame()

    for f, m in zip(other_features, means):
        df[f] = [m]

    def predict(x, y):
        """
        Makes a prediction using the model
        """
        df[plot_features[0]] = [x]
        df[plot_features[1]] = [y]

        return model.predict(df)

    # Create a 3d plot of predictions
    x_vals = np.array(np.linspace(mins[plot_features[0]], maxes[plot_features[0]], 20))
    y_vals = np.array(np.linspace(mins[plot_features[1]], maxes[plot_features[1]], 20))

    return surface(
        x_vals,
        y_vals,
        predict,
        title="Model Prediction",
        axis_title_x=plot_features[0],
        axis_title_y=plot_features[1],
        axis_title_z="Probability",
    )


def save_plot_as_image(
    fig, file="./plot.jpg", width=None, height="400", scale=1, img_format="jpg"
):
    """
    Convert a figure to a static image and write it to a file or writeable object
    If "width" not set, plotly will set the aspect ration based on "hight"

    Parameters

        fig - Figure object or dict representing a figure
        file (str or writeable) - A string representing a local file path
            or a writeable object (e.g. an open file descriptor)
        img_format (str or None) - The desired image format:

                'png'
                'jpg' or 'jpeg'
                'webp'
                'svg'
                'pdf'
                'eps' (Requires the poppler library to be installed and on the PATH)

        width (int or None) - The width of the exported image in layout pixels.
        height (int or None) - The height of the exported image in layout pixels.

        scale (int or float or None) - The scale factor to use when exporting
            the figure.
        A scale factor larger than 1.0 will increase the image resolution
        with respect to the figure's layout pixel dimensions.
        Whereas as scale factor of less than 1.0 will decrease the image resolution.
    """
    pio.write_image(
        fig,
        file=file,
        width=width,
        height=height,
        scale=scale,
        format=img_format,
        engine="kaleido",
    )


def plot_oneway_anova_p_values(
    dataframe: pd.DataFrame,
    categorical_column: str,
    classes: tuple[Union[str, float, int], Union[str, float, int]],
) -> None:
    """
    Plot a histogram of the p-values of each numerical column of the dataframe,
    split by categorical column.

    Args:
        dataframe (pd.DataFrame): The DataFrame to plot.
        categorical_column (str): The name of the categorical column.
        classes (tuple): The list of classes of the categorical column.

    Returns:
        None.
    """
    anova = pd.DataFrame({})

    for col in dataframe.select_dtypes("number").columns:
        anova.loc[col, "p_value"] = f_oneway(
            dataframe.loc[dataframe[categorical_column] == classes[0], col].dropna(),
            dataframe.loc[dataframe[categorical_column] == classes[1], col].dropna(),
        )[1]

    # Plot the bar chart with Plotly Express
    fig = px.bar(
        anova.sort_values(by="p_value", ascending=True),
        color="p_value",
        y="p_value",
        hover_data=["p_value"],
        title="ANOVA P-Value",
        labels={
            "p_value": "P-Value",
            "index": "Feature",
        },
    )
    fig.show()


def plot_empty_values(dataframe: pd.DataFrame) -> None:
    """
    Plot a histogram of empty values percentage per columns of the input
    DataFrame.

    Args:
        dataframe: The DataFrame to plot.

    Returns:
        None.
    """
    num_rows = len(dataframe.index)
    if num_rows == 0:
        logging.warning("No data to plot.")
        return

    # Get the percentage of empty values per column
    columns_emptiness = (
        pd.DataFrame(
            {
                col: {
                    "count": dataframe[col].isna().sum(),
                    "percent": 100 * dataframe[col].isna().sum() / num_rows,
                }
                for col in dataframe.columns
            }
        )
        .transpose()
        .sort_values(by=["count"])
    )

    # Plot the bar chart with Plotly Express
    fig = px.bar(
        columns_emptiness,
        color="percent",
        y="percent",
        labels={
            "index": "column name",
            "percent": "% of empty values",
            "count": "# of empty values",
        },
        hover_data=["count"],
        title="Empty values per column",
        width=1200,
        height=800,
    )
    fig.show()


def plot_categories_bars(
    dataframe: pd.DataFrame,
    plot_columns: Optional[list[str]] = None,
    categorical_column: Optional[str] = None,
) -> None:
    """
    Draw one bar chart per categorical or boolean variable, split per class and
    target.

    Arguments :
    - dataframe : Pandas DataFrame containing the data, including the
                  categorical_columns and target column
    - plot_columns : list of columns to plot, if None, all bool & category
                     columns are plotted
    - categorical_column : string representing the name of the variable
                           containing the categories

    Returns : None
    """
    if plot_columns is None:
        plot_columns = dataframe.select_dtypes(
            include=["bool", "category"],
        ).columns.tolist()

    for col in plot_columns:
        df_g = (
            dataframe.groupby([col, categorical_column], dropna=False)
            .size()
            .reset_index()
        )
        df_g["percentage"] = (
            dataframe.groupby([col, categorical_column], dropna=False)
            .size()
            .groupby(level=0)
            .apply(lambda x: 100 * x / float(x.sum()))
            .values
        )
        df_g.columns = [col, categorical_column, "Count", "Percentage"]
        df_g.sort_values(
            by=["Count", "Percentage"],
            ascending=False,
            inplace=True,
        )
        fig = px.bar(
            df_g,
            x=col,
            y=["Count"],
            color=categorical_column,
            hover_data=["Percentage"],
            title=f"{col} Categories distribution and {categorical_column} \
                ratio",
            width=800,
            height=400,
        )
        fig.show()


def plot_boxes(
    dataframe: pd.DataFrame,
    plot_columns: Optional[list[str]] = None,
    categorical_column: Optional[str] = None,
) -> None:
    """Draw one boxplot per numerical variable, split per categories.

    Arguments :
    - dataframe : Pandas DataFrame containing the data, including the
                  categorical_column and numerical_columns
    - plot_columns : list of columns to plot, if None, all numerical columns
                     are plotted
    - categorical_column : string representing the name of the variable
                           containing the categories

    Returns : None
    """
    if plot_columns is None:
        plot_columns = dataframe.select_dtypes(include="number").columns.tolist()

    for col in plot_columns:
        fig = px.box(
            dataframe,
            y=col,
            points=False,
            color=categorical_column,
            title=f"{col} distribution per TARGET",
            width=800,
            height=400,
        )
        fig.update_traces(boxmean="sd")
        fig.update_traces(notched=True)
        fig.show()


def plot_classifier_results(
    classifier: ClassifierMixin,
    X: pd.DataFrame,
    y_true: pd.Series,
    y_pred: Union[pd.Series, None] = None,
    y_pred_proba: Union[pd.Series, None] = None,
    title: str = "Classifier Results",
) -> None:
    """
    Plots the confusion_matrix, precision_recall_curve and roc_curve of a
    classifier on test data.

    Args:
        classifier (ClassifierMixin): sklearn Classifier
        X (pd.DataFrame): test data
        y_true (pd.Series): true values
        y_pred (pd.Series): predicted values
        y_pred_proba (pd.Series): predicted values probabilities
    """
    if (y_pred_proba is None or y_pred is None) and not is_classifier(classifier):
        raise ValueError(f"{classifier} is not a classifier.")

    _, ax = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(24, 8),
    )

    if y_pred_proba is None or y_pred is None:
        ConfusionMatrixDisplay.from_estimator(
            classifier,
            X,
            y_true,
            ax=ax[0],
        )
        PrecisionRecallDisplay.from_estimator(
            classifier,
            X,
            y_true,
            name=classifier.__class__.__name__,
            ax=ax[1],
        )
        RocCurveDisplay.from_estimator(
            classifier,
            X,
            y_true,
            name=classifier.__class__.__name__,
            ax=ax[2],
        )
    else:
        ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred,
            ax=ax[0],
        )
        PrecisionRecallDisplay.from_predictions(
            y_true,
            y_pred_proba,
            name=classifier.__class__.__name__,
            ax=ax[1],
        )
        RocCurveDisplay.from_predictions(
            y_true,
            y_pred_proba,
            name=classifier.__class__.__name__,
            ax=ax[2],
        )

    plt.suptitle(title)

    plt.show()


def plot_permutation_importance(
    model: ClassifierMixin,
    X: pd.DataFrame,
    y: pd.Series,
) -> None:
    """Plot the permutation importances of each variable as a BoxPlot.

    Args:
        model (ClassifierMixin): Fitted classifier
        X (pd.DataFrame): X variables
        y (pd.Series): Targer variable
    """
    importances = permutation_importance(
        model,
        X,
        y,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )
    sorted_idx = importances.importances_mean.argsort()
    sorted_idx = list(sorted_idx[:10]) + list(sorted_idx[-10:])

    fig, ax = plt.subplots()
    ax.boxplot(
        importances.importances[sorted_idx].T,
        vert=False,
        labels=X.columns[sorted_idx],
    )
    ax.set_title("Permutation Importances")
    fig.set_size_inches(10, 10)
    plt.show()


def plot_pca_2d(
    data: pd.DataFrame,
    categories: pd.DataFrame,
) -> None:
    """
    Draw a 2D PCA plot of the data and the feture importances.

    Args:
        data (pd.DataFrame) : Pandas DataFrame containing the data
        categories (pd.DataFrame) : Pandas DataFrame containing the categories

    Returns : None
    """
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)

    # Plot the data in the PCA space
    fig = px.scatter(
        x=data_pca[:, 0],
        y=data_pca[:, 1],
        color=categories.values.flatten(),
        symbol=categories.values.flatten(),
        trendline="ols",
        title="PCA 2D",
        opacity=0.2,
        width=1200,
        height=800,
    )

    # Plot the feature importances in the PCA space
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    for i, feature in enumerate(data.columns):
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=loadings[i, 0],
            y1=loadings[i, 1],
        )
        fig.add_annotation(
            x=loadings[i, 0],
            y=loadings[i, 1],
            ax=0,
            ay=0,
            xanchor="center",
            yanchor="bottom",
            text=feature,
        )

    fig.show()


def plot_top_words(model, feature_names, n_top_words, n_topics, title):
    n_cols = 5
    n_lines = int(np.ceil(min(n_topics, model.n_components) / n_cols))
    fig, axes = plt.subplots(n_lines, n_cols, figsize=(30, n_lines * 5), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_[0:n_topics]):
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 20})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=15)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=30)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()
