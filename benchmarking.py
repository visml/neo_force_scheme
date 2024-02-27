import os

import numpy as np

# Read data from a csv
import pandas
import pandas as pd
import plotly.graph_objects as go
from sklearn import datasets

from neo_force_scheme import NeoForceScheme, ProjectionMode
from neo_force_scheme.engines.neo_force_scheme_cpu import (
    non_numeric_processor,
)


def datacollecting(
    n_dimension,
    learning_rate,
    decay,
    max_iteration,
    starting_projection,
    dataset,
    dataname,
    fig_path=None,
    scaler=None,
    fixed_axis=None,
    X_excp=None,
    num_process_axis=None,
    z_axis_moving_range=None,
    verbose=False,
):
    # read the distance matrix
    nfs = NeoForceScheme(
        verbose=verbose, learning_rate0=learning_rate, decay=decay, max_it=max_iteration
    )

    if num_process_axis is not None:
        dataset = non_numeric_processor(dataset, num_process_axis)

    projection = nfs.fit_transform(
        X=dataset,
        starting_projection_mode=starting_projection,
        random_state=1,
        n_dimension=n_dimension,
        fix_column_to_z_projection_axis=fixed_axis,
        drop_columns_from_dataset=X_excp,
        scaler=scaler,
        z_axis_moving_range=z_axis_moving_range,
    )

    if starting_projection == ProjectionMode.TSNE:
        starting_projection_name = "TSNE"
    elif starting_projection == ProjectionMode.PCA:
        starting_projection_name = "PCA"
    else:
        starting_projection_name = "RANDOM"
    # calculate stress
    stress = nfs.score(projection)
    value = [
        n_dimension,
        learning_rate,
        decay,
        max_iteration,
        starting_projection_name,
        dataname,
        stress,
        fixed_axis if fixed_axis is not None else "None",
        "NO" if fixed_axis in X_excp else "YES",
        "NO" if scaler is False else str(scaler),
        str(z_axis_moving_range),
    ]

    if fig_path is not None:
        if not os.path.exists(fig_path):
            os.mkdir(fig_path)

        # create a figure and save it
        if n_dimension == 2:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=projection[:, 0],
                    y=projection[:, 1],
                    mode="markers",
                    marker=go.scatter.Marker(
                        size=5,
                        color=projection[:, 1],
                        opacity=0.6,
                        colorscale="Viridis",
                    ),
                )
            )
        else:
            fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=projection[:, 0],
                        y=projection[:, 1],
                        z=projection[:, 2],
                        mode="markers",
                        marker=dict(
                            size=5,
                            color=projection[
                                :, 2
                            ],  # set color to an array/list of desired values
                            colorscale="Viridis",  # choose a colorscale
                            opacity=0.8,
                        ),
                    )
                ]
            )
        fig_name = (
            str(n_dimension)
            + "d_"
            + str(learning_rate)
            + "_"
            + str(decay)
            + "_"
            + str(max_iteration)
            + "_"
            + starting_projection_name
            + "_"
            + dataname
            + "_"
            + str(fixed_axis)
            + "_"
            + str(X_excp)
            + "_"
            + str(scaler)
            + "_"
            + str(z_axis_moving_range)
        )

        fig.write_image(fig_path + "/" + fig_name + ".png", engine="kaleido")

        pd.DataFrame(projection).to_csv(
            fig_path + "/" + fig_name + ".csv", header=False, index=False
        )
    print(value)
    return value


def loop_script(
    dataset,
    dataname,
    scaler_list,
    nd_list,
    lr_list,
    dc_list,
    maxi_list,
    start_project,
    fig_path,
    fixed_axis,
    X_excp=None,
    num_process_axis=None,
    verbose=False,
    z_axis_moving_range_list=[(0, 0)],
):
    ret = []
    for nd in nd_list:
        for lr in lr_list:
            for dc in dc_list:
                for max_iter in maxi_list:
                    for sp in start_project:
                        for scaler in scaler_list:
                            for _fixed_axis in fixed_axis:
                                for z_axis_moving_range in z_axis_moving_range_list:
                                    r = datacollecting(
                                        n_dimension=nd,
                                        learning_rate=lr,
                                        decay=dc,
                                        max_iteration=max_iter,
                                        starting_projection=sp,
                                        dataset=dataset,
                                        dataname=dataname,
                                        fig_path=fig_path,
                                        scaler=scaler,
                                        fixed_axis=_fixed_axis,
                                        X_excp=X_excp,
                                        num_process_axis=num_process_axis,
                                        z_axis_moving_range=z_axis_moving_range,
                                        verbose=verbose,
                                    )
                                    ret.append(r)
    return ret


if __name__ == "__main__":
    ##########################################
    output_file = "data_collecting.csv"
    value_title = [
        "n_dimension",
        "learning_rate",
        "decay",
        "max_iteration",
        "starting_projection",
        "dataset",
        "stress",
        "fixed_axis",
        "excluded_fixed_axis_from_dataset",
        "scaler",
        "z_axis_moving_range",
    ]
    pd.DataFrame(columns=value_title).to_csv(output_file, header=True, index=False)

    ## for testing
    # scaler_list = [None]
    # nd_list = [2]
    # lr_list = [0.1]
    # dc_list = [0.88]
    # maxi_list = [300]
    # start_project = [ProjectionMode.TSNE]

    # scaler_list = [None, (0, 1), (0, 10), (0, 100), (0, 500), (0, 1000)]
    # nd_list = [2, 3]
    # lr_list = [0.1, 0.3, 0.5]
    # dc_list = [0.88, 0.93, 0.95, 0.97]
    # maxi_list = [300, 500, 900]
    # z_axis_moving_range_list = [(0, 0), (-0.2, 0.2), (-0.5, 0.5)]
    scaler_list = [(0, 1)]
    number_dimension_list = [2]
    learning_rate_list = [0.3]
    decay_list = [0.88]
    max_iteration_list = [300]
    z_axis_moving_range_list = [(0, 0), (-0.2, 0.2), (-0.5, 0.5)]
    # ^^^ only works fine for scaler range (0, 1)
    # (otherwise the range is too large so that figure will not be very different)
    # start_project = [ProjectionMode.TSNE, ProjectionMode.PCA, ProjectionMode.RANDOM]
    start_project = [ProjectionMode.RANDOM]

    ##########################################　iris

    dataset = datasets.load_iris().data
    label = datasets.load_iris().target.reshape(len(dataset), 1)
    dataset = np.concatenate((dataset, label), axis=1)
    dataname = "iris"
    fig_path = "images/" + dataname + "_fixed_axis_images"

    r = loop_script(
        dataset=dataset,
        dataname=dataname,
        scaler_list=scaler_list,
        nd_list=number_dimension_list,
        lr_list=learning_rate_list,
        dc_list=decay_list,
        maxi_list=max_iteration_list,
        start_project=start_project,
        fig_path=fig_path,
        fixed_axis=[None] + list(range(dataset.shape[1])),
        X_excp=[dataset.shape[1] - 1],
        num_process_axis=None,
        verbose=False,
        z_axis_moving_range_list=z_axis_moving_range_list,
    )
    print("END DATASET")
    pd.DataFrame(r, columns=value_title).to_csv(
        output_file, mode="a", header=False, index=False, sep=";"
    )

    # ##########################################　mammals
    dataset = np.loadtxt("./examples/datasets/mammals.data", delimiter=",")
    dataname = "mammals"
    fig_path = "images/" + dataname + "_fixed_axis_images"
    r = loop_script(
        dataset=dataset,
        dataname=dataname,
        scaler_list=scaler_list,
        nd_list=number_dimension_list,
        lr_list=learning_rate_list,
        dc_list=decay_list,
        maxi_list=max_iteration_list,
        start_project=start_project,
        fig_path=fig_path,
        fixed_axis=[None] + list(range(dataset.shape[1])),
        X_excp=[dataset.shape[1] - 1],
        num_process_axis=None,
        z_axis_moving_range_list=z_axis_moving_range_list,
    )
    print("END DATASET")
    pd.DataFrame(r, columns=value_title).to_csv(
        output_file, mode="a", header=False, index=False
    )

    # ##########################################　whr2019_label_included

    raw_data = pandas.read_csv("./examples/datasets/whr2019.csv", sep=",")
    dataset = raw_data.values
    dataname = "whr2019_included"
    fig_path = "images/" + dataname + "_fixed_axis_images"
    r = loop_script(
        dataset=dataset,
        dataname=dataname,
        scaler_list=scaler_list,
        nd_list=number_dimension_list,
        lr_list=learning_rate_list,
        dc_list=decay_list,
        maxi_list=max_iteration_list,
        start_project=start_project,
        fig_path=fig_path,
        fixed_axis=list(range(dataset.shape[1])),
        X_excp=[],
        num_process_axis=None,
        z_axis_moving_range_list=z_axis_moving_range_list,
    )
    print("END DATASET")
    pd.DataFrame(r, columns=value_title).to_csv(
        output_file, mode="a", header=False, index=False
    )

    # ##########################################　whr2019

    raw_data = pandas.read_csv("./examples/datasets/whr2019.csv", sep=",")
    dataset = raw_data.values
    dataname = "whr2019"
    fig_path = "images/" + dataname + "_fixed_axis_images"
    r = loop_script(
        dataset=dataset,
        dataname=dataname,
        scaler_list=scaler_list,
        nd_list=number_dimension_list,
        lr_list=learning_rate_list,
        dc_list=decay_list,
        maxi_list=max_iteration_list,
        start_project=start_project,
        fig_path=fig_path,
        fixed_axis=[None] + list(range(dataset.shape[1])),
        X_excp=[0],
        num_process_axis=None,
        z_axis_moving_range_list=z_axis_moving_range_list,
    )
    print("END DATASET")
    pd.DataFrame(r, columns=value_title).to_csv(
        output_file, mode="a", header=False, index=False
    )

    # ##########################################　image_segmentation

    raw_data = pandas.read_csv("./examples/datasets/segmentation-normcols.csv", sep=",")
    dataset = raw_data.values
    dataname = "segmentation"
    fig_path = "images/" + dataname + "_fixed_axis_images"
    r = loop_script(
        dataset=dataset,
        dataname=dataname,
        scaler_list=scaler_list,
        nd_list=number_dimension_list,
        lr_list=learning_rate_list,
        dc_list=decay_list,
        maxi_list=max_iteration_list,
        start_project=start_project,
        fig_path=fig_path,
        fixed_axis=[None] + list(range(dataset.shape[1])),
        X_excp=[0, dataset.shape[1] - 1],
        num_process_axis=None,
        z_axis_moving_range_list=z_axis_moving_range_list,
    )
    print("END DATASET")
    pd.DataFrame(r, columns=value_title).to_csv(
        output_file, mode="a", header=False, index=False
    )

    # ##########################################　cancer

    dataset = datasets.load_breast_cancer().data
    label = datasets.load_breast_cancer().target.reshape(len(dataset), 1)
    dataset = np.concatenate((dataset, label), axis=1)
    dataname = "cancer"
    fig_path = "images/" + dataname + "_fixed_axis_images"
    r = loop_script(
        dataset=dataset,
        dataname=dataname,
        scaler_list=scaler_list,
        nd_list=number_dimension_list,
        lr_list=learning_rate_list,
        dc_list=decay_list,
        maxi_list=max_iteration_list,
        start_project=start_project,
        fig_path=fig_path,
        fixed_axis=[None] + list(range(dataset.shape[1])),
        X_excp=[dataset.shape[1] - 1],
        num_process_axis=None,
        z_axis_moving_range_list=z_axis_moving_range_list,
    )
    print("END DATASET")
    pd.DataFrame(r, columns=value_title).to_csv(
        output_file, mode="a", header=False, index=False
    )

    # ##########################################　boston

    dataset = datasets.load_boston().data
    label = datasets.load_boston().target.reshape(len(dataset), 1)
    dataset = np.concatenate((dataset, label), axis=1)
    dataname = "boston"
    fig_path = "images/" + dataname + "_fixed_axis_images"
    r = loop_script(
        dataset=dataset,
        dataname=dataname,
        scaler_list=scaler_list,
        nd_list=number_dimension_list,
        lr_list=learning_rate_list,
        dc_list=decay_list,
        maxi_list=max_iteration_list,
        start_project=start_project,
        fig_path=fig_path,
        fixed_axis=[None] + list(range(dataset.shape[1])),
        X_excp=[dataset.shape[1] - 1],
        num_process_axis=None,
        z_axis_moving_range_list=z_axis_moving_range_list,
    )
    print("END DATASET")
    pd.DataFrame(r, columns=value_title).to_csv(
        output_file, mode="a", header=False, index=False
    )

    # ##########################################　diabetes

    dataset = datasets.load_diabetes().data
    label = datasets.load_diabetes().target.reshape(len(dataset), 1)
    dataset = np.concatenate((dataset, label), axis=1)
    dataname = "diabetes"
    fig_path = "images/" + dataname + "_fixed_axis_images"
    r = loop_script(
        dataset=dataset,
        dataname=dataname,
        scaler_list=scaler_list,
        nd_list=number_dimension_list,
        lr_list=learning_rate_list,
        dc_list=decay_list,
        maxi_list=max_iteration_list,
        start_project=start_project,
        fig_path=fig_path,
        fixed_axis=[None] + list(range(dataset.shape[1])),
        X_excp=[dataset.shape[1] - 1],
        num_process_axis=None,
        z_axis_moving_range_list=z_axis_moving_range_list,
    )
    print("END DATASET")
    pd.DataFrame(r, columns=value_title).to_csv(
        output_file, mode="a", header=False, index=False
    )
