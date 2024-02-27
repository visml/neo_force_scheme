# %%
import os, sys

from sklearn import datasets

os.chdir("/Users/christinoleo/Projects/neo_force_scheme")
import pandas as pd
import numpy as np
from examples.benchmarking import loop_script

from neo_force_scheme.neo_force_scheme import ProjectionMode

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

scaler_list = [None, (0, 1), (0, 10), (0, 100), (0, 500), (0, 1000)]
nd_list = [2, 3]
lr_list = [0.1, 0.3, 0.5]
dc_list = [0.88, 0.93, 0.95, 0.97]
maxi_list = [300, 500, 900]
z_axis_moving_range_list = [(0, 0), (-0.2, 0.2), (-0.5, 0.5)]
# ^^^ only works fine for scaler range (0, 1)
# (otherwise the range is too large so that figure will not be very different)
start_project = [ProjectionMode.TSNE, ProjectionMode.PCA, ProjectionMode.RANDOM]

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
    nd_list=nd_list,
    lr_list=lr_list,
    dc_list=dc_list,
    maxi_list=maxi_list,
    start_project=start_project,
    fig_path=fig_path,
    fixed_axis=[None, -1],
    X_excp=[-1],
    num_process_axis=None,
    verbose=False,
    z_axis_moving_range_list=z_axis_moving_range_list,
)
print("END DATASET")
pd.DataFrame(r, columns=value_title).to_csv(
    output_file, mode="a", header=False, index=False
)
# %%

##########################################　mammals
dataset = np.loadtxt("./datasets/mammals.data", delimiter=",")
dataname = "mammals"
fig_path = "images/" + dataname + "_fixed_axis_images"
r = loop_script(
    dataset=dataset,
    dataname=dataname,
    scaler_list=scaler_list,
    nd_list=nd_list,
    lr_list=lr_list,
    dc_list=dc_list,
    maxi_list=maxi_list,
    start_project=start_project,
    fig_path=fig_path,
    fixed_axis=[None, -1],
    X_excp=[-1],
    num_process_axis=None,
)
print("END DATASET")
pd.DataFrame(r, columns=value_title).to_csv(
    output_file, mode="a", header=False, index=False
)

##########################################　whr2019_label_included

raw_data = pandas.read_csv("./datasets/whr2019.csv", sep=",")
dataset = raw_data.values
dataname = "whr2019_included"
fig_path = "images/" + dataname + "_fixed_axis_images"
r = loop_script(
    dataset=dataset,
    dataname=dataname,
    scaler_list=scaler_list,
    nd_list=nd_list,
    lr_list=lr_list,
    dc_list=dc_list,
    maxi_list=maxi_list,
    start_project=start_project,
    fig_path=fig_path,
    fixed_axis=[0],
    X_excp=[],
    num_process_axis=None,
)
print("END DATASET")
pd.DataFrame(r, columns=value_title).to_csv(
    output_file, mode="a", header=False, index=False
)

##########################################　whr2019

raw_data = pandas.read_csv("./datasets/whr2019.csv", sep=",")
dataset = raw_data.values
dataname = "whr2019"
fig_path = "images/" + dataname + "_fixed_axis_images"
r = loop_script(
    dataset=dataset,
    dataname=dataname,
    scaler_list=scaler_list,
    nd_list=nd_list,
    lr_list=lr_list,
    dc_list=dc_list,
    maxi_list=maxi_list,
    start_project=start_project,
    fig_path=fig_path,
    fixed_axis=[None, 0],
    X_excp=[0],
    num_process_axis=None,
)
print("END DATASET")
pd.DataFrame(r, columns=value_title).to_csv(
    output_file, mode="a", header=False, index=False
)

##########################################　image_segmentation

raw_data = pandas.read_csv("./datasets/segmentation-normcols.csv", sep=",")
dataset = raw_data.values
dataname = "segmentation"
fig_path = "images/" + dataname + "_fixed_axis_images"
r = loop_script(
    dataset=dataset,
    dataname=dataname,
    scaler_list=scaler_list,
    nd_list=nd_list,
    lr_list=lr_list,
    dc_list=dc_list,
    maxi_list=maxi_list,
    start_project=start_project,
    fig_path=fig_path,
    fixed_axis=[None, -1],
    X_excp=[0, -1],
    num_process_axis=None,
)
print("END DATASET")
pd.DataFrame(r, columns=value_title).to_csv(
    output_file, mode="a", header=False, index=False
)

##########################################　cancer

dataset = datasets.load_breast_cancer().data
label = datasets.load_breast_cancer().target.reshape(len(dataset), 1)
dataset = np.concatenate((dataset, label), axis=1)
dataname = "cancer"
fig_path = "images/" + dataname + "_fixed_axis_images"
r = loop_script(
    dataset=dataset,
    dataname=dataname,
    scaler_list=scaler_list,
    nd_list=nd_list,
    lr_list=lr_list,
    dc_list=dc_list,
    maxi_list=maxi_list,
    start_project=start_project,
    fig_path=fig_path,
    fixed_axis=[None, -1],
    X_excp=[-1],
    num_process_axis=None,
)
print("END DATASET")
pd.DataFrame(r, columns=value_title).to_csv(
    output_file, mode="a", header=False, index=False
)

##########################################　boston

dataset = datasets.load_boston().data
label = datasets.load_boston().target.reshape(len(dataset), 1)
dataset = np.concatenate((dataset, label), axis=1)
dataname = "boston"
fig_path = "images/" + dataname + "_fixed_axis_images"
r = loop_script(
    dataset=dataset,
    dataname=dataname,
    scaler_list=scaler_list,
    nd_list=nd_list,
    lr_list=lr_list,
    dc_list=dc_list,
    maxi_list=maxi_list,
    start_project=start_project,
    fig_path=fig_path,
    fixed_axis=[None, -1],
    X_excp=[-1],
    num_process_axis=None,
)
print("END DATASET")
pd.DataFrame(r, columns=value_title).to_csv(
    output_file, mode="a", header=False, index=False
)

##########################################　diabetes

dataset = datasets.load_diabetes().data
label = datasets.load_diabetes().target.reshape(len(dataset), 1)
dataset = np.concatenate((dataset, label), axis=1)
dataname = "diabetes"
fig_path = "images/" + dataname + "_fixed_axis_images"
r = loop_script(
    dataset=dataset,
    dataname=dataname,
    scaler_list=scaler_list,
    nd_list=nd_list,
    lr_list=lr_list,
    dc_list=dc_list,
    maxi_list=maxi_list,
    start_project=start_project,
    fig_path=fig_path,
    fixed_axis=[None, -1],
    X_excp=[-1],
    num_process_axis=None,
)
print("END DATASET")
pd.DataFrame(r, columns=value_title).to_csv(
    output_file, mode="a", header=False, index=False
)
