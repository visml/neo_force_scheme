import os

import numpy as np
# Read data from a csv
import pandas
import plotly.graph_objects as go
import xlrd
import xlwt
from sklearn import datasets
from xlutils.copy import copy

from neo_force_scheme import NeoForceScheme, ProjectionMode


def datacollecting(n_dimension,
                   learning_rate,
                   decay,
                   max_iteration,
                   starting_projection,
                   dataset,
                   dataname,
                   fig_path,
                   scaler=False,
                   fixed_axis=None,
                   X_excp=None,
                   Xd_excp=None,
                   num_process_axis=None):
    # read the distance matrix
    nfs = NeoForceScheme(verbose=False, learning_rate0=learning_rate, decay=decay, max_it=max_iteration)

    if num_process_axis is not None:
        dataset = nfs.non_numeric_processor(dataset, num_process_axis)

    projection = nfs.fit_transform(data=dataset,
                                   starting_projection_mode=starting_projection,
                                   random_state=1,
                                   n_dimension=n_dimension,
                                   fixed_axis=fixed_axis,
                                   X_exception_axes=X_excp,
                                   Xd_exception_axes=Xd_excp,
                                   scaler=scaler)

    if starting_projection == ProjectionMode.TSNE:
        starting_projection_name = 'TSNE'
    elif starting_projection == ProjectionMode.PCA:
        starting_projection_name = 'PCA'
    else:
        starting_projection_name = 'RANDOM'
    # calculate stress
    stress = nfs.score(projection)
    value = [[n_dimension,
              learning_rate,
              decay,
              max_iteration,
              starting_projection_name,
              dataname,
              stress,
              fixed_axis if fixed_axis is not None else 'None',
              'NO' if fixed_axis in X_excp else 'YES',
              'NO' if scaler is False else str(scaler)]]

    if not os.path.exists(fig_path):
        os.mkdir(fig_path)

    # create a figure and save it
    if n_dimension == 2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=projection[:, 0],
            y=projection[:, 1],
            mode="markers",
            marker=go.scatter.Marker(
                size=5,
                color=projection[:, 1],
                opacity=0.6,
                colorscale="Viridis"
            )))
    else:
        fig = go.Figure(
            data=[go.Scatter3d(x=projection[:, 0],
                               y=projection[:, 1],
                               z=projection[:, 2],
                               mode='markers',
                               marker=dict(
                                   size=5,
                                   color=projection[:, 2],  # set color to an array/list of desired values
                                   colorscale='Viridis',  # choose a colorscale
                                   opacity=0.8
                               )
                               )])
    fig_name = str(n_dimension) + "d_" + str(learning_rate) + "_" + str(decay) + \
               "_" + str(max_iteration) + "_" + starting_projection_name + "_" + \
               dataname + "_" + str(fixed_axis) + "_" + str(X_excp) + "_" + str(scaler)

    fig_path = fig_path + "/" + fig_name + ".png"
    fig.write_image(fig_path, engine="kaleido")
    print(value)
    return value


def write_excel_xls(path, sheet_name, values):
    index = len(values)
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet(sheet_name)
    for i in range(0, index):
        for j in range(0, len(values[i])):
            sheet.write(i, j, values[i][j])
    workbook.save(path)


def write_sheet_xls(path, sheet_name, value):
    index = len(value)
    rb = xlrd.open_workbook(path, formatting_info=True)
    workbook = copy(rb)
    sheet = workbook.add_sheet(sheet_name)
    for i in range(0, index):
        for j in range(0, len(value[i])):
            sheet.write(i, j, value[i][j])
    workbook.save(path)


def write_excel_xls_append(path, value):
    index = len(value)
    workbook = xlrd.open_workbook(path)
    sheets = workbook.sheet_names()
    worksheet = workbook.sheet_by_name(sheets[-1])
    rows_old = worksheet.nrows
    new_workbook = copy(workbook)
    new_worksheet = new_workbook.get_sheet(-1)
    for i in range(0, index):
        for j in range(0, len(value[i])):
            new_worksheet.write(i + rows_old, j, value[i][j])
    new_workbook.save(path)


def read_excel_xls(path):
    workbook = xlrd.open_workbook(path)
    sheets = workbook.sheet_names()
    worksheet = workbook.sheet_by_name(sheets[0])
    for i in range(0, worksheet.nrows):
        for j in range(0, worksheet.ncols):
            print(worksheet.cell_value(i, j), "\t", end="")
        print()


def loop_script(dataset,
                dataname,
                scaler_list,
                nd_list,
                lr_list,
                dc_list,
                maxi_list,
                start_project,
                fig_path,
                fixed_axis=None,
                X_excp=None,
                Xd_excp=None,
                num_process_axis=None):
    for nd in nd_list:
        for lr in lr_list:
            for dc in dc_list:
                for max_iter in maxi_list:
                    for sp in start_project:
                        for scaler in scaler_list:
                            value = datacollecting(n_dimension=nd,
                                                   learning_rate=lr,
                                                   decay=dc,
                                                   max_iteration=max_iter,
                                                   starting_projection=sp,
                                                   dataset=dataset,
                                                   dataname=dataname,
                                                   fig_path=fig_path,
                                                   scaler=scaler,
                                                   fixed_axis=fixed_axis,
                                                   X_excp=X_excp,
                                                   Xd_excp=Xd_excp,
                                                   num_process_axis=num_process_axis)
                            write_excel_xls_append(book_name_xls, value)


##########################################
book_name_xls = 'data_collecting.xls'
value_title = [['n_dimension', 'learning_rate', 'decay', 'max_iteration', 'starting_projection', 'dataset', 'stress']]

scaler_list = [False, (0, 1), (0, 10), (0, 100), (0, 500), (0, 1000)]
nd_list = [2, 3]
lr_list = [0.1, 0.3, 0.5]
dc_list = [0.88, 0.93, 0.95, 0.97]
maxi_list = [300, 500, 900]
start_project = [ProjectionMode.TSNE, ProjectionMode.PCA, ProjectionMode.RANDOM]

##########################################　mammals
dataset = np.loadtxt('./mammals.data', delimiter=",")

dataname = 'mammals'

# sheet_name_xls = 'test_run_data_'+dataname
fig_path = 'images/' + dataname + '_fixed_axis_images'

# write_sheet_xls(book_name_xls, sheet_name_xls, value_title)
loop_script(dataset=dataset,
            dataname=dataname,
            scaler_list=scaler_list,
            nd_list=nd_list,
            lr_list=lr_list,
            dc_list=dc_list,
            maxi_list=maxi_list,
            start_project=start_project,
            fig_path=fig_path,
            fixed_axis=-1,
            X_excp=[-1],
            Xd_excp=[-1],
            num_process_axis=None)

read_excel_xls(book_name_xls)

##########################################　whr2019_label_included

raw_data = pandas.read_csv(
    "./whr2019.csv",
    sep=",")
dataset = raw_data.values

dataname = 'whr2019_included'

# sheet_name_xls = 'test_run_data_'+dataname
fig_path = 'images/' + dataname + '_fixed_axis_images'

# write_sheet_xls(book_name_xls, sheet_name_xls, value_title)
loop_script(dataset=dataset,
            dataname=dataname,
            scaler_list=scaler_list,
            nd_list=nd_list,
            lr_list=lr_list,
            dc_list=dc_list,
            maxi_list=maxi_list,
            start_project=start_project,
            fig_path=fig_path,
            fixed_axis=0,
            X_excp=[],
            Xd_excp=[],
            num_process_axis=None)

read_excel_xls(book_name_xls)

##########################################　whr2019


raw_data = pandas.read_csv(
    "./whr2019.csv",
    sep=",")
dataset = raw_data.values

dataname = 'whr2019'

# sheet_name_xls = 'test_run_data_'+dataname
fig_path = 'images/' + dataname + '_fixed_axis_images'

# write_sheet_xls(book_name_xls, sheet_name_xls, value_title)
loop_script(dataset=dataset,
            dataname=dataname,
            scaler_list=scaler_list,
            nd_list=nd_list,
            lr_list=lr_list,
            dc_list=dc_list,
            maxi_list=maxi_list,
            start_project=start_project,
            fig_path=fig_path,
            fixed_axis=0,
            X_excp=[0],
            Xd_excp=[0],
            num_process_axis=None)

read_excel_xls(book_name_xls)

##########################################　image_segmentation

raw_data = pandas.read_csv(
    "./segmentation-normcols.csv",
    sep=",")
dataset = raw_data.values

dataname = 'segmentation'

# sheet_name_xls = 'test_run_data_'+dataname
fig_path = 'images/' + dataname + '_fixed_axis_images'

# write_sheet_xls(book_name_xls, sheet_name_xls, value_title)
loop_script(dataset=dataset,
            dataname=dataname,
            scaler_list=scaler_list,
            nd_list=nd_list,
            lr_list=lr_list,
            dc_list=dc_list,
            maxi_list=maxi_list,
            start_project=start_project,
            fig_path=fig_path,
            fixed_axis=-1,
            X_excp=[0, -1],
            Xd_excp=[0, -1],
            num_process_axis=None)

read_excel_xls(book_name_xls)

##########################################　cancer

dataset = datasets.load_breast_cancer().data
label = datasets.load_breast_cancer().target.reshape(len(dataset), 1)
dataset = np.concatenate((dataset, label), axis=1)

dataname = 'cancer'

# sheet_name_xls = 'test_run_data_'+dataname
fig_path = 'images/' + dataname + '_fixed_axis_images'

# write_sheet_xls(book_name_xls, sheet_name_xls, value_title)
loop_script(dataset=dataset,
            dataname=dataname,
            scaler_list=scaler_list,
            nd_list=nd_list,
            lr_list=lr_list,
            dc_list=dc_list,
            maxi_list=maxi_list,
            start_project=start_project,
            fig_path=fig_path,
            fixed_axis=-1,
            X_excp=[-1],
            Xd_excp=[-1],
            num_process_axis=None)

read_excel_xls(book_name_xls)

##########################################　iris

dataset = datasets.load_iris().data
label = datasets.load_iris().target.reshape(len(dataset), 1)
dataset = np.concatenate((dataset, label), axis=1)

dataname = 'iris'

# sheet_name_xls = 'test_run_data_'+dataname
fig_path = 'images/' + dataname + '_fixed_axis_images'

# write_sheet_xls(book_name_xls, sheet_name_xls, value_title)
loop_script(dataset=dataset,
            dataname=dataname,
            scaler_list=scaler_list,
            nd_list=nd_list,
            lr_list=lr_list,
            dc_list=dc_list,
            maxi_list=maxi_list,
            start_project=start_project,
            fig_path=fig_path,
            fixed_axis=-1,
            X_excp=[-1],
            Xd_excp=[-1],
            num_process_axis=None)

read_excel_xls(book_name_xls)

##########################################　boston

dataset = datasets.load_boston().data
label = datasets.load_boston().target.reshape(len(dataset), 1)
dataset = np.concatenate((dataset, label), axis=1)

dataname = 'boston'

# sheet_name_xls = 'test_run_data_'+dataname
fig_path = 'images/' + dataname + '_fixed_axis_images'

# write_sheet_xls(book_name_xls, sheet_name_xls, value_title)
loop_script(dataset=dataset,
            dataname=dataname,
            scaler_list=scaler_list,
            nd_list=nd_list,
            lr_list=lr_list,
            dc_list=dc_list,
            maxi_list=maxi_list,
            start_project=start_project,
            fig_path=fig_path,
            fixed_axis=-1,
            X_excp=[-1],
            Xd_excp=[-1],
            num_process_axis=None)

read_excel_xls(book_name_xls)

##########################################　diabetes

dataset = datasets.load_diabetes().data
label = datasets.load_diabetes().target.reshape(len(dataset), 1)
dataset = np.concatenate((dataset, label), axis=1)

dataname = 'diabetes'

# sheet_name_xls = 'test_run_data_'+dataname
fig_path = 'images/' + dataname + '_fixed_axis_images'

# write_sheet_xls(book_name_xls, sheet_name_xls, value_title)
loop_script(dataset=dataset,
            dataname=dataname,
            scaler_list=scaler_list,
            nd_list=nd_list,
            lr_list=lr_list,
            dc_list=dc_list,
            maxi_list=maxi_list,
            start_project=start_project,
            fig_path=fig_path,
            fixed_axis=-1,
            X_excp=[-1],
            Xd_excp=[-1],
            num_process_axis=None)

read_excel_xls(book_name_xls)
