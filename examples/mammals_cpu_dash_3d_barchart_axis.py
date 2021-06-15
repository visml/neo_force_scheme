from datetime import timedelta
from timeit import default_timer as timer

import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
# Read data from a csv
from dash import dash
from dash.dependencies import Output, Input
from dash.exceptions import PreventUpdate
from sklearn import datasets
from neo_force_scheme import NeoForceScheme, ProjectionMode, kruskal_stress

data = np.loadtxt("./mammals.data", delimiter=",")
#data = datasets.load_breast_cancer().data
original_len = len(data)

# read the distance matrix
nfs = NeoForceScheme(verbose=True, learning_rate0=0.2, decay=0.95)
data = nfs.create_maxmin_point(data)

n, d = data.shape[0], data.shape[1]

x = data[:, range(d - 1)]
t = data[:, d - 1]

# execute force scheme
start = timer()
# projection = nfs.fit_transform(X=x, starting_projection_mode=ProjectionMode.RANDOM, random_state=1, n_dimension=3)
# projection = nfs.fit_transform(X=x, Xd=data, starting_projection_mode=ProjectionMode.TSNE, random_state=1, n_dimension=3)
projection = nfs.fit_transform(X=x, Xd=data, starting_projection_mode=ProjectionMode.PCA, random_state=1, n_dimension=3)

error = nfs.projection_error_
end = timer()

print(f'ForceScheme took {timedelta(seconds=end - start)} to execute with error {error}')

# calculate stress
stress = kruskal_stress(nfs.embedding_, projection)
print('Kruskal stress {0}:'.format(stress))

# save projection
# np.savetxt(input_file + "_projection.txt", projection, delimiter=" ", fmt="%s")

# show projection

# Default parameters which are used when `layout.scene.camera` is not provided
camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=1.25, y=1.25, z=1.25)
)

# mammal data do not have variable names, so here create the names manually
variable_name = []
for number in range(int((len(projection) - original_len) / 2)):
    variable_name.append('variable ' + str(number))

x_axis, y_axis = nfs.calculate_height(projection, original_len, eye=dict(x=1.25, y=1.25, z=1.25))
x_heights, y_heights = nfs.sort_height(variable_name, x_axis, y_axis)

variable_name_main_figure = []
for number in range(int((len(projection) - original_len) / 2)):
    variable_name_main_figure.append('variable ' + str(number))
    variable_name_main_figure.append('variable ' + str(number))

"""
# main figure
fig1 = go.Figure(
    data=[go.Scatter3d(x=projection[:original_len, 0],
                       y=projection[:original_len, 1],
                       z=projection[:original_len, 2],
                       mode='markers',
                       marker=dict(
                           size=12,
                           color=projection[:, 2],  # set color to an array/list of desired values
                           colorscale='Viridis',  # choose a colorscale
                           opacity=0.8
                       )
                       )])


fig1.add_trace(go.Scatter3d(x=projection[:original_len, 0],
                       y=projection[:original_len, 1],
                       z=projection[:original_len, 2],
                       mode='markers',
                       marker=dict(
                           size=12,
                           color=projection[:, 2],  # set color to an array/list of desired values
                           colorscale='Viridis',  # choose a colorscale
                           opacity=0.8
                       )))
                       """

fig1 = px.line_3d(x=projection[original_len:, 0],
                       y=projection[original_len:, 1],
                       z=projection[original_len:, 2],
                       color= variable_name_main_figure
                                     )

fig1.update_layout(
    title='Mammals data',
    width=400, height=400,
    margin=dict(t=40, r=0, l=20, b=20)
)

fig2 = go.Figure(data=[go.Bar(x=x_heights[:10, 0], y=list(map(float, x_heights[:10, 1])))])

fig2.update_layout(
    title='x axis legend',
    width=200, height=200,
    margin=dict(t=40, r=0, l=20, b=20)
)

fig3 = go.Figure(data=[go.Bar(x=y_heights[:10, 0], y=list(map(float, y_heights[:10, 1])))])

fig3.update_layout(
    title='y axis legend',
    width=200, height=200,
    margin=dict(t=40, r=0, l=20, b=20)
)
# manually set the viewpoint to x-z plane

name = 'force scheme 3d visualization'

fig1.update_layout(scene_camera=camera, title=name)

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div([
    html.H1('force_3d', id='force_3d'),
    dbc.Row(
        [
            dbc.Col(dcc.Graph(figure=fig3, id='y_legend'), width=4),
            dbc.Col(dcc.Graph(figure=fig1, id='main_fig'), width=8),
        ], justify="start"
    ),
    dbc.Row(
        [
            dbc.Col(dcc.Graph(figure=fig2, id='x_legend'), width=4)
        ], justify="center",
    )
])
"""
    html.H1('force_3d', id='force_3d'),
    dcc.Graph(figure=fig1, id='main_fig'),
    dcc.Graph(figure=fig2, id='x_legend'),
    dcc.Graph(figure=fig3, id='y_legend'),
])
"""

@app.callback(
    Output(component_id='x_legend', component_property='figure'),
    Output(component_id='y_legend', component_property='figure'),
    Input(component_id='main_fig', component_property='relayoutData')
)
def update_output_div(input_value):
    if input_value is None:
        raise PreventUpdate
    #print(input_value['scene.camera'].get('eye'))
    x_axis_new, y_axis_new = nfs.calculate_height(projection, original_len, eye=input_value['scene.camera'].get('eye'))
    x_heights_new, y_heights_new = nfs.sort_height(variable_name, x_axis_new, y_axis_new)
    fig2_new = go.Figure(data=[go.Bar(x=x_heights_new[:10, 0], y=list(map(float, x_heights_new[:10, 1])))])
    fig3_new = go.Figure(data=[go.Bar(x=y_heights_new[:10, 0], y=list(map(float, y_heights_new[:10, 1])))])

    fig2_new.update_layout(
        title='x axis legend',
        width=200, height=200,
        margin=dict(t=40, r=0, l=20, b=20)
    )
    fig3_new.update_layout(
        title='y axis legend',
        width=200, height=200,
        margin=dict(t=40, r=0, l=20, b=20)
    )
    return fig2_new, fig3_new

"""
used for debug. update for random heights
def update_output_div(input_value):
    if input_value is None:
        raise PreventUpdate

    #print(input_value['scene.camera'].get('eye'))
    x_axis_new, y_axis_new = nfs.calculate_height(projection, original_len, eye=input_value['scene.camera'].get('eye'))
    x_heights_new, y_heights_new = nfs.sort_height(variable_name, x_axis_new, y_axis_new)
    np.random.shuffle(x_heights_new)
    np.random.shuffle(y_heights_new)
    fig2_new = go.Figure(data=[go.Bar(x=x_heights_new[:10, 0], y=list(map(float, x_heights_new[:10, 1])))])
    fig3_new = go.Figure(data=[go.Bar(x=y_heights_new[:10, 0], y=list(map(float, y_heights_new[:10, 1])))])

    fig2_new.update_layout(
        title='x axis legend',
        width=400, height=400,
        margin=dict(t=40, r=0, l=20, b=20)
    )
    fig3_new.update_layout(
        title='y axis legend',
        width=400, height=400,
        margin=dict(t=40, r=0, l=20, b=20)
    )
    print(print(input_value['scene.camera']))
    print('updated!')
        # create a new figure
    return fig2_new, fig3_new
"""

app.run_server(debug=True, use_reloader=True)  # Turn off reloader if inside Jupyter
