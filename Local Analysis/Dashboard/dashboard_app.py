################## DATA DISTRIBUTIONS ##################
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from ComplexNetwork import ComplexNetwork
from divergence import shannon_divergence

dataset = "CIFAR10"
architecture = 'cnn'
input_shape = ((28,28,1) if dataset=="MNIST" else (32,32,3))
output_size = 10
file_ = "./weights/{}/{}_small_cnn_nlayers-4_init-variance-scaling-normal-fanin_support-0.05_seed-7_realaccuracy-0.4609_binaccuracy-0.4500.npy".format(dataset, dataset)
num_layers = 4
num_conv_layers = 2
paddings, strides = (0,0), (1,1)
saved_accuracy = '0.4609'
saved_model_size = 'small'
saved_imag_path = "./results/images/{}/".format(dataset)
img_format = '.png'
W = np.load(file_, allow_pickle=True)  # load parameters
CNet = ComplexNetwork(architecture, num_layers, num_conv_layers, W, input_shape, output_size, strides=strides, paddings=paddings, flatten=False)

################## DASHBOARD ##################
# ##import
import plotly.graph_objects as go       # plytly graph objects
from plotly.subplots import make_subplots       # plotly.subplots make subplots
import pandas as pd             # pandas
import numpy as np          # numpy
import dash         # Dash
import dash_core_components as dcc      # dash core components dcc
import dash_html_components as html      # dash core component html
import plotly.express as px
import requests         # requests
from datetime import datetime , timedelta
from dash.dependencies import Output , Input
##

## data
ts_cases_path = "Data/time_series_cases (1).csv"
ts_deaths_path = "Data/time_series_deaths.csv"
ts_recovery_path = "Data/time_series_recovery.csv"
##

## define external style
## !! to do define the external stylesheet for the dashboard
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css',]
##

## init app
app3 = dash.Dash(__name__, external_stylesheets = external_stylesheets)
server = app3.server
##

## layout app
app3.layout = html.Div([
        html.Div([
            html.H1(
                children = "DISTRIBUTIONS PAPER",
                style = {
                    'color': 'grey',
                    'backgrundColor' : 'black',
                }
            )
        ], className = 'row'),
        html.Div([
            html.Div([
                dcc.Graph(id = 'fig_1',),
                dcc.Interval(id = 'fig_1_update', interval = 360 * 1000, n_intervals = 0)
            ], className = 'six columns'),
            html.Div([
                dcc.Graph(id = 'fig_2',),
                dcc.Interval(id = 'fig_2_update', interval = 360 * 1000, n_intervals = 0)
            ], className = 'six columns')
        ], className = 'row')
        ], style = {'backgroundColor' : "#18191c"
} , className = 'container-fluid')
##

## callback
@app3.callback([Output('fig_1', 'figure'),
                Output('fig_2', 'figure')],
                [Input('fig_1_update', 'n_intervals')])

def display_graph():
    for l in range(num_layers):
        link_weights = np.concatenate((CNet.weights[l].flatten(), CNet.biases[l]))
        min_, max_ = np.min(link_weights), np.max(link_weights)
        x = np.arange(min_, max_, abs(max_ - min_) / 1000)
        density = stats.kde.gaussian_kde(link_weights)
        df = pd.DataFrame({'x': x, 'y': density(x)})

        fig_1 = (px.histogram(
            df, x="x", y="y",
            title="{} Link Weights LAYER {}".format(dataset, l),
            nbins=100,
            hover_data=df.columns,
            opacity=0.8,
            color_discrete_sequence=['royalblue'],
            labels={"Density"}
        ))
        fig_1.update_layout(
            margin=dict(l=40, r=40, t=40, b=40),
            paper_bgcolor="white",
            xaxis_title_text='s',  # xaxis label
            yaxis_title_text='PDF(s)',  # yaxis label
            legend_title="Legend Title",
            font=dict(
                family="Courier New, monospace",
                size=12,
                color="black"
            ),
            title={
                'y': 0.985,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'}
        )


        fig_2 = (px.histogram(
            df, x="x", y="y",
            title="{} Link Weights LAYER {}".format(dataset, l),
            nbins=100,
            hover_data=df.columns,
            opacity=0.8,
            color_discrete_sequence=['royalblue'],
            labels={"Density"}
        ))
        fig_2.update_layout(
            margin=dict(l=40, r=40, t=40, b=40),
            paper_bgcolor="white",
            xaxis_title_text='s',  # xaxis label
            yaxis_title_text='PDF(s)',  # yaxis label
            legend_title="Legend Title",
            font=dict(
                family="Courier New, monospace",
                size=12,
                color="black"
            ),
            title={
                'y': 0.985,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'}
        )

        return [fig_1, fig_2]

if __name__ == '__main__':
    app3.run_server(debug=True)










