import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from typing import List


def create_scatter(pos_3d: np.ndarray,
                   mode: str = 'markers',
                   marker_size: int = None,
                   name: str = None,
                   opacity: float = None,
                   color: List = None,
                   colorscale: str = None,
                   ):
    """Creates Scatter3D objects for use in plotly.

    Args:
        pos_3d: [N, 3] array containing N points with
            euclidean coordinates.
        mode: How to display points.
            Use 'markers' for scatter.
            Use 'lines' for lines connecting consecutive points.
            Use 'lines+markers' for scatter and lines.
        marker_size: Size of markers.
        name: Label of plotting layer to be displayed in legend.
        opacity: Transparency of points.
    """
    x, y, z = np.split(pos_3d, 3, axis=-1)
    args_dict = {
        'x': x[:, 0],
        'y': y[:, 0],
        'z': z[:, 0],
        'mode': mode,
        'marker': {}
    }
    if marker_size:
        args_dict['marker']['size'] = marker_size
    if name:
        args_dict['name'] = name
    if opacity:
        args_dict['opacity'] = opacity
    if color:
        args_dict['marker']['color'] = color
    if colorscale:
        args_dict['marker']['colorscale'] = colorscale
    return go.Scatter3d(**args_dict)


def create_cones(
        tail_3d: np.ndarray,
        head_3d: np.ndarray,
        sizemode: str = 'absolute',
        name: str = None,
        opacity: float = None,
        sizeref: int = None,
    ):
    """Creates Cone object for use in plotly.

    Args:
        tail_3d: [N, 3] array containing N points for where the cones begin.
        head_3d: [N, 3] array containing N points of the offsets from tail_3d.
        sizemode: Determines size the cones.
        sizeref: Scaling factor for cone size.
        name: Label of plotting layer to be displayed in legend.
        opacity: Transparency of points.
    """
    x, y, z = np.split(tail_3d, 3, axis=-1)
    u, v, w = np.split(head_3d, 3, axis=-1)
    args_dict = {
        'x': x[:, 0],
        'y': y[:, 0],
        'z': z[:, 0],
        'u': u[:, 0],
        'v': v[:, 0],
        'w': w[:, 0]
    }
    if sizemode:
        args_dict['sizemode'] = sizemode
    if name:
        args_dict['name'] = name
    if opacity:
        args_dict['opacity'] = opacity
    if sizeref:
        args_dict['sizeref'] = sizeref
    return go.Cone(**args_dict)


def plot_traces(
        fig_traces,
        height=500,
        width=600,
        title=None
    ):
    """Constructs
    """
    layout_args = {
        'height': height,
        'width': width,
    }
    if title is not None:
        layout_args['title'] = title
    layout = go.Layout(**layout_args)
    fig = go.Figure(data=fig_traces, layout=layout)
    fig.show()


def plot_traces_and_layout(
        fig_traces,
        layout
    ):
    fig = go.Figure(data=fig_traces, layout=layout)
    fig.show()


def create_static_layout(
        height=500,
        width=600,
        title=None,
        xaxis=None,
        yaxis=None,
    ):
    layout_args = {
        'height': height,
        'width': width,
    }
    if title is not None:
        layout_args['title'] = title
    if xaxis is not None:
        layout_args['xaxis'] = dict(range=xaxis, autorange=False)
    if yaxis is not None:
        layout_args['yaxis'] = dict(range=yaxis, autorange=False)
    return go.Layout(**layout_args)


def create_dynamic_layout(
        height=500,
        width=600,
        title=None,
        scene_range=None
    ):
    layout_args = {
        'height': height,
        'width': width,
        'autosize': False,
        'updatemenus': [{
            "buttons": [
                {
                    "args": [
                        None,
                        {
                            "frame": {
                                "duration": 30
                            },
                            "transition": {
                                "duration": 30
                            }
                        }
                    ],
                    "label": "Play",
                    "method": "animate"
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }]
    }
    if scene_range is not None:
        layout_args['scene'] = {
            'xaxis': scene_range,
            'yaxis': scene_range,
            'zaxis': scene_range,
            'aspectmode': 'cube',
        }
    if title is not None:
        layout_args['title'] = title
    return go.Layout(**layout_args)

# Plot samples
def plot_sample_grid(samples, num_res, motif_bb_3d=None, true_bb_3d=None):
    ncols, nrows = 3, 3
    fig = make_subplots(
        rows=nrows, cols=ncols,
        specs=[[{'type': 'surface'}] * nrows]*ncols)
    fig.update_layout(
        title_text=f'Samples',
        height=1000,
        width=1000,
    )
    for i in range(nrows):
        for j in range(ncols):
            b_idx = i*nrows+j
            sample_ij = samples[b_idx][:(num_res[b_idx])]
            sample_bb_3d = create_scatter(
                sample_ij, mode='lines+markers', marker_size=3,
                opacity=1.0, name=f'Sample {i*nrows+j}: length_{num_res[b_idx]}')
            fig.add_trace(sample_bb_3d, row=i+1, col=j+1)

            if motif_bb_3d is not None: fig.add_trace(motif_bb_3d, row=i+1, col=j+1)
            if true_bb_3d is not None: fig.add_trace(true_bb_3d, row=i+1, col=j+1)


    fig.show()
