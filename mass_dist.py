from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Slider, Button, HoverTool, Range1d
from bokeh.plotting import figure

import numpy as np
import statsmodels.api as sm
from scipy.integrate import cumulative_trapezoid, trapezoid
from scripts.utils import mass_at_radius, virial_radius
import pickle

data = pickle.load(open("samples.pkl", "rb"))
rvir = virial_radius(data["gammas"], data["phi0s"])
rvir_med = np.median(rvir)

# initial setup
masses = mass_at_radius(rvir, data["gammas"], data["phi0s"], full=True)
kde = sm.nonparametric.KDEUnivariate(masses)
kde.fit()
x = np.linspace(masses.min(), masses.max(), 500)
y = kde.evaluate(x)
q = cumulative_trapezoid(y, x, initial=0) * 100
source = ColumnDataSource(data=dict(x=x, y=y, q=q))

plot = figure(
    # height=800,
    # width=800,
    title=f"Mass distribution at r = {rvir_med:.1f} kpc",
    tools="save",
    x_range=[0.0, 2],
)

plot.yaxis.visible = False
plot.y_range.start = y.min()
plot.ygrid.visible = False
plot.xaxis.axis_label = "Mass [x10^12 M_sun]"

hover = HoverTool(
    tooltips=[
        ("Percentile", "@q{0.00}%"),
        (r"Mass [x10^12 M_sun]", "@x"),
    ],
    mode="vline",
)

plot.add_tools(hover)

plot.line("x", "y", source=source, line_width=3, line_alpha=0.6)

# Set up widgets
radius_slider = Slider(
    title="radius [kpc]", value=rvir_med, start=1.0, end=350.0, step=0.5
)
virial_button = Button(label="Virial radius")


def update_virial_radius():
    radius_slider.value = rvir_med


def update_data(attrname, old, new):

    # Get the current slider values
    r = radius_slider.value
    masses = mass_at_radius(r, data["gammas"], data["phi0s"], full=True)
    kde = sm.nonparametric.KDEUnivariate(masses)
    kde.fit()
    x = np.linspace(masses.min(), masses.max(), 500)
    y = kde.evaluate(x)
    q = cumulative_trapezoid(y, x, initial=0) * 100
    plot.y_range.start = y.min()
    source.data = dict(x=x, y=y, q=q)


def update_title(attrname, old, new):
    plot.title.text = f"Mass distribution at r = {radius_slider.value:.1f} kpc"


virial_button.on_click(update_virial_radius)
radius_slider.on_change("value", update_title)
radius_slider.on_change("value", update_data)


# Set up layouts and add to document
inputs = column(radius_slider, virial_button)

curdoc().add_root(column(plot, inputs, width=800))
curdoc().title = "Mass distribution"
