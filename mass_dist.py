from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, Button, HoverTool
from bokeh.plotting import figure

import numpy as np
import statsmodels.api as sm
from scipy.integrate import cumulative_trapezoid
from scripts.utils import mass_at_radius, virial_radius
import pickle

data = pickle.load(open("samples.pkl", "rb"))
rvir = virial_radius(data["gammas"], data["phi0s"])
rvir_med = np.median(rvir)

# initial setup
masses = mass_at_radius(rvir, data["gammas"], data["phi0s"], full=True)
kde = sm.nonparametric.KDEUnivariate(masses)
kde.fit()
x = np.array([0])
x = np.append(x, np.linspace(masses.min(), masses.max(), 250))
x = np.append(x, 2)
y = kde.evaluate(x)
q = cumulative_trapezoid(y, x, initial=0)
source = ColumnDataSource(data=dict(x=x, y=y, q=q))

plot = figure(
    # height=800,
    # width=800,
    title=f"Mass distribution at r = {rvir_med:.1f} kpc",
    tools="pan,reset,save,wheel_zoom",
    x_range=[0.0, 2],
)

plot.yaxis.visible = False
plot.ygrid.visible = False

hover = HoverTool(
    tooltips=[
        ("Percentile", "@q{%F}"),
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
    x = np.array([0])
    x = np.append(x, np.linspace(masses.min(), masses.max(), 250))
    x = np.append(x, 2)
    y = kde.evaluate(x)
    q = cumulative_trapezoid(y, x, initial=0)
    source.data = dict(x=x, y=y, q=q)


def update_title(attrname, old, new):
    plot.title.text = f"Mass distribution at r = {radius_slider.value:.1f} kpc"


virial_button.on_click(update_virial_radius)
radius_slider.on_change("value", update_title)
radius_slider.on_change("value", update_data)


# Set up layouts and add to document
inputs = column(radius_slider, virial_button)

curdoc().add_root(row(inputs, plot, width=800))
curdoc().title = "Mass distribution"
