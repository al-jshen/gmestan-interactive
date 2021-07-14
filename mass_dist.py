from bokeh.io import curdoc
from bokeh.layouts import layout, column, row
from bokeh.models import (
    ColumnDataSource,
    Slider,
    Button,
    HoverTool,
    Range1d,
    Patch,
    Span,
)
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

slice = figure(
    title=f"Mass distribution at r = {rvir_med:.1f} kpc",
    tools="save",
    x_range=[0.0, 2],
)


slice.yaxis.visible = False
slice.y_range.start = y.min()
slice.ygrid.visible = False
slice.xaxis.axis_label = "Mass [x10^12 M_sun]"
slice.xaxis.axis_label_text_font_size = "16pt"
slice.title.text_font_size = "20pt"

hover = HoverTool(
    tooltips=[
        ("Percentile", "@q{0.00}%"),
        (r"Mass [x10^12 M_sun]", "@x"),
    ],
    mode="vline",
)

slice.add_tools(hover)

slice.line("x", "y", source=source, line_width=3, line_alpha=0.6)

# cumulative mass profile

dist = figure(
    title="Cumulative mass profile", tools="save", x_range=[0.0, 350], y_range=[0.0, 2]
)


dist.xaxis.axis_label = "Radius [kpc]"
dist.xaxis.axis_label_text_font_size = "16pt"
dist.yaxis.axis_label = "Mass [x10^12 M_sun]"
dist.yaxis.axis_label_text_font_size = "16pt"
dist.title.text_font_size = "20pt"

x1 = np.arange(1, 350.5, 0.5)
x2 = x1[::-1]
massprof = np.zeros((len(x1), 7))
for i in range(len(x1)):
    massprof[i] = mass_at_radius(x1[i], data["gammas"], data["phi0s"])
massprof = massprof.T

cis = ColumnDataSource(
    dict(
        x=np.hstack((x1, x2)),
        ci68=np.hstack((massprof[2], massprof[4][::-1])),
        ci95=np.hstack((massprof[1], massprof[5][::-1])),
        ci997=np.hstack((massprof[0], massprof[6][::-1])),
    )
)


glyph68 = Patch(x="x", y="ci68", line_alpha=0, fill_color="#5E81AC", fill_alpha=0.5)
glyph95 = Patch(x="x", y="ci95", line_alpha=0, fill_color="#5E81AC", fill_alpha=0.35)
glyph997 = Patch(x="x", y="ci997", line_alpha=0, fill_color="#5E81AC", fill_alpha=0.2)

dist.add_glyph(cis, glyph68)
dist.add_glyph(cis, glyph95)
dist.add_glyph(cis, glyph997)

dist.line(x1, massprof[3], line_width=2)

vline = Span(location=rvir_med, dimension="height", line_color="#81A1C1", line_width=2)

dist.renderers.extend([vline])

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
    slice.y_range.start = y.min()
    source.data = dict(x=x, y=y, q=q)
    vline.location = r


def update_title(attrname, old, new):
    slice.title.text = f"Mass distribution at r = {radius_slider.value:.1f} kpc"


virial_button.on_click(update_virial_radius)
radius_slider.on_change("value", update_title)
radius_slider.on_change("value", update_data)


grid = column(row(dist, slice), radius_slider, virial_button)

curdoc().add_root(grid)
curdoc().title = "Mass distribution"
