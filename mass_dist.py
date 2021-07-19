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
    NumericInput,
    Div,
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

slice.line("x", "y", source=source, line_width=3, color="#8FBCBB", line_alpha=1)

slice_vline = Span(
    location=x[np.argmin(np.abs(q - 50))],
    dimension="height",
    line_color="#D08770",
    line_width=2,
)


slice.renderers.extend([slice_vline])

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

vline = Span(location=rvir_med, dimension="height", line_color="#8FBCBB", line_width=2)

dist.renderers.extend([vline])

# Set up widgets
radius_slider = Slider(
    title="Radius [kpc]", value=rvir_med, start=1.0, end=350.0, step=0.5
)
radius_input = NumericInput(
    mode="float",
    value=np.around(radius_slider.value, 3),
    low=0,
    high=350,
)

virial_button = Button(label="Virial radius")

percentile_slider = Slider(title="Percentile", value=50, start=0, end=100, step=0.05)

percentile_input = NumericInput(
    title="Percentile (press enter after changing)",
    mode="float",
    value=50,
    low=0,
    high=100,
)

percentile_value = Div(
    text=f"<h2>50th percentile mass: {np.percentile(masses, 50):.3f} x10^12 M_sun<h2>"
)

fixed_percentiles = Div(
    text=f"""
    <h2>
    16th percentile mass: {np.percentile(masses, 16):.3f}
    </h2>
    <h2>
    50th percentile mass: {np.percentile(masses, 50):.3f}
    </h2>
    <h2>
    84th percentile mass: {np.percentile(masses, 84):.3f}
    </h2>
    """
)


def update_virial_radius():
    radius_slider.value = rvir_med
    radius_input.value = np.around(radius_slider.value, 3)


def update_percentile():
    slice_vline.location = source.data["x"][
        np.argmin(np.abs(percentile_input.value - source.data["q"]))
    ]
    percentile_value.text = f"<h2>Mass: {np.percentile(masses, percentile_input.value):.3f} x10^12 M_sun<h2>"
    fixed_percentiles.text = f"""
    <h2>
    16th percentile mass: {np.percentile(masses, 16):.3f}
    </h2>
    <h2>
    50th percentile mass: {np.percentile(masses, 50):.3f}
    </h2>
    <h2>
    84th percentile mass: {np.percentile(masses, 84):.3f}
    </h2>
    """


def update_data(attrname, old, new):

    # Get the current slider values
    global masses
    r = new
    if radius_slider.value == new:
        radius_input.value = np.around(r, 3)
    elif radius_input.value == new:
        radius_slider.value = r
    masses = mass_at_radius(r, data["gammas"], data["phi0s"], full=True)
    kde = sm.nonparametric.KDEUnivariate(masses)
    kde.fit()
    x = np.linspace(masses.min(), masses.max(), 500)
    y = kde.evaluate(x)
    q = cumulative_trapezoid(y, x, initial=0) * 100
    slice.y_range.start = y.min()
    source.data = dict(x=x, y=y, q=q)
    vline.location = r
    update_percentile()


def update_title(attrname, old, new):
    slice.title.text = f"Mass distribution at r = {radius_slider.value:.1f} kpc"


def update_percentile_wrapper(attrname, old, new):
    if percentile_slider.value == new:
        percentile_input.value = np.around(new, 3)
    elif percentile_input.value == new:
        percentile_slider.value = np.around(new, 3)
    update_percentile()


virial_button.on_click(update_virial_radius)
radius_slider.on_change("value_throttled", update_title)
radius_slider.on_change("value_throttled", update_data)
radius_input.on_change("value", update_data)
percentile_slider.on_change("value_throttled", update_percentile_wrapper)
percentile_input.on_change("value", update_percentile_wrapper)

grid = column(
    row(dist, slice),
    row(
        column(
            radius_slider,
            row(radius_input, virial_button),
            percentile_slider,
            row(percentile_input, percentile_value),
        ),
        fixed_percentiles,
    ),
)

curdoc().add_root(grid)
curdoc().title = "Mass distribution"
