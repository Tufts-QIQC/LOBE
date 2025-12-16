import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
from cycler import cycler
import os

_widths = {
    # a4paper columnwidth = 426.79135 pt = 5.93 in
    # letterpaper columnwidth = 443.57848 pt = 6.16 in
    "onecolumn": {"a4paper": 5.93, "letterpaper": 6.16},
    # a4paper columnwidth = 231.84843 pt = 3.22 in
    # letterpaper columnwidth = 240.24199 pt = 3.34 in
    "twocolumn": {"a4paper": 3.22, "letterpaper": 3.34},
}

_wide_widths = {
    # a4paper wide columnwidth = 426.79135 pt = 5.93 in
    # letterpaper wide columnwidth = 443.57848 pt = 6.16 in
    "onecolumn": {"a4paper": 5.93, "letterpaper": 6.16},
    # a4paper wide linewidth = 483.69687 pt = 6.72 in
    # letterpaper wide linewidth = 500.48400 pt = 6.95 in
    "twocolumn": {"a4paper": 6.72, "letterpaper": 6.95},
}

_fontsizes = {
    10: {
        "tiny": 5,
        "scriptsize": 7,
        "footnotesize": 8,
        "small": 9,
        "normalsize": 10,
        "large": 12,
        "Large": 14,
        "LARGE": 17,
        "huge": 20,
        "Huge": 25,
    },
    11: {
        "tiny": 6,
        "scriptsize": 8,
        "footnotesize": 9,
        "small": 10,
        "normalsize": 11,
        "large": 12,
        "Large": 14,
        "LARGE": 17,
        "huge": 20,
        "Huge": 25,
    },
    12: {
        "tiny": 6,
        "scriptsize": 8,
        "footnotesize": 10,
        "small": 11,
        "normalsize": 12,
        "large": 14,
        "Large": 17,
        "LARGE": 20,
        "huge": 25,
        "Huge": 25,
    },
}

_width = 1
_wide_width = 1
_quantumviolet = "#53257F"
_quantumgray = "#555555"


def global_setup(columns="twocolumn", paper="a4paper", fontsize=10):
    plt.rcdefaults()

    # Seaborn white is a good base style
    plt.style.use(
        [
            "seaborn-v0_8-white",
            "../../quantum-plots.mplstyle",
        ]
    )

    try:
        # This hackery is necessary so that jupyther shows the plots
        mpl.use("pgf")
        # %matplotlib inline
        plt.plot()
        mpl.use("pgf")
    except:
        print("Call to matplotlib.use had no effect")

    mpl.interactive(False)

    # Now prepare the styling that depends on the settings of the document

    global _width
    _width = _widths[columns][paper]

    global _wide_width
    _wide_width = _wide_widths[columns][paper]

    # Use the default fontsize scaling of LaTeX
    global _fontsizes
    fontsizes = _fontsizes[fontsize]

    plt.rcParams["axes.labelsize"] = fontsizes["small"]
    plt.rcParams["axes.titlesize"] = fontsizes["large"]
    plt.rcParams["xtick.labelsize"] = fontsizes["footnotesize"]
    plt.rcParams["ytick.labelsize"] = fontsizes["footnotesize"]
    plt.rcParams["font.size"] = fontsizes["small"]

    plt.rcParams["xtick.minor.visible"] = False
    plt.rcParams["ytick.minor.visible"] = False
    plt.rcParams["ytick.minor.size"] = 2
    plt.rcParams["ytick.minor.width"] = 0.6

    return {
        "fontsizes": fontsizes,
        "colors": {"quantumviolet": _quantumviolet, "quantumgray": _quantumgray},
    }


def plot_setup(aspect_ratio=1 / 1.62, width_ratio=1.0, wide=False):
    width = (_wide_width if wide else _width) * width_ratio
    height = width * aspect_ratio

    return plt.figure(figsize=(width, height), dpi=300, facecolor="white")
