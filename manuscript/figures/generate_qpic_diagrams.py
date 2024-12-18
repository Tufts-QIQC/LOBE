from os import listdir
from os.path import isfile, join
import subprocess
import glob
import pathlib
import os

import os

import os

filenames = [
    os.path.abspath(os.path.join(root, filename))
    for root, _, files in os.walk("qpic-code/")
    for filename in files
]

for filename in filenames:
    if filename[-5:] == ".qpic":
        diagram_filename = filename[:-5] + ".tex"
        f = open(diagram_filename, "w")
        subprocess.call(["qpic", filename], stdout=f)
