import os
import pandas as pd
from mprof import read_mprofile_file

CMD_BASE = "mprof run -o {o}.dat Rscript L0Learn_Profile.R --n {n} --p {p} --k {k} --s {s} --t {t} --w {w} --m {m} --f {f}"


file_name = "test_run3"

run = {
    "n": 1000,
    "p": 10000,
    "k": 10,
    "s": 1,
    "t": 2.1,
    "w": 4,
    "m": 1,
    "f": file_name,
    "o": file_name,
}
cmd = CMD_BASE.format(**run)

os.system(
    cmd
)  # Creates <file_name>.dat and <file_name>.csv files in same directory as file.
# This cmd will often error out for no reason.
# https://github.com/pythonprofilers/memory_profiler/issues/240

memory_usage = read_mprofile_file(file_name + ".dat")
timing = pd.read_csv(file_name + ".csv")
