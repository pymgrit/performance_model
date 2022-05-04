import json
import os
import sys
import numpy as np

def read_results(dir_in, dir_out):
    runtimes = []
    dir_list = [x[0] for x in os.walk(dir_in)]
    for item in dir_list:
        if (dir_in+'/') in item:
            with open(item+'/runtime.json') as json_fi:
                runtimes.append(json.load(json_fi)["total"])
    np.save(dir_out, np.array(runtimes))

stdin = sys.argv
dir_in = stdin[1]
dir_out = stdin[2]
runtimes = read_results(dir_in=dir_in, dir_out=dir_out)
