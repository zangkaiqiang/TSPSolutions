import random
import pandas as pd
import numpy as np
from solomon import Solomon

if __name__ == '__main__':
    filepath = 'data/solomon-100/c101.25.txt'
    slm = Solomon()
    slm.read_solomon(filepath)
    slm.compute_matrix()
    expect_path, expect_distace = slm.tabu_search()
    print(expect_distace)
    print(expect_path)
    slm.plot(expect_path)