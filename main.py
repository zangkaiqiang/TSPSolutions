from core.vrppd import vrp

vrp_case = vrp()
file_name = 'data/solomon-100/In/r101.txt'
vrp_case.read_solomon(file_name)
vrp_case.plot()