# import all the modules for each geometry type and run the self test

import importlib

modules = (
    "hexahedron",
    "prism",
    "pyramid",
    "quadrilateral",
    "tetrahedron",
)

for module in modules:
    m = importlib.import_module(module)
    m.self_test()
