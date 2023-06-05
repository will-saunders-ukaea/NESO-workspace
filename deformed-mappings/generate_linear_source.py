import sys
import os
import prism
import tetrahedron
import hexahedron
import quad
from newton_generation import *

geom_types = (
    prism.get_geom_type(),
    tetrahedron.get_geom_type(),
    hexahedron.get_geom_type(),
    quad.get_geom_type(),
)
geom_objects = [gx() for gx in geom_types]
geom_newton = [Newton(gx) for gx in geom_objects]
geom_ccode = [NewtonLinearCCode(gx) for gx in geom_newton]

output = {}
for gx in geom_ccode:
    filename = "{}.hpp".format(gx.newton.geom.namespace.lower())
    source = """/**
    This is a generated file. Please make none ephemeral changes by
    modifing the script which generates this file.
*/
#ifndef __GENERATED_{NAMESPACE}_LINEAR_NEWTON_H__
#define __GENERATED_{NAMESPACE}_LINEAR_NEWTON_H__

namespace NESO {{
namespace {NAMESPACE} {{

{STEP}

{RESIDUAL}

}}
}}

#endif
""".format(
        NAMESPACE=gx.newton.geom.namespace, STEP=gx.step(), RESIDUAL=gx.residual()
    )

    output[filename] = source


output_dir = sys.argv[1]
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    assert os.path.exists(output_dir)

for filename in output.keys():
    p = os.path.join(output_dir, filename)
    with open(p, "w") as fh:
        fh.write(output[filename])
