import sys
import os
import prism
import pyramid
import tetrahedron
import hexahedron
import quad
from newton_generation import *

if len(sys.argv) < 2 or "-h" in sys.argv:
    print(
        """
This is a wrapper script that generates the implementations to compute Newton
steps and residuals for 3D linear Nektar++ geometry objects (and 2D
linear quadrilaterals). The script is called with a single argument like

python generate_linear_source.py <output_dir>

where <output_dir> is the output directory where generated code will be placed.
"""
    )
    quit()


geom_types = (
    prism.get_geom_type(),
    pyramid.get_geom_type(),
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

#include <neso_particles.hpp>
using namespace NESO;
using namespace NESO::Particles;

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


includes = "\n".join([f'#include "{s}"' for s in output.keys()])

with open(os.path.join(output_dir, "linear_newton_implementation.hpp"), "w") as fh:
    fh.write(
        f"""#ifndef __GENERATED_LINEAR_NEWTON_IMPLEMENTATIONS_H__
#define __GENERATED_LINEAR_NEWTON_IMPLEMENTATIONS_H__

{includes}

#endif
"""
    )