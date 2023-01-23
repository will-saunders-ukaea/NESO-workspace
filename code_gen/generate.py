import os
import hashlib

if not os.path.exists("generated_code"):
    os.mkdir("generated_code")




class GeneratedClass:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        s = f"""
/**
* Generated class {self.name}
*/
class {self.name} {{
private:
    std::map<std::string, std::shared_ptr<void>> dats;
public:

{self.name}(){{}};

template<typename T>
inline void register_symbol(std::string name, std::shared_ptr<T> dat){{
    this->dats[name] = dat;
}}

}};
"""
        return s



def write_generated_classes(generated_classes, path):
    
    generated_source = [str(gx) for gx in generated_classes]
    h = hashlib.md5("\n".join(generated_source).encode("utf-8")).hexdigest()

    interface_source = """#ifndef _H_INTERFACE_{HASH}
#define _H_INTERFACE_{HASH}

#include <neso_particles.hpp>
#include <string>
#include <memory>

namespace NESO::Particles {{

{SOURCE}


}}
#endif
"""
    interface_source = interface_source.format(
        HASH=h,
        SOURCE="\n".join(generated_source),
    )

    with open(path, "w") as fh:
        fh.write(interface_source)





g = GeneratedClass("Foo")






generated_classes = [g,]
write_generated_classes(generated_classes, "generated_code/interface.hpp")




