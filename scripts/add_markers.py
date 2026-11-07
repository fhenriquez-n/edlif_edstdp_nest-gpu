"""
add_markers.py

Adds the NESTML marker comments to a clean NEST GPU source tree.

NESTML does not add models to NEST GPU by itself: it rewrites the text that sits
between two marker comments, in three files of NEST GPU. A clean NEST GPU does not have
those markers, so they have to be added once. This script does it.

    src/CMakeLists.txt     1 pair   list of source files to compile
                                    (markers start with "# ")
    src/neuron_models.h    2 pairs  the number of each model / the name of each model
    src/neuron_models.cu   2 pairs  the #include of each model / the code that creates it
                                    (markers start with "// ")

Every pair is two lines, with nothing between them:
    <BEGIN_NESTML_GENERATED>
    <END_NESTML_GENERATED>
NESTML fills the gap when it generates a model.

Usage:
    python3 add_markers.py [nest_gpu_root]
nest_gpu_root is the root folder of the NEST GPU repository (the one that has
CMakeLists.txt and src/). If it is not given, <work>/src is used 

It is safe to run it again
"""

import os
import sys

BEGIN = "<<BEGIN_NESTML_GENERATED>>"
END = "<<END_NESTML_GENERATED>>"


# ---------------------------------------------------------------------------
# Where things are
# ---------------------------------------------------------------------------
package_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if len(sys.argv) > 1:
    nest_gpu_root = os.path.expanduser(sys.argv[1])
else:
    work_folder = os.environ.get("NESTGPU_WORK", os.path.join(package_folder, "work"))
    nest_gpu_root = os.path.join(work_folder, "src")

# The C++ code of NEST GPU is in the src/ folder of its root
nest_gpu_src = os.path.join(nest_gpu_root, "src")

if not os.path.isfile(os.path.join(nest_gpu_src, "neuron_models.h")):
    sys.exit(f"{nest_gpu_root} is not the root of NEST GPU: {nest_gpu_src}/neuron_models.h does not exist.\n"
             f"(It must be the folder that contains CMakeLists.txt and src/, not the src/ folder itself.)")

print(f"NEST GPU root: {nest_gpu_root}")


# ---------------------------------------------------------------------------
# Helper function
# ---------------------------------------------------------------------------
def add_markers(file_name, old, new, what):
    """
    In a file of NEST GPU, changes the text old into the text new. new is old with a
    pair of markers added, so this is how a pair gets into the file.

    If the begin marker is already where it goes, nothing is done
    """
    path = os.path.join(nest_gpu_src, file_name)
    with open(path) as file:
        text = file.read()

    # The start of new, up to and including the begin marker, is in the file only once marked
    already_marked = new[:new.index(BEGIN) + len(BEGIN)]

    if already_marked in text:
        print(f"  skip   {file_name}: {what} (already there)")
    elif text.count(old) == 1:
        with open(path, "w") as file:
            file.write(text.replace(old, new))
        print(f"  done   {file_name}: {what}")
    else:
        sys.exit(f"  FAILED {file_name}: {what}: the text to look for was found {text.count(old)} times "
                 f"(expected 1). Look at this place by hand.")


# ---------------------------------------------------------------------------
# CMakeLists.txt: the list of files to compile
# ---------------------------------------------------------------------------
# The pair goes after the last file of the list (nestgpu_C.cpp), still inside the list.
add_markers("CMakeLists.txt",
            old="nestgpu_C.cpp\n",
            new=f"nestgpu_C.cpp\n# {BEGIN}\n# {END}\n",
            what="pair after nestgpu_C.cpp")


# ---------------------------------------------------------------------------
# neuron_models.h: the numbers and the names of the neuron models
# ---------------------------------------------------------------------------
# The list of model numbers: after the last one (user_m2), before N_NEURON_MODELS.
add_markers("neuron_models.h",
            old="  i_user_m2_model,\n",
            new=f"  i_user_m2_model,\n  // {BEGIN}\n  // {END}\n",
            what="pair after i_user_m2_model")

# The list of model names: NEST GPU closes it on the same line as the last name
# ("user_m2"};). The closing }; goes to its own line, below the pair. NESTML writes
# its names with a comma in front (, "name"), so the last name needs no comma.
add_markers("neuron_models.h",
            old='    "user_m2"};\n',
            new=f'    "user_m2"\n  // {BEGIN}\n  // {END}\n  }};\n',
            what='pair after the name "user_m2"')


# ---------------------------------------------------------------------------
# neuron_models.cu: the #include of each model and the code that creates it
# ---------------------------------------------------------------------------
# The #include list: after #include "user_m2.h"
add_markers("neuron_models.cu",
            old='#include "user_m2.h"\n',
            new=f'#include "user_m2.h"\n// {BEGIN}\n// {END}\n',
            what='pair after #include "user_m2.h"')

# _Create is a chain of "} else if (...) {" that ends with "} else {" and a throw
# ("Unknown neuron model name"). The pair goes between the last "}" and the "else",
# which then starts its own line. NESTML writes its "else if" blocks in that gap.
add_markers("neuron_models.cu",
            old='  } else {\n    throw ngpu_exception(std::string("Unknown neuron model name',
            new=f'  }}\n  // {BEGIN}\n  // {END}\n  else {{\n    throw ngpu_exception(std::string("Unknown neuron model name',
            what="pair before the final else of _Create")


# ---------------------------------------------------------------------------
# Final check: every file must have the number of pairs it should
# ---------------------------------------------------------------------------
# CMakeLists.txt must have 1 pair, neuron_models.h 2 pairs, neuron_models.cu 2 pairs
for file_name, pairs in [("CMakeLists.txt", 1), ("neuron_models.h", 2), ("neuron_models.cu", 2)]:
    with open(os.path.join(nest_gpu_src, file_name)) as file:
        text = file.read()
    if text.count(BEGIN) != pairs or text.count(END) != pairs:
        sys.exit(f"\nMarkers are NOT in place: {file_name} should have {pairs} pair(s) "
                 f"but has {text.count(BEGIN)} begin / {text.count(END)} end markers.")

print("\nAll marker pairs are in place.")
