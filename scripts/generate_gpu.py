"""
generate_gpu.py

Turns the EDLIF neuron models written in NESTML (models/*.nestml) into CUDA/C++ code
and registers them inside a NEST GPU source tree.

Unlike the CPU, the result is not a separate module: the generated code is written
INTO the source of NEST GPU, so NEST GPU has to be rebuilt afterwards.

Run it on a python environment with NESTML 9.0.0 installed. No GPU is needed.

Usage:
    python3 generate_gpu.py [nest_gpu_root] [--with-nestml-build]

nest_gpu_root   Root folder of the NEST GPU repository: the one that has CMakeLists.txt
                and src/. If it is not given, <work>/src is used (<work> is $NESTGPU_WORK,
                or <package>/work).
--with-nestml-build
                NESTML normally also tries to compile NEST GPU after generating the code
                (cmake + make + make install into folders next to the NEST GPU root called
                nest-gpu-build and nest-gpu-install). Nothing in this package uses that
                result, it takes a long time and it needs nvcc, so it is skipped unless
                asked for it with this flag.

What NESTML does, in this order:
    1. It writes the generated code (.h and .cu) in a staging folder, <work>/nestml_target.
       This script empties that folder first, so only the models of models/ are in it.
    2. It copies ALL the .h and .cu files of the staging folder into <root>/src.
    3. It rewrites what is between the marker comments <<BEGIN_NESTML_GENERATED>> and
       <<END_NESTML_GENERATED>> in three files of <root>/src, after saving a copy of
       each as <file>.bak:
           CMakeLists.txt     the list of source files to compile
           neuron_models.h    the number of each model and its name
           neuron_models.cu   the #include of each model and the code that creates it
       The markers are added to a clean NEST GPU by add_markers.py. If one is missing,
       NESTML does not complain: it cuts the file. This script checks them BEFORE it
       calls NESTML and refuses to continue if they are not in place.

After this script: patch_generated.py (fixes bugs of the generated code, needed after
EVERY generation), then build.sh.
"""

import os
import sys

BEGIN = "<<BEGIN_NESTML_GENERATED>>"
END = "<<END_NESTML_GENERATED>>"

# The two models, in alphabetical order (the order in which they were registered in the
# validated NEST GPU tree). NESTML adds "_nestml" to their names, because of the suffix option below.
model_names = ["edlif_psc_alpha", "edlif_psc_exp"]
suffix = "_nestml"


# ---------------------------------------------------------------------------
# Where things are
# ---------------------------------------------------------------------------
package_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
work_folder = os.path.expanduser(os.environ.get("NESTGPU_WORK", os.path.join(package_folder, "work")))
models_folder = os.path.join(package_folder, "models")
staging_folder = os.path.join(work_folder, "nestml_target")

# The arguments: the root of NEST GPU, and the flag --with-nestml-build
with_nestml_build = "--with-nestml-build" in sys.argv[1:]
arguments = [argument for argument in sys.argv[1:] if argument != "--with-nestml-build"]

if len(arguments) > 0:
    nest_gpu_root = os.path.abspath(os.path.expanduser(arguments[0]))
else:
    nest_gpu_root = os.path.abspath(os.path.join(work_folder, "src"))

# The C++ code of NEST GPU is in the src/ folder of its root
nest_gpu_src = os.path.join(nest_gpu_root, "src")

if not os.path.isfile(os.path.join(nest_gpu_src, "neuron_models.h")):
    sys.exit(f"{nest_gpu_root} is not the root of NEST GPU: {nest_gpu_src}/neuron_models.h does not exist.\n"
             f"(It must be the folder that contains CMakeLists.txt and src/, not the src/ folder itself.)")

for model_name in model_names:
    if not os.path.isfile(os.path.join(models_folder, model_name + ".nestml")):
        sys.exit(f"{os.path.join(models_folder, model_name + '.nestml')} does not exist.")


# ---------------------------------------------------------------------------
# Check the markers before NESTML runs
# ---------------------------------------------------------------------------
# NESTML cuts a file when a marker is missing, so this must be checked first.
# CMakeLists.txt must have 1 pair of markers, neuron_models.h 2 pairs, neuron_models.cu 2 pairs.
for file_name, pairs in [("CMakeLists.txt", 1), ("neuron_models.h", 2), ("neuron_models.cu", 2)]:
    file_path = os.path.join(nest_gpu_src, file_name)
    if not os.path.isfile(file_path):
        sys.exit(f"{file_path} does not exist.")

    with open(file_path) as file:
        text = file.read()
    if text.count(BEGIN) != pairs or text.count(END) != pairs:
        sys.exit(f"The NESTML markers are not in place: {file_name} should have {pairs} pair(s) "
                 f"but has {text.count(BEGIN)} begin / {text.count(END)} end markers.\n"
                 f"Run add_markers.py first.")


# ---------------------------------------------------------------------------
# Load NESTML and switch off its own build
# ---------------------------------------------------------------------------
# NESTML is imported only now, so that step 1 works even without NESTML installed.
from pynestml.frontend.pynestml_frontend import generate_nest_gpu_target

if not with_nestml_build:
    # NESTML builds NEST GPU by calling the method build() of NESTGPUBuilder.
    # It is replaced by a method that does nothing.
    import pynestml.codegeneration.nest_gpu_builder as nest_gpu_builder

    def build_nothing(self):
        pass

    nest_gpu_builder.NESTGPUBuilder.build = build_nothing

# NESTML finds the NEST GPU tree through this variable
os.environ["NEST_GPU"] = nest_gpu_root


# ---------------------------------------------------------------------------
# Empty the staging folder
# ---------------------------------------------------------------------------
# Models generated earlier must not be copied into NEST GPU again.
os.makedirs(staging_folder, exist_ok=True)
for file_name in os.listdir(staging_folder):
    if file_name.endswith(".h") or file_name.endswith(".cu"):
        os.remove(os.path.join(staging_folder, file_name))

print(f"NEST GPU root : {nest_gpu_root}")
print(f"Models        : {model_names}")
print(f"Staging folder: {staging_folder}\n")


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------
generate_nest_gpu_target(
    input_path=[os.path.join(models_folder, model_name + ".nestml") for model_name in model_names],
    target_path=staging_folder,
    logging_level="INFO",
    suffix=suffix,
    codegen_opts={"solver": "numeric"},   # numeric solver (RK5 on the GPU)
)


# ---------------------------------------------------------------------------
# Check the result
# ---------------------------------------------------------------------------
# NESTML reports its errors in its log instead of stopping, so the result is checked here.
problems = []

for model_name in model_names:
    full_name = model_name + suffix

    # The two files of the model must be in NEST GPU
    for extension in [".h", ".cu"]:
        if not os.path.isfile(os.path.join(nest_gpu_src, full_name + extension)):
            problems.append(f"{full_name}{extension} was not copied into {nest_gpu_src}")

    # The model must be mentioned in the three files NESTML edits
    with open(os.path.join(nest_gpu_src, "CMakeLists.txt")) as file:
        text = file.read()
    if full_name + ".cu" not in text:
        problems.append(f"CMakeLists.txt does not mention {full_name}.cu")

    with open(os.path.join(nest_gpu_src, "neuron_models.h")) as file:
        text = file.read()
    if f"i_{full_name}_model" not in text or f'"{full_name}"' not in text:
        problems.append(f"neuron_models.h does not mention i_{full_name}_model and \"{full_name}\"")

    with open(os.path.join(nest_gpu_src, "neuron_models.cu")) as file:
        text = file.read()
    if f'#include "{full_name}.h"' not in text or f"new {full_name}" not in text:
        problems.append(f"neuron_models.cu does not include {full_name}.h and create {full_name}")

# The markers must still be there (same count as in step 1)
for file_name, pairs in [("CMakeLists.txt", 1), ("neuron_models.h", 2), ("neuron_models.cu", 2)]:
    with open(os.path.join(nest_gpu_src, file_name)) as file:
        text = file.read()
    if text.count(BEGIN) != pairs or text.count(END) != pairs:
        problems.append(f"{file_name} should have {pairs} marker pair(s) but has "
                        f"{text.count(BEGIN)} begin / {text.count(END)} end")

if len(problems) > 0:
    sys.exit("\nGeneration FAILED (look for ERROR in the output above):\n  " + "\n  ".join(problems))


# ---------------------------------------------------------------------------
# Remove the .bak files
# ---------------------------------------------------------------------------
# NESTML leaves a .bak next to each file it rewrote. They are only noise in a throw-away clone.
for file_name in ["CMakeLists.txt", "neuron_models.h", "neuron_models.cu"]:
    backup = os.path.join(nest_gpu_src, file_name + ".bak")
    if os.path.isfile(backup):
        os.remove(backup)

print("\nSuccess: models generated and registered:", [model_name + suffix for model_name in model_names])
print("Next: python3 patch_generated.py", nest_gpu_root)
