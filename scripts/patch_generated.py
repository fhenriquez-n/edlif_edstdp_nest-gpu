"""
patch_generated.py

Fixes three bugs in the CUDA code that NESTML 9.0.0 writes for the two models of this
package (edlif_psc_exp and edlif_psc_alpha). Run it after EVERY generate_gpu.py,
because generating writes those files again from scratch. Running it twice does no harm.

Usage:
    python3 patch_generated.py [nest_gpu_root]
nest_gpu_root: root folder of the NEST GPU repository (the one with CMakeLists.txt and
src/). If it is not given, <work>/src is used (<work> is $NESTGPU_WORK, or
<package>/work).

The three fixes, made in the file <model>_nestml.cu of each model:

 1. Spike weights 1000 times too small.
    NESTML writes (0.001 * y[N_SCAL_VAR + i_exc_spikes]) for every incoming spike. The
    0.001 is removed. (NESTML's own NEST GPU test model has the same factor.)

 2. Garbage in four state variables that have no equation: r, alp, bet and
    homeostatic_s_enabled. NESTML gives them no dydx line, so the integrator reads
    memory nobody wrote, and the ATP ends up as NaN. Each gets dydx = 0.

 3. The continuous input I_stim is read from param[] but never given a value.
    It is set to 0 (the NEST GPU program can change it later with SetNeuronParam).

To see what NESTML wrote before the fixes, run generate_gpu.py again.
"""

import os
import sys

# ---------------------------------------------------------------------------
# Where things are
# ---------------------------------------------------------------------------
package_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if len(sys.argv) > 1:
    nest_gpu_root = os.path.expanduser(sys.argv[1])
else:
    work_folder = os.path.expanduser(os.environ.get("NESTGPU_WORK", os.path.join(package_folder, "work")))
    nest_gpu_root = os.path.join(work_folder, "src")

# The C++ code of NEST GPU is in the src/ folder of its root
nest_gpu_src = os.path.join(nest_gpu_root, "src")
if not os.path.isdir(nest_gpu_src):
    sys.exit(f"{nest_gpu_root} is not the root of NEST GPU: it has no src/ folder.")

# The two models. NESTML adds "_nestml" to their names.
model_names = ["edlif_psc_alpha_nestml", "edlif_psc_exp_nestml"]


# ---------------------------------------------------------------------------
# Check that the generated files exist
# ---------------------------------------------------------------------------
for model_name in model_names:
    file_path = os.path.join(nest_gpu_src, model_name + ".cu")
    if not os.path.isfile(file_path):
        sys.exit(f"{file_path} does not exist. Run generate_gpu.py first.")


# ---------------------------------------------------------------------------
# Patch each model
# ---------------------------------------------------------------------------
for model_name in model_names:
    file_path = os.path.join(nest_gpu_src, model_name + ".cu")

    with open(file_path) as file:
        text = file.read()
    text_before = text

    # -----------------------------------------------------------------------
    # Fix 1: remove the factor 0.001 from the spike inputs
    # -----------------------------------------------------------------------
    text = text.replace("(0.001 * y[N_SCAL_VAR + i_exc_spikes])", "(y[N_SCAL_VAR + i_exc_spikes])")
    text = text.replace("(0.001 * y[N_SCAL_VAR + i_inh_spikes])", "(y[N_SCAL_VAR + i_inh_spikes])")

    # If a 0.001 is still there, it is in a form this script does not know
    if "0.001 *" in text:
        sys.exit(f"{file_path}: a 0.001 factor is left that this script does not know how to remove.")

    # -----------------------------------------------------------------------
    # Fix 2: dydx = 0 for the state variables that have no equation
    # -----------------------------------------------------------------------
    # The new lines go in Derivatives(), after the dydx = 0 lines of the recordable
    # variables, right before NESTML's comment about the input ports.
    place = "\n\n    // Input port variables should always be set to 0\n"
    new_lines = ("\n    dydx[i_r] = 0;"
                 "\n    dydx[i_alp] = 0;"
                 "\n    dydx[i_bet] = 0;"
                 "\n    dydx[i_homeostatic_s_enabled] = 0;")

    if "dydx[i_r] = 0;" in text:
        pass    # already done
    elif text.count(place) == 1:
        text = text.replace(place, new_lines + place)
    else:
        sys.exit(f"{file_path}: could not find the place in Derivatives() for the dydx = 0 lines "
                 f"(the comment 'Input port variables should always be set to 0').")

    # -----------------------------------------------------------------------
    # Fix 3: give I_stim a starting value
    # -----------------------------------------------------------------------
    # The new line goes in NodeInit(), right before NESTML's comment about the input ports.
    place = "\n  // Input port variables\n"
    new_lines = ("\n  // Continuous input ports: NESTML reads them from param[] but never sets them"
                 "\n  param[i_I_stim] = 0;\n")

    if "param[i_I_stim] = 0;" in text:
        pass    # already done
    elif text.count(place) == 1:
        text = text.replace(place, new_lines + place)
    else:
        sys.exit(f"{file_path}: could not find the place in NodeInit() for the I_stim line "
                 f"(the comment 'Input port variables').")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    if text == text_before:
        print(f"{model_name}.cu: already patched")
    else:
        with open(file_path, "w") as file:
            file.write(text)
        print(f"{model_name}.cu: patched")

print("\nNext: build.sh")
