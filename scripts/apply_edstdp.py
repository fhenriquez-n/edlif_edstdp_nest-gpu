"""
apply_edstdp.py

Adds the ED-STDP synapse to a NEST GPU source tree 

NEST GPU does not have this rule, so the script copies two new files into it
and edits some of its source files.

Files copied (from the edstdp/ folder of this package):
    edstdp.h, edstdp.cu  the plasticity rule

Files edited inside NEST GPU:
    syn_model.h     registers the new synapse model, and gives the function that
                    updates a weight a fourth argument: the target neuron
    syn_model.cu    creates the model when asked for, and at calibration builds a
                    table that tells the GPU where each neuron keeps its ATP
    get_spike.h     passes the target neuron where a weight is weakened
    rev_spike.h     passes the target neuron where a weight is strengthened
    rev_spike.cu    removes an old declaration of the update function (it would
                    clash with the new 4-argument version)
    CMakeLists.txt  adds edstdp.h and edstdp.cu to the list of files to compile

Usage:
    python3 apply_edstdp.py [nest_gpu_root]
nest_gpu_root is the root folder of the NEST GPU repository (the one that has
CMakeLists.txt and src/). If it is not given, <work>/src is used 

It is safe to run it again.

When it finishes, NEST GPU has to be built manually via the nest-gpu page 
instructions or using build.sh
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
    nest_gpu_root = os.path.join(os.environ.get("NESTGPU_WORK", os.path.join(package_folder, "work")), "src")

# The C++ code of NEST GPU is in the src/ folder of its root
nest_gpu_src = os.path.join(nest_gpu_root, "src")

# edstdp.h and edstdp.cu are taken from the edstdp/ folder of this package
edstdp_folder = os.path.join(package_folder, "edstdp")

if not os.path.isfile(os.path.join(nest_gpu_src, "syn_model.h")):
    sys.exit(f"{nest_gpu_root} is not the root of NEST GPU: {nest_gpu_src}/syn_model.h does not exist.\n"
             f"(It must be the folder that contains CMakeLists.txt and src/, not the src/ folder itself.)")


# ---------------------------------------------------------------------------
# Helper function to make the edits
# ---------------------------------------------------------------------------
def replace_once(file_name, old, new, what):
    """
    In a file of NEST GPU, changes the text old into the text new. Most of the time
    new is old plus something added, so the original text stays.

    If new is already in the file, the edit was made before and nothing is done.
    """
    path = os.path.join(nest_gpu_src, file_name)
    with open(path) as file:
        text = file.read()

    if new in text:
        print(f"  skip   {file_name}: {what} (already there)")
    elif text.count(old) == 1:
        with open(path, "w") as file:
            file.write(text.replace(old, new))
        print(f"  done   {file_name}: {what}")
    else:
        sys.exit(f"  FAILED {file_name}: {what}: the text to look for was found {text.count(old)} times "
                 f"(expected 1). Look at this place by hand.")


# ---------------------------------------------------------------------------
# C++ code that is inserted at the end of the calibration (syn_model.cu)
# ---------------------------------------------------------------------------
# If any synapse group uses ED-STDP, it makes a table with one entry
# per group of neurons. Each entry says where the ATP of that group is in the GPU
# memory (address of the first neuron's ATP, distance to the next one, number of
# the first neuron). The update function uses the table to find the ATP of the
# target neuron from its number. Groups without an ATP variable (iaf neurons)
# get an empty entry: their synapses are not modulated.
ATP_TABLE_CODE = r'''

  // ED-STDP: tell device code where each node group keeps its ATP variable
  // Cost is null if there are no edstdp groups
  bool has_edstdp = false;
  for (unsigned int i = 0; i < syn_group_vect_.size(); i++) {
    if (syn_group_vect_[i]->type_ == i_edstdp_model) {
      has_edstdp = true;
    }
  }

  if (has_edstdp) {
    int n_ng = node_vect_.size(); // Number of neuron groups
    std::vector<edstdp_ns::ATPAccess> h_map(n_ng, edstdp_ns::ATPAccess{nullptr, 0, 0}); // Create table on host

    // Loop on neuron groups
    for (int i = 0; i < n_ng; i++) {
      BaseNeuron *bn = node_vect_[i]; // assumes index i is node group id 

      // Only proceds if neuron group has ATP
      if (bn->IsScalVar("ATP")) {
        float *p0 = bn->GetVarPt(0, "ATP");  // Pointer to ATP direction on first neuron of the group inside the group, not global
        int stride = (bn->n_node_ > 1) ? (int)(bn->GetVarPt(1, "ATP") - p0) : 0;  // size of each neuron
        h_map[i] = edstdp_ns::ATPAccess{p0, stride, bn->i_node_0_};  // Saves position of ATP on first neuron, stride, and pointer of first neuron
        std::cout << "ED-STDP: node group " << i << " has ATP (first node "
                  << bn->i_node_0_ << ", " << bn->n_node_ << " nodes, stride "
                  << stride << ")" << std::endl;
      }
    }
    edstdp_ns::ATPAccess *d_map;
    CUDAMALLOCCTRL("&d_ATPAccessMap", &d_map, n_ng * sizeof(edstdp_ns::ATPAccess));
    gpuErrchk(cudaMemcpy(d_map, h_map.data(), n_ng * sizeof(edstdp_ns::ATPAccess),
                         cudaMemcpyHostToDevice));
    edstdp_ns::ATPAccessInit<<<1, 1>>>(d_map);
    CUDASYNC;
  }'''



# ---------------------------------------------------------------------------
# Copy the new files
# ---------------------------------------------------------------------------
print("Copying edstdp.h / edstdp.cu")
for file_name in ["edstdp.h", "edstdp.cu"]:
    with open(os.path.join(edstdp_folder, file_name)) as file:
        text = file.read()
    with open(os.path.join(nest_gpu_src, file_name), "w") as file:
        file.write(text)
    print("  copied", file_name)


# ---------------------------------------------------------------------------
# syn_model.h: register the new synapse model
# ---------------------------------------------------------------------------
print("\nsyn_model.h")

replace_once("syn_model.h",
             old='#include "stdp.h"\n',
             new='#include "stdp.h"\n#include "edstdp.h"\n',
             what="include")

# New id for the model, in the list of synapse models (right after the STDP one)
replace_once("syn_model.h",
             old="  i_stdp_model,\n  N_SYN_MODELS\n",
             new="  i_stdp_model,\n  i_edstdp_model,\n  N_SYN_MODELS\n",
             what="enum entry")

# The function that updates a weight receives the target neuron as a 4th argument
replace_once("syn_model.h",
             old="float Dt) {",
             new="float Dt, int i_target) {",
             what="4th argument")

# When the update function is called for an ED-STDP group, use the ED-STDP rule
replace_once("syn_model.h",
             old="  case i_stdp_model:\n    stdp_ns::STDPUpdate(w, Dt, param);\n    break;\n",
             new="  case i_stdp_model:\n    stdp_ns::STDPUpdate(w, Dt, param);\n    break;\n"
                 "  case i_edstdp_model:\n    edstdp_ns::EDSTDPUpdate(w, Dt, param, i_target);\n    break;\n",
             what="switch case")

# Name by which Python asks for the model: CreateSynGroup("edstdp", ...)
replace_once("syn_model.h",
             old='{"", "test_syn_model", "stdp"}',
             new='{"", "test_syn_model", "stdp", "edstdp"}',
             what="model name")

# The class of the new model, written right after the class of the STDP model
replace_once("syn_model.h",
             old="  STDP() { _Init(); }\n\n  int Init() { return _Init(); }\n};",
             new="  STDP() { _Init(); }\n\n  int Init() { return _Init(); }\n};"
                 "\n\nclass EDSTDP : public SynModel {\n  int _Init();\n\npublic:\n"
                 "  EDSTDP() { _Init(); }\n  int Init() { return _Init(); }\n};",
             what="class")


# ---------------------------------------------------------------------------
# syn_model.cu: create the model, and build the ATP table
# ---------------------------------------------------------------------------
print("\nsyn_model.cu")

replace_once("syn_model.cu",
             old='#include "stdp.h"\n',
             new='#include "stdp.h"\n#include "edstdp.h"\n',
             what="include")

# When CreateSynGroup is called with "edstdp", create an ED-STDP group
replace_once("syn_model.cu",
             old="    STDP *stdp_group = new STDP;\n    syn_group_vect_.push_back(stdp_group);\n",
             new="    STDP *stdp_group = new STDP;\n    syn_group_vect_.push_back(stdp_group);\n"
                 "  } else if (model_name == syn_model_name[i_edstdp_model]) {\n"
                 "    EDSTDP *edstdp_group = new EDSTDP;\n    syn_group_vect_.push_back(edstdp_group);\n",
             what="CreateSynGroup branch")

# The ATP table (see ATP_TABLE_CODE above), written after the synapse groups are initialised
replace_once("syn_model.cu",
             old="SynGroupInit<<<1, 1>>>(d_SynGroupTypeMap, d_SynGroupParamMap);\n  CUDASYNC;",
             new="SynGroupInit<<<1, 1>>>(d_SynGroupTypeMap, d_SynGroupParamMap);\n  CUDASYNC;" + ATP_TABLE_CODE,
             what="ATP table")


# ---------------------------------------------------------------------------
# get_spike.h and rev_spike.h: pass the target neuron to SynapseUpdate
# ---------------------------------------------------------------------------
print("\ncall sites and the old extern")

# Where a presynaptic spike arrives (the weight is weakened). The text is the end of
# the call SynapseUpdate(syn_group, &(conn_struct.weight), ...).
replace_once("get_spike.h",
             old="-NESTGPUTimeResolution * Dt_int);",
             new="-NESTGPUTimeResolution * Dt_int, i_target);",
             what="presynaptic call")

# Where a postsynaptic spike happens (the weight is strengthened): same, with the name used there
replace_once("rev_spike.h",
             old="NESTGPUTimeResolution * Dt_int);",
             new="NESTGPUTimeResolution * Dt_int, target);",
             what="postsynaptic call")


# ---------------------------------------------------------------------------
# rev_spike.cu: delete the old declaration
# ---------------------------------------------------------------------------
# rev_spike.cu declares the old 3-argument version of the function, which would clash
# with the new one. That line is deleted.
old_declaration = "extern __device__ void SynapseUpdate(int syn_group, float *w, float Dt);\n"

with open(os.path.join(nest_gpu_src, "rev_spike.cu")) as file:
    text = file.read()

if old_declaration in text:
    with open(os.path.join(nest_gpu_src, "rev_spike.cu"), "w") as file:
        file.write(text.replace(old_declaration, ""))
    print("  done   rev_spike.cu: old 3-argument extern deleted")
else:
    print("  skip   rev_spike.cu: old extern not found (already deleted?)")


# ---------------------------------------------------------------------------
# CMakeLists.txt: add the new files to the list of files to compile
# ---------------------------------------------------------------------------
print("\nCMakeLists.txt")

# Each name is on its own line, at the start of the line
replace_once("CMakeLists.txt",
             old="\nstdp.cu\n",
             new="\nstdp.cu\nedstdp.cu\n",
             what="edstdp.cu next to stdp.cu")

replace_once("CMakeLists.txt",
             old="\nstdp.h\n",
             new="\nstdp.h\nedstdp.h\n",
             what="edstdp.h next to stdp.h")

print("\nNext: add_markers.py / generate_gpu.py / patch_generated.py, then build.sh.")
