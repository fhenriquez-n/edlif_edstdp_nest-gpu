# EdLIF neuron and ED-STDP synapse for NEST GPU

## 1. What is this package?

[NEST GPU](https://github.com/nest/nest-gpu) is a simulator of spiking neural networks that runs on
NVIDIA GPUs. This package adds two things to it:

- **The EdLIF neuron** (energy-dependent leaky integrate-and-fire). It keeps track of its own energy
  (ATP): spiking and receiving spikes use ATP, and the neuron slowly makes it again. There are two
  versions: one with exponential synaptic currents and one with alpha-shaped synaptic currents. They are
  written in NESTML and compiled into NEST GPU.
- **The ED-STDP synapse** (energy-dependent STDP). It is a normal STDP synapse, except that how much a
  synapse gets stronger depends on the ATP of the neuron that receives the spike.

NEST GPU does not have either of them, so this package does **not** contain a copy of NEST GPU. It
contains:

- **scripts** that take a clean NEST GPU, add the two pieces, and compile it;
- the **files** the scripts need (the NESTML models and the ED-STDP code);
- a short **example** that uses both.

Only the GPU version of NEST is covered. Nothing here is for NEST (CPU).

## 2. Versions and where it was tested

| | |
|---|---|
| NEST GPU | `nest/nest-gpu`, commit `ac212d5444ac4b6f48fb9d751d8eb44340d75f33` (2026-10-02, merge of PR #143, version 2.0-100). `get_nestgpu.sh` downloads exactly this commit |
| NESTML | 9.0.0 |
| CUDA toolkit | 12.4 |
| License | GPL-2.0-or-later (it is derived from NEST GPU). See `LICENSE` |

**Where it was tested:** on the Patagon cluster (UACH), in the container `cuda-dev-12.4`, on a node with an
NVIDIA A100 GPU, with NESTML 9.0.0 in its own Python environment. There:

- the build ended with `BUILD OK`;
- the example ended with `EXAMPLE OK`;


**Not tested:** other NEST GPU or NESTML versions, other CUDA versions, MPI or several GPUs.

## 3. Requirements

You need all of this on the machine where you build. Below the table there is a block of commands to check
it.

| What | Why | Notes |
|---|---|---|
| NVIDIA GPU, and its driver | To compile and to run | NEST GPU links against the driver library (`libcuda`), so **build on a machine that has the GPU**. On a cluster, that means inside a GPU job |
| CUDA toolkit (`nvcc`) | To compile | 12.4 was tested. It needs the compiler `nvcc` and the libraries `cudart` and `curand` |
| C and C++ compilers (`gcc`, `g++`), `make` | To compile | The `g++` version must be one your CUDA version supports |
| CMake 3.17 or newer | To configure the build | |
| MPI with its development files (for example Open MPI) | To compile | NEST GPU asks for MPI by default, and CMake stops if it is missing. Even if you do not use MPI, you need it installed |
| Python 3.11 or newer, with its development files (`Python.h`) | NEST GPU needs 3.8 or newer. NESTML asks for 3.11 or newer | On Debian and Ubuntu the development files come in the package `python3-dev` |
| `git` and access to GitHub | To download NEST GPU | Or a local copy of NEST GPU: see `NESTGPU_REPO` in section 4 |
| Python packages | NESTML 9.0.0, numpy, matplotlib | Listed in `requirements.txt`. See the next subsection |

Check what you have (these commands only read, they change nothing):

```bash
nvidia-smi              # the GPU and the driver
nvcc --version          # the CUDA compiler
gcc --version           # the C compiler
g++ --version           # the C++ compiler
make --version          # make
cmake --version         # must be 3.17 or newer
mpicc --version         # MPI
git --version
python3 --version       # must be 3.11 or newer
python3 -c "import sysconfig, os; print(os.path.exists(os.path.join(sysconfig.get_paths()['include'], 'Python.h')))"
                        # prints True if Python.h (the development files) is there
```

These requirements come from the `CMakeLists.txt` files of the pinned NEST GPU commit, from
[its installation page](https://nest-gpu.readthedocs.io/en/latest/installation/index.html), and from
[the installation page of NESTML](https://nestml.readthedocs.io/en/latest/installation.html).

### Installing the Python packages in an environment
Run this once, in the folder of this README:

```bash
python3 -m venv .venv                   # creates the environment in the folder .venv
source .venv/bin/activate               # turns it on (the prompt shows "(.venv)")
pip install -r requirements.txt         # installs the packages (needs internet)
```

- **Turn it on in every new shell** (`source .venv/bin/activate`) before you build and before you run
  anything. Use the same environment for the build and for the simulations: NEST GPU installs its Python
  module in a folder named after the Python version it found when it was compiled (for example
  `python3.11`), so the Python that runs the simulation must have the same version.
- On Debian and Ubuntu, `python3 -m venv` may ask you to install `python3-venv` first.
- If you already have NEST GPU built and only want to run the example, `pip install numpy matplotlib` is
  enough. NESTML is only needed to build.
- The module `nestgpu` is **not** installed by `pip`. Step 6 of section 5 installs it.

## 4. Environment variables

**To build, you do not need to set any**: all of them have a default. Set one only if you want to change
it.

| Variable | Used by | Default | What it is | When to change it |
|---|---|---|---|---|
| `NESTGPU_WORK` | all the scripts | `work/`, next to this README | The one folder where everything is created: `src/` (NEST GPU), `build/`, `install/` and `nestml_target/` | If you want it somewhere else, for example a disk with more space |
| `GPU_ARCH` | `build.sh` | `80` | The CUDA architecture the library is compiled for. 80 is the A100, 89 is the L40 | **Almost always.** Set it to your GPU. The library only runs on the architecture it was compiled for. To find yours: `nvidia-smi --query-gpu=compute_cap --format=csv,noheader` prints a number like `8.9`; remove the dot (`89`). If your driver does not know that query, look your GPU up in [NVIDIA's list](https://developer.nvidia.com/cuda-gpus) |
| `JOBS` | `build.sh` | `8` | How many files are compiled at the same time (`make -j`) | If the machine has few cores, or many more |
| `NESTGPU_REPO` | `get_nestgpu.sh` | `https://github.com/nest/nest-gpu.git` | Where NEST GPU is downloaded from | If you have no access to GitHub and have a mirror or a local copy |

Example:

```bash
GPU_ARCH=89 JOBS=16 bash scripts/build.sh
```

**To run a simulation, two variables are needed**. You do not type them: this line sets both (the file
`nestgpu_vars.sh` is written by NEST GPU when it is installed):

```bash
source work/install/bin/nestgpu_vars.sh
```

| Variable | What it is |
|---|---|
| `NESTGPU_LIB` | The path to the NEST GPU library (`libnestgpukernel.so`). Without it, `import nestgpu` fails |
| `PYTHONPATH` | The folder of the Python module `nestgpu` is added to it |

They last only in the shell where you ran `source`. In a new shell, run it again. If you run your
simulations through a job system that opens a new shell for each job (like `srun` or `sbatch`), put the
`source` line inside the job, before the Python command. Building does not need these two.

**Do not set `NEST_GPU`.** NESTML reads this variable to find NEST GPU. `generate_gpu.py` sets it by itself
and ignores any value you had.

If you read the install page of NEST GPU, you will see `SOURCE_DIR`, `BUILD_DIR`, `INSTALL_DIR` and `NAME`.
They are only shortcuts that the page uses to write its commands. Neither CMake nor NEST GPU reads them,
and this package does not need them. Here, the same three folders are always inside `NESTGPU_WORK`:
`src/`, `build/` and `install/`.

## 5. Steps to use the package

Run these commands from the folder of this README, in this order. Everything that is created goes into
one folder (`work/`, or `NESTGPU_WORK`). Nothing outside it is touched.

```bash
bash scripts/get_nestgpu.sh                  # 1. download NEST GPU (the pinned commit) into work/src
python3 scripts/add_markers.py               # 2. add marker comments where NESTML will write
python3 scripts/apply_edstdp.py              # 3. add the ED-STDP synapse
python3 scripts/generate_gpu.py              # 4. NESTML turns models/*.nestml into GPU code (needs NESTML)
python3 scripts/patch_generated.py           # 5. fix three bugs in that generated code
GPU_ARCH=89 bash scripts/build.sh            # 6. compile and install (needs nvcc and a GPU: see section 3)
source work/install/bin/nestgpu_vars.sh      # 7. now `import nestgpu` works in this shell
```

Notes:

- In step 6, use the `GPU_ARCH` of **your** GPU (see section 4). `89` is only an example.
- Step 1 needs internet. Steps 2 to 5 only edit files, they do not need a GPU. Step 6 needs the GPU, so
  on a cluster run it inside a GPU job. The last lines of its output say `BUILD OK` when it worked.
- If a step stops with a message, read it: it says what is missing or which step to run first.
- You can run steps 2 to 5 again: a step that is already done is skipped. Run step 5 again after every
  run of step 4, because step 4 writes the generated files again.
- To start again from a clean NEST GPU: `rm -rf work/src` (or `$NESTGPU_WORK/src`), then repeat from
  step 1.
- Steps 2 to 5 work on `work/src`. Each of them also accepts another NEST GPU folder as an argument, for
  example `python3 scripts/add_markers.py ~/my_nestgpu`. `get_nestgpu.sh` and `build.sh` take no argument:
  they use `work/` (or `NESTGPU_WORK`).

## 6. How to use the example

`examples/edlif_edstdp_example.py` is the shortest script that uses both pieces.
It does this:

1. creates 500 EdLIF neurons (400 excitatory, 100 inhibitory);
2. connects all neurons to all neurons; the excitatory-to-excitatory synapses use ED-STDP;
3. simulates one second;
4. prints the firing rates, the ATP and the weights of the plastic synapses;
5. draws a figure and saves it as `edlif_edstdp_example.png` in the current folder (it is skipped if
   `matplotlib` is not installed);
6. prints three lines that say `yes` or `NO`: did the neurons spike, is the ATP a valid number below 100,
   did ED-STDP change the weights.

You need the install of section 5 (step 6), a GPU, and numpy (and matplotlib, for the figure).

```bash
source .venv/bin/activate                    # the Python environment of section 3
source work/install/bin/nestgpu_vars.sh      # tells Python where NEST GPU is
python3 examples/edlif_edstdp_example.py
```

If everything went well it ends with `EXAMPLE OK`. If one of the three lines says `NO`, it ends with
`EXAMPLE FAILED`. In a job system, run the same three lines inside the job.


## 7. What changed in NEST GPU

The scripts make all these changes for you. This section tells you what they are, in case you want to
check them, or something fails with a new NEST GPU version.

### Files edited and added (steps 2 and 3)

Eight NEST GPU files are edited and two are added. Every edit must find its place exactly once, or the
script stops with a message (and you can run it again after fixing the cause: what is done is skipped).

| File in `work/src/src/` | Change |
|---|---|
| `edstdp.h`, `edstdp.cu` | **New.** The ED-STDP rule and its parameters (the originals are in `edstdp/`) |
| `syn_model.h` | Registers the `edstdp` synapse. `SynapseUpdate` gets a fourth argument: the target neuron |
| `syn_model.cu` | Creates the `EDSTDP` synapse group. At the end of `SynGroupCalibrate` it builds a table that tells the GPU where each neuron group keeps its `ATP` variable |
| `get_spike.h`, `rev_spike.h` | Pass the target neuron to `SynapseUpdate` (when a spike is sent: depression; when the target spikes: strengthening) |
| `rev_spike.cu` | Removes the old declaration of `SynapseUpdate` (3 arguments) |
| `neuron_models.h`, `neuron_models.cu` | Marker comments: NESTML writes the registration of the new neuron models between them |
| `CMakeLists.txt` | Compiles `edstdp.cu`. It also gets a pair of marker comments for NESTML |

The table of `ATP` assumes that the position of a neuron group in NEST GPU's internal list is its group
number. If a build fails, or ED-STDP does not see the ATP, look at the block `ATP_TABLE_CODE` in
`scripts/apply_edstdp.py`, and at what it uses: `IsScalVar`, `GetVarPt`, `n_node_` and `i_node_0_` in
`base_neuron.h`, and `node_vect_` in `nestgpu.h`.

### The generated neuron models (steps 4 and 5)

`scripts/generate_gpu.py` runs NESTML 9.0.0 on `models/*.nestml` with the numeric solver. NESTML writes
four new files (`edlif_psc_exp_nestml.h/.cu` and `edlif_psc_alpha_nestml.h/.cu`) into `work/src/src/` and
registers the models between the marker comments of `CMakeLists.txt`, `neuron_models.h` and
`neuron_models.cu`. **If a marker is missing, NESTML does not stop: it cuts the file.** That is why step 2
adds them and why step 4 checks them first.

NESTML also tries to compile NEST GPU by itself after generating. Nothing here uses that result, so the
script turns it off (`--with-nestml-build` turns it back on). The compilation that counts is step 6.

The code NESTML generates for this NEST GPU version has three bugs. `scripts/patch_generated.py` fixes
them with plain text replacements, and stops with a message if the text it expects is not there:

1. **Spike weights 1000 times too small.** The generator writes `0.001 *` in front of every spike input.
   (NESTML's own NEST GPU test model has the same factor.)
2. **Garbage in four state variables** (`r`, `alp`, `bet`, `homeostatic_s_enabled`). They have no
   equation, and the generator never sets their derivative, so the integrator reads random memory and
   `ATP` becomes NaN. The derivative is set to 0.
3. **`I_stim` is never initialised.** It is set to 0 (you can still change it with `SetNeuronParam`).

The script keeps no copy of the unfixed file. To see what NESTML wrote, run step 4 again.

## What is in this folder

```
models/            edlif_psc_exp.nestml, edlif_psc_alpha.nestml
docs/              modelos_nestml.md  (Spanish: the NESTML models against the originals)
edstdp/            edstdp.h, edstdp.cu  (copied into NEST GPU by apply_edstdp.py)
examples/          edlif_edstdp_example.py
scripts/           get_nestgpu.sh  add_markers.py  apply_edstdp.py  generate_gpu.py
                   patch_generated.py  build.sh
requirements.txt   the Python packages: NESTML 9.0.0, numpy, matplotlib
work/              created by the scripts (NEST GPU, build, install). Not part of the package
```
