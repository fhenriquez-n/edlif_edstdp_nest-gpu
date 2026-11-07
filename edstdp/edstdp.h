/*
 *  edstdp.h
 *
 *  Energy-dependent STDP (ED-STDP) synapse model for NEST GPU.
 *
 *  This file is part of a package that extends NEST GPU. It is modelled on
 *  NEST GPU's stdp.h and therefore distributed under the same licence.
 *
 *  Copyright (C) 2021 The NEST Initiative (NEST GPU, stdp.h)
 *  ED-STDP additions: the authors of this package. Modified from stdp.h, November 2026.
 *
 *  NEST GPU is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST GPU is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST GPU.  If not, see <http://www.gnu.org/licenses/>.
 *
 *  Energy-dependent STDP (ED-STDP). Modelled on stdp.h, after Jaras et al.
 *  (2025), PLoS Comput Biol: potentiation is scaled by
 *  exp(-eta * (ATP_h - A) / ATP_h), where A is the postsynaptic ATP (percent)
 *  read at the moment of the post spike. Depression is the same as in plain
 *  STDP and never reads ATP.
 *
 *  Notes:
 *   - Potentiation only for Dt > 0. Dt = 0 (simultaneous spikes) is depression.
 *   - Default parameters: alpha 0.5, eta 50. 
 */

#ifndef EDSTDP_H
#define EDSTDP_H
#include <cmath>
#include <stdint.h>
#include <string>

extern __device__ int16_t *NodeGroupMap; // defined in node_group.cu

namespace edstdp_ns {
enum ParamIndexes {
  i_tau_plus = 0,
  i_tau_minus,
  i_lambda,
  i_alpha,
  i_mu_plus,
  i_mu_minus,
  i_Wmax,
  i_eta,
  i_ATP_h,
  N_PARAM
};

const std::string edstdp_param_name[N_PARAM] = {
    "tau_plus", "tau_minus", "lambda", "alpha", "mu_plus",
    "mu_minus", "Wmax",      "eta",    "ATP_h"};

// Where the ATP state variable of one node group lives on the device.
struct ATPAccess {
  float *base;  // ATP of the first node of the group (nullptr: no ATP here)
  int stride;   // distance, in floats, between ATP of consecutive nodes
  int i_node_0; // index of the first node of the group
};

extern __device__ ATPAccess *ATPAccessMap; // indexed by node group

__global__ void ATPAccessInit(ATPAccess *atp_access_map);

__device__ __forceinline__ double EDSTDPGetATP(int i_target,
                                               double default_val) {
  ATPAccess acc = ATPAccessMap[NodeGroupMap[i_target]];
  if (acc.base == nullptr) {
    return default_val; // target has no ATP variable: no modulation
  }
  return (double)acc.base[(int64_t)(i_target - acc.i_node_0) * acc.stride];
}

__device__ __forceinline__ void EDSTDPUpdate(float *weight_pt, float Dt,
                                             float *param, int i_target) {
  double tau_plus = param[i_tau_plus];
  double tau_minus = param[i_tau_minus];
  double lambda = param[i_lambda];
  double alpha = param[i_alpha];
  double mu_plus = param[i_mu_plus];
  double mu_minus = param[i_mu_minus];
  double Wmax = param[i_Wmax];
  double eta = param[i_eta];
  double ATP_h = param[i_ATP_h];

  double w = *weight_pt;
  double w1;
  if (Dt > 0) {
    // potentiation: modulated by the postsynaptic energy (ATP is read only here)
    double atp = EDSTDPGetATP(i_target, ATP_h);
    double energy_factor = exp(-eta * (ATP_h - atp) / ATP_h);
    double fact = lambda * energy_factor * exp(-(double)Dt / tau_plus);
    w1 = w + fact * Wmax * pow(1.0 - w / Wmax, mu_plus);
  } else {
    // depression (Dt <= 0): unchanged, independent of ATP
    double fact = -alpha * lambda * exp((double)Dt / tau_minus);
    w1 = w + fact * Wmax * pow(w / Wmax, mu_minus);
  }

  w1 = w1 > 0.0 ? w1 : 0.0;
  w1 = w1 < Wmax ? w1 : Wmax;
  *weight_pt = (float)w1;
}
} // namespace edstdp_ns

#endif
