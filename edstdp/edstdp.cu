/*
 *  edstdp.cu
 *
 *  Energy-dependent STDP (ED-STDP) synapse model for NEST GPU.
 *
 *  This file is part of a package that extends NEST GPU. It is modelled on
 *  NEST GPU's stdp.cu and therefore distributed under the same licence.
 *
 *  Copyright (C) 2021 The NEST Initiative (NEST GPU, stdp.cu)
 *  ED-STDP additions: the authors of this package.
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
 *  Energy-dependent STDP (ED-STDP). Modelled on stdp.cu.
 *  Default parameters: additive rule (mu = 0), lambda 0.01, alpha 0.5, eta 50.
 */

#include "cuda_error.h"
#include "edstdp.h"
#include "ngpu_exception.h"
#include "syn_model.h"
#include <config.h>
#include <iostream>
#include <stdio.h>

using namespace edstdp_ns;

namespace edstdp_ns {
__device__ ATPAccess *ATPAccessMap;

__global__ void ATPAccessInit(ATPAccess *atp_access_map) {
  ATPAccessMap = atp_access_map;
}
} // namespace edstdp_ns

int EDSTDP::_Init() {
  type_ = i_edstdp_model;
  n_param_ = N_PARAM;
  param_name_ = edstdp_param_name;
  CUDAMALLOCCTRL("&d_param_arr_", &d_param_arr_, n_param_ * sizeof(float));
  SetParam("tau_plus", 20.0);
  SetParam("tau_minus", 20.0);
  SetParam("lambda", 0.01);
  SetParam("alpha", 0.5);
  SetParam("mu_plus", 0.0);
  SetParam("mu_minus", 0.0);
  SetParam("Wmax", 100.0);
  SetParam("eta", 50.0);
  SetParam("ATP_h", 100.0);

  return 0;
}
