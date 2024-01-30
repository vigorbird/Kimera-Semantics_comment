// NOTE: Most code is derived from voxblox: github.com/ethz-asl/voxblox
// Copyright (c) 2016, ETHZ ASL
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of voxblox nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

/**
 * @file   semantic_tsdf_server.cpp
 * @brief  Semantic TSDF Server to interface with ROS
 * @author Antoni Rosinol
 */

#include "kimera_semantics_ros/semantic_tsdf_server.h"

#include <glog/logging.h>

#include <voxblox_ros/ros_params.h>

#include <kimera_semantics/semantic_tsdf_integrator_factory.h>

#include "kimera_semantics_ros/ros_params.h"

namespace kimera {

SemanticTsdfServer::SemanticTsdfServer(const ros::NodeHandle& nh,
                                       const ros::NodeHandle& nh_private)
    : SemanticTsdfServer(nh,
                         nh_private,
                         vxb::getTsdfMapConfigFromRosParam(nh_private),
                         vxb::getTsdfIntegratorConfigFromRosParam(nh_private),
                         vxb::getMeshIntegratorConfigFromRosParam(nh_private)) {
}

SemanticTsdfServer::SemanticTsdfServer(
    const ros::NodeHandle& nh,
    const ros::NodeHandle& nh_private,
    const vxb::TsdfMap::Config& config,
    const vxb::TsdfIntegratorBase::Config& integrator_config,
    const vxb::MeshIntegratorConfig& mesh_config) : vxb::TsdfServer(nh, nh_private, config, integrator_config, mesh_config),
                                                  semantic_config_(getSemanticTsdfIntegratorConfigFromRosParam(nh_private)),
                                                  semantic_layer_(nullptr) {

  /// Semantic layer
  semantic_layer_.reset(new vxb::Layer<SemanticVoxel>(  config.tsdf_voxel_size, 
                                                        config.tsdf_voxels_per_side));
  /// Replace the TSDF integrator by the SemanticTsdfIntegrator
  //getSemanticTsdfIntegratorTypeFromRosParam函数返回的是string
  tsdf_integrator_ = SemanticTsdfIntegratorFactory::create( 
                                                            getSemanticTsdfIntegratorTypeFromRosParam(nh_private),
                                                            integrator_config,
                                                            semantic_config_,
                                                            tsdf_map_->getTsdfLayerPtr(),
                                                            semantic_layer_.get());

  CHECK(tsdf_integrator_);
}

}  // Namespace kimera
