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
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

/**
 * @file   semantic_tsdf_integrator_fast.cpp
 * @brief  Integrator of semantic and geometric information
 * @author Antoni Rosinol
 */

#include "kimera_semantics/semantic_tsdf_integrator_fast.h"

#include <list>
#include <memory>
#include <utility>

#include <voxblox/utils/timing.h>

#include "kimera_semantics/color.h"

namespace kimera {

FastSemanticTsdfIntegrator::FastSemanticTsdfIntegrator(
    const Config& config,
    const SemanticConfig& semantic_config,
    vxb::Layer<vxb::TsdfVoxel>* tsdf_layer,
    vxb::Layer<SemanticVoxel>* semantic_layer)
    : TsdfIntegratorBase(config, tsdf_layer),
      SemanticIntegratorBase(semantic_config, semantic_layer) {}

//这个函数整个代码中只被 integratePointCloud函数调用
void FastSemanticTsdfIntegrator::integrateSemanticFunction(
    const vxb::Transformation& T_G_C,
    const vxb::Pointcloud& points_C,
    const vxb::Colors& colors,
    const SemanticLabels& semantic_labels,
    const bool freespace_points,
    vxb::ThreadSafeIndex* index_getter) {
  DCHECK(index_getter != nullptr);

  size_t point_idx;
  while (index_getter->getNextIndex(&point_idx) &&
         (std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - integration_start_time_).count() < config_.max_integration_time_s * 1000000)) 
  {
    const vxb::Point& point_C = points_C[point_idx];
    const vxb::Color& color = colors[point_idx];
    const SemanticLabel& semantic_label = semantic_labels[point_idx];
    bool is_clearing;
    //1.剔除太远太近的点， 并根据label剔除可能的移动物体
    //和voxblox相比多了对动态语义的剔除
    if (!isPointValid(point_C, freespace_points, &is_clearing) ||
        !isSemanticLabelValid(semantic_label)) {
      continue;
    }

    const vxb::Point origin = T_G_C.getPosition();
    const vxb::Point point_G = T_G_C * point_C;
    // Checks to see if another ray in this scan has already started 'close'
    // to this location. If it has then we skip ray casting this point. We
    // measure if a start location is 'close' to another points by inserting
    // the point into a set of voxels. This voxel set has a resolution
    // start_voxel_subsampling_factor times higher then the voxel size.
    //2.获取global的id 和voxblox一致
    vxb::GlobalIndex global_voxel_idx;
    global_voxel_idx = vxb::getGridIndexFromPoint<vxb::GlobalIndex>(point_G, 
                                                                    config_.start_voxel_subsampling_factor * voxel_size_inv_);
    //start_voxel_approx_set_这个变量仅是用来判断地图中的voxel是否重复，如果两个点对应相同的voxel那么第二个voxel会直接continue                                                
    if (!start_voxel_approx_set_.replaceHash(global_voxel_idx)) {
      continue;
    }

    //4.构建ray_caster, 和voxblox完全一致
    static constexpr bool cast_from_origin = false;
    vxb::RayCaster ray_caster(origin,//载体中心
                              point_G,//3d点
                              is_clearing,
                              config_.voxel_carving_enabled,
                              config_.max_ray_length_m,
                              voxel_size_inv_,
                              config_.default_truncation_distance,
                              cast_from_origin);

    int64_t consecutive_ray_collisions = 0;

    vxb::Block<vxb::TsdfVoxel>::Ptr block = nullptr;
    vxb::BlockIndex block_idx;
    vxb::Block<SemanticVoxel>::Ptr semantic_block = nullptr;
    vxb::BlockIndex semantic_block_idx;
    while (ray_caster.nextRayIndex(&global_voxel_idx)) {
      // Check if the current voxel has been seen by any ray cast this scan.
      // If it has increment the consecutive_ray_collisions counter, otherwise
      // reset it. If the counter reaches a threshold we stop casting as the
      // ray is deemed to be contributing too little new information.
      if (!voxel_observed_approx_set_.replaceHash(global_voxel_idx)) {
        ++consecutive_ray_collisions;
      } else {
        consecutive_ray_collisions = 0;
      }
      if (consecutive_ray_collisions > config_.max_consecutive_ray_collisions) {
        break;
      }
      //voxblox对应的代码
      vxb::TsdfVoxel* voxel = allocateStorageAndGetVoxelPtr(global_voxel_idx, 
                                                            &block, &block_idx);

      //voxblox对应的代码， 3d点越远则权重越小                                                     
      const float weight = getVoxelWeight(point_C);

      //搜索 void TsdfIntegratorBase::updateTsdfVoxel(
      //隶属于 voxblox的代码
      updateTsdfVoxel(origin, point_G, global_voxel_idx, color, weight, voxel);

      //从下面开始的代码是专属于semantic-kimera代码
      //根据全局的voxel坐标，得到语义对应的block索引和内存地址
      SemanticVoxel* semantic_voxel = allocateStorageAndGetSemanticVoxelPtr( global_voxel_idx, 
                                                                            &semantic_block, &semantic_block_idx);
      //SemanticProbabilities 等价于 matrix<float,21,1>                                                                 
      SemanticProbabilities semantic_label_frequencies = SemanticProbabilities::Zero();
      CHECK_LT(semantic_label, semantic_label_frequencies.size());

      semantic_label_frequencies[semantic_label] += 1.0f;
      //这个函数是语义voxel独有的函数！！！！！
      //搜索 void SemanticIntegratorBase::updateSemanticVoxel(
      //详见算法实现文档
      updateSemanticVoxel(global_voxel_idx,
                          semantic_label_frequencies,
                          &mutexes_,
                          voxel,
                          semantic_voxel);
    }//while end 遍历光线透过的voxel
  }//end 遍历所有的点
}//end function integrateSemanticFunction

void FastSemanticTsdfIntegrator::integratePointCloud(const vxb::Transformation& T_G_C,//当前帧激点云的位姿
                                                      const vxb::Pointcloud& points_C,//当前帧激光点云
                                                      const vxb::Colors& colors,//每个点对因
                                                      const bool freespace_points) 
{
  SemanticLabels semantic_labels(colors.size());
  // TODO(Toni): parallelize with openmp
  //1.根据点云颜色获取语义标签
  for (size_t i = 0; i < colors.size(); i++) {
    const vxb::Color& color = colors[i];
    CHECK(semantic_config_.semantic_label_to_color_);
    semantic_labels[i] = semantic_config_.semantic_label_to_color_->getSemanticLabelFromColor(HashableColor(color.r, color.g, color.b, 255u));
  }

  vxb::timing::Timer integrate_timer("integrate/fast");
  CHECK_EQ(points_C.size(), colors.size());

  integration_start_time_ = std::chrono::steady_clock::now();

  static int64_t reset_counter = 0;
  //clear_checks_every_n_frames = 1
  if ((++reset_counter) >= config_.clear_checks_every_n_frames) {
    reset_counter = 0;
    start_voxel_approx_set_.resetApproxSet();
    voxel_observed_approx_set_.resetApproxSet();
  }

  std::unique_ptr<vxb::ThreadSafeIndex> index_getter(  vxb::ThreadSafeIndexFactory::get(config_.integration_order_mode,  points_C));

  std::list<std::thread> integration_threads;
  for (size_t i = 0; i < config_.integrator_threads; ++i) {
    integration_threads.emplace_back(
        &FastSemanticTsdfIntegrator::integrateSemanticFunction,//非常重要的函数！！！
        this,
        T_G_C,
        points_C,
        colors,
        semantic_labels,
        freespace_points,
        index_getter.get());
  }

  for (std::thread& thread : integration_threads) {
    thread.join();
  }

  integrate_timer.Stop();

  vxb::timing::Timer insertion_timer("inserting_missed_blocks");
  updateLayerWithStoredBlocks();//这两个函数和voxblox代码完全相同
  updateSemanticLayerWithStoredBlocks();//搜索 SemanticIntegratorBase::updateSemanticLayerWithStoredBlocks() {
  insertion_timer.Stop();
}

}  // Namespace kimera
