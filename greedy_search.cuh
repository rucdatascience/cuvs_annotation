/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "macros.cuh"
#include "priority_queue.cuh"
#include "vamana_structs.cuh"
#include <cub/cub.cuh>
#include <cuvs/neighbors/vamana.hpp>

#include <cuvs/distance/distance.hpp>
#include <raft/util/warp_primitives.cuh>
#include <rmm/resource_ref.hpp>

#include <chrono>
#include <cstdio>
#include <vector>

namespace cuvs::neighbors::vamana::detail {

/* @defgroup greedy_search_detail greedy search
 * @{
 */

/* Combines edge and candidate lists, removes duplicates, and sorts by distance
 * Uses CUB primitives, so needs to be templated. Called with Macros for supported sizes above */
template <typename accT, typename IdxT, int CANDS>//模板函数，参数变量依次为距离值的类型，索引的类型，候选点的总数
//GPU 上运行的高性能函数模板，用于对访问过的候选点按照距离进行排序
__forceinline__ __device__ void sort_visited(//GPU设备函数，强制内联
  QueryCandidates<IdxT, accT>* query,//传入指向一个模板结构体 QueryCandidates实例的指针
  typename cub::BlockMergeSort<DistPair<IdxT, accT>, 32, (CANDS / 32)>::TempStorage* sort_mem)//cub::BlockMergeSort：一个在cub中的高效排序器，参数模板中的参数依次为：索引和距离组成的结构体，一个block有多少线程，一个线程中处理元素的数量
  //BlockMergeSort是排序器所需要的临时内存的储存位置，sort_num是指向此内存的指针
{
  const int ELTS   = CANDS / 32;//每个线程处理的元素个数
  using BlockSortT = cub::BlockMergeSort<DistPair<IdxT, accT>, 32, ELTS>;//对排序器重命名
  
  DistPair<IdxT, accT> tmp[ELTS];//没有__shared__ 或 __global__前缀，证明该数组是每个线程的私有数组
  for (int i = 0; i < ELTS; i++) {
    tmp[i].idx  = query->ids[ELTS * threadIdx.x + i];
    tmp[i].dist = query->dists[ELTS * threadIdx.x + i];
  }
  //每个线程将自己所负责的候选点存入私有数组temp，方便后续排序
  __syncthreads();//线程块内的同步屏障，当线程块中所以线程都跑完在执行下一步
  BlockSortT(*sort_mem).Sort(tmp, CmpDist());//排序
  __syncthreads();//再同步

  for (int i = 0; i < ELTS; i++) {
    query->ids[ELTS * threadIdx.x + i]   = tmp[i].idx;
    query->dists[ELTS * threadIdx.x + i] = tmp[i].dist;
  }//拷贝结果
  __syncthreads();
}


namespace {

/********************************************************************************************
  GPU kernel to perform a batched GreedySearch on a graph. Since this is used for
  Vamana construction, the entire visited list is kept and stored within the query_list.
  Input - graph with edge lists, dataset vectors, query_list_ptr with the ids of dataset
          vectors to be searched. All inputs, including dataset,  must be device accessible.

  Output - the id and dist lists in query_list_ptr will be updated with the nodes visited
           during the GreedySearch.
**********************************************************************************************/
template <typename T,//向量中每个元素的数据类型
          typename accT,//累加器类型
          typename IdxT     = uint32_t,
          typename Accessor = raft::host_device_accessor<std::experimental::default_accessor<T>,
                                                         raft::memory_type::host>>//一个能同时访问cpu和gpu的内存读取器
__global__ void GreedySearchKernel(
  raft::device_matrix_view<IdxT, int64_t> graph,//用二维数组表示的邻接图，graph[i][j] 表示第 i 个点的第 j 个邻居的 ID
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,//常量二维向量矩阵，表示原始向量数据集[n, dim]
  void* query_list_ptr,//结果存储的地址 为num_queries*topk型的矩阵
  int num_queries,//结果的大小
  int medoid_id,//贪心搜索的起点
  int topk,//每个 query 需要找几个最近邻
  cuvs::distance::DistanceType metric,//数量计算
  int max_queue_size,//候选节点的最大数量
  int sort_smem_size)//每个线程用于排序的共享内存大小
{
  int n      = dataset.extent(0);//向量数
  int dim    = dataset.extent(1);//向量维数
  int degree = graph.extent(1);//一个点所能拥有的最大边数

  QueryCandidates<IdxT, accT>* query_list =
    static_cast<QueryCandidates<IdxT, accT>*>(query_list_ptr);//空指针转换

  static __shared__ int topk_q_size;//维护的 topk 候选集合的实际大小
  static __shared__ int cand_q_size;//待访问节点的数量
  static __shared__ accT cur_k_max;//topk候选集合中距离的最大值
  static __shared__ int k_max_idx;//最大值所在的索引

  static __shared__ Point<T, accT> s_query;//存储被查询最近邻的点

  union ShmemLayout {
    // All blocksort sizes have same alignment (16)
    typename cub::BlockMergeSort<DistPair<IdxT, accT>, 32, 1>::TempStorage sort_mem;
    T coords;
    Node<accT> topk_pq;
    int neighborhood_arr;
    DistPair<IdxT, accT> candidate_queue;
  };

  int align_padding = (((dim - 1) / alignof(ShmemLayout)) + 1) * alignof(ShmemLayout) - dim;
//内存对齐
  // Dynamic shared memory used for blocksort, temp vector storage, and neighborhood list
  extern __shared__ __align__(alignof(ShmemLayout)) char smem[];

  size_t smem_offset = sort_smem_size;  // temp sorting memory takes first chunk
//为共享内存 smem 中排在排序缓冲区之后的部分计算起始偏移量。
  T* s_coords = reinterpret_cast<T*>(&smem[smem_offset]);//用来存放查询向量的数据

  smem_offset += (dim + align_padding) * sizeof(T);//更新共享内存偏移指针，为下一个数据结构的共享内存空间腾出位置

  Node<accT>* topk_pq = reinterpret_cast<Node<accT>*>(&smem[smem_offset]);//用来储存topk的优先队列
  smem_offset += topk * sizeof(Node<accT>);//更新共享内存的偏移量，防止冲突

  int* neighbor_array = reinterpret_cast<int*>(&smem[smem_offset]);//分配共享内存来储存当前节点的邻居的id列表
  smem_offset += degree * sizeof(int);//更新共享内存的偏移量

  DistPair<IdxT, accT>* candidate_queue_smem =
    reinterpret_cast<DistPair<IdxT, accT>*>(&smem[smem_offset]);//构造候选集合的队列（每个点包括：ID 和 距离）

  s_query.coords = s_coords;//查询向量在共享内存中的位置
  s_query.Dim    = dim;//维度
  //核函数中可以直接用 s_query 来计算距离

  PriorityQueue<IdxT, accT> heap_queue;//优先队列找topk

  if (threadIdx.x == 0) {
    heap_queue.initialize(candidate_queue_smem, max_queue_size, &cand_q_size);
  }
  //让第一个线程负责初始化优先队列
  //参数依次是：储存地址，最大容量，队列当前对象的数量

  static __shared__ int num_neighbors;//正在处理节点的邻居数目

  for (int i = blockIdx.x; i < num_queries; i += gridDim.x) {//每次循环是一个线程块参与计算
    //当前线程块在网格（grid）中的索引，是从0开始的|需要处理的任务总数（这里是查询向量的个数)|网格中线程块的总数
    __syncthreads();

    // resetting visited list
    query_list[i].reset();

    // storing the current query vector into shared memory
    update_shared_point<T, accT>(&s_query, &dataset(0, 0), query_list[i].queryId, dim);

    if (threadIdx.x == 0) {
      topk_q_size = 0;
      cand_q_size = 0;
      s_query.id  = query_list[i].queryId;
      cur_k_max   = 0;
      k_max_idx   = 0;
      heap_queue.reset();
    }

    __syncthreads();

    Point<T, accT>* query_vec;//通过 query_vec 就能访问查询向量的数据

    // Just start from medoid every time, rather than multiple set_ups
    query_vec        = &s_query;
    query_vec->Dim   = dim;
    const T* medoid  = &dataset((size_t)medoid_id, 0);
    accT medoid_dist = dist<T, accT>(query_vec->coords, medoid, dim, metric);
    //给查询向量 query_vec 赋值并计算它和某个“medoid”点的距离

    if (threadIdx.x == 0) { heap_queue.insert_back(medoid_dist, medoid_id); }//贪心搜索的初始化
    __syncthreads();

    while (cand_q_size != 0) {
      __syncthreads();

      int cand_num;
      accT cur_distance;
      if (threadIdx.x == 0) {
        Node<accT> test_cand;
        DistPair<IdxT, accT> test_cand_out = heap_queue.pop();//从候选堆里取出当前距离最近的点，它的编号是 idx，距离是 dist
        test_cand.distance                 = test_cand_out.dist;
        test_cand.nodeid                   = test_cand_out.idx;
        cand_num                           = test_cand.nodeid;
        cur_distance                       = test_cand_out.dist;
      }
      __syncthreads();

      cand_num = raft::shfl(cand_num, 0);//从 线程 0 拿出变量 a 的值，然后 复制一份给 这个 warp 里所有的线程

      __syncthreads();

      if (query_list[i].check_visited(cand_num, cur_distance)) { continue; }

      cur_distance = raft::shfl(cur_distance, 0);//广播

      // stop condition for the graph traversal process
      //进行枝剪
      bool done      = false;
      bool pass_flag = false;
      
      if (topk_q_size == topk) {//如果topk集合满了
        // Check the current node with the worst candidate in top-k queue
        if (threadIdx.x == 0) {
          if (cur_k_max <= cur_distance) { done = true; }
        }//只让0号线程处理，如果说这个点的距离比大顶堆的堆顶距离还大，就没必要扩展它的邻居了
  

        done = raft::shfl(done, 0);//向其他线程广播该点是否值得扩展
        if (done) {
          if (query_list[i].size < topk) {
            pass_flag = true;
          }//如果有重复点导致大顶堆实际上没满则继续加入

          else if (query_list[i].size >= topk) {
            break;//真满了就直接跳出循环
          }
        }
      }

      // The current node is closer to the query vector than the worst candidate in top-K queue, so
      // enquee the current node in top-k queue
      Node<accT> new_cand;
      new_cand.distance = cur_distance;
      new_cand.nodeid   = cand_num;//建立一个新的候选节点

      if (check_duplicate(topk_pq, topk_q_size, new_cand) == false) {
        if (!pass_flag) {//加入大顶堆
          parallel_pq_max_enqueue<accT>(
            topk_pq, &topk_q_size, topk, new_cand, &cur_k_max, &k_max_idx);

          __syncthreads();
        }
      } else {
        // already visited
        continue;
      }

      num_neighbors = degree;//初始化邻居的个数为最大值
      __syncthreads();

      for (size_t j = threadIdx.x; j < degree; j += blockDim.x) {
        // Load neighbors from the graph array and store them in neighbor array (shared memory)
        neighbor_array[j] = graph(cand_num, j);
        if (neighbor_array[j] == raft::upper_bound<IdxT>())//遇到了哨兵值
          atomicMin(&num_neighbors, (int)j);  // warp-wide min to find the number of neighbors
      }//动态裁剪邻居的个数

      // computing distances between the query vector and neighbor vectors then enqueue in priority
      // queue.
      enqueue_all_neighbors<T, accT, IdxT>(
        num_neighbors, query_vec, &dataset(0, 0), neighbor_array, heap_queue, dim, metric);
    //对每个邻居计算距离并尝试加入大顶堆
      __syncthreads();

    }  // End cand_q_size != 0 loop

    bool self_found = false;
    // Remove self edges
    for (int j = threadIdx.x; j < query_list[i].size; j += blockDim.x) {
      if (query_list[i].ids[j] == query_vec->id) {
        query_list[i].dists[j] = raft::upper_bound<accT>();
        query_list[i].ids[j]   = raft::upper_bound<IdxT>();
        self_found             = true;  // Flag to reduce size by 1
      }
    }//查询点本身从查询点的邻居列表（即top-k结果）里移除

    for (int j = query_list[i].size + threadIdx.x; j < query_list[i].maxSize; j += blockDim.x) {
      query_list[i].ids[j]   = raft::upper_bound<IdxT>();
      query_list[i].dists[j] = raft::upper_bound<accT>();
    }//清理尾部无效空间，保证邻居列表中只有size个有效邻居

    __syncthreads();
    if (self_found) query_list[i].size--;//剔除自身

    SEARCH_SELECT_SORT(topk);//重新对topk排序
  }

  return;
}

}  // namespace

/**
 * @}
 */

}  // namespace cuvs::neighbors::vamana::detail
