#ifndef RAJA_SparseKernel_HPP
#define RAJA_SparseKernel_HPP

#include "RAJA/config.hpp"
#include "RAJA/loopchain/SymExec.hpp"
#include "RAJA/loopchain/SymbolicSegment/SymbolicSegment.hpp"
#include "RAJA/pattern/kernel.hpp"
#include "RAJA/loopchain/KernelWrapper.hpp"
#include "RAJA/loopchain/Sparse/SparseView.hpp"

namespace RAJA {

template <typename ViewType>
std::vector<camp::idx_t> extract_sparse_access_order(std::vector<SymAccess> accesses, ViewType & view) {
  for(auto access : accesses) {
    if(access.isSparse && access.is_view(view)) {
      return access.argument_order();
    }
  }
  std::cerr << "did not find access.\n";
  return std::vector<camp::idx_t>();
}

//creates a kernel object using the same interface as the kernel function
template <typename KernelPol, camp::idx_t SymEvalLamIdx=0, typename SegmentTuple, typename ViewType, typename... Bodies>
auto make_sparse_kernel(const SegmentTuple & segment, ViewType& view, Bodies const &... bodies) {
  

  auto dense_knl = make_kernel<KernelPol>(segment, bodies...);
  auto accesses = dense_knl.template execute_symbolically<SymEvalLamIdx>();
  
  auto accessOrder = extract_sparse_access_order(accesses, view);
  
  auto sparseSpace = sparsify_iteration_space<KernelPol>(segment, view, accessOrder);
  return make_kernel<KernelPol>(sparseSpace, bodies...);
}
template <typename KernelPol, camp::idx_t SymEvalLamIdx=0, typename... Segments, typename ViewType, typename... Bodies, typename...T>

auto make_sparse_kernel_sym(tuple<T...> symVals, const tuple<Segments...> & segment, ViewType& view, Bodies const &... bodies) {
  

  auto dense_knl = make_kernel<KernelPol>(segment, bodies...);

  auto accesses = dense_knl.template execute_symbolically<SymEvalLamIdx>(symVals);
  
  auto accessOrder = extract_sparse_access_order(accesses, view);
  
  auto sparseSpace = sparsify_iteration_space_compressed<KernelPol>(segment, view, accessOrder);
  return make_kernel<KernelPol>(sparseSpace, bodies...);
}
}

#endif


