// Contains functions that extract isl polyhedral sets from kernel objects.

#ifndef RAJA_sparse_view_HPP
#define RAJA_sparse_view_HPP
#include <cassert>
#include <cmath>
#include <unordered_set>
#include <random>
#include "RAJA/loopchain/Sparse/SparseImpl.hpp"
#include "RAJA/loopchain/Sparse/COO.hpp"
#include "RAJA/loopchain/Sparse/DIAG.hpp"
#include "RAJA/loopchain/Sparse/CSR.hpp"

namespace RAJA {

template <camp::idx_t NumDims>
struct SparseSymbolicSegmentLead;

template <camp::idx_t NumDims>
struct SparseSymbolicSegmentFollow;

template <typename ElmType, size_t NumDims, typename FormatType>
struct  SparseView {
 
 
  FormatType impl;

  SparseView(FormatType i) : impl(i) {}

  void* get_data() { return impl.get_data();}
  
  ElmType & val(idx_t i) {return impl.val[i];}
  std::vector<idx_t> & dim(idx_t i) {return impl.dim(i);}
  auto size() {return impl.size();}
  template <typename... Args>
  SymAccessList operator () (SymIterator symIterator, Args... args) 
  {
    std::vector<SymIterator> allIterators{{symIterator, args...}};

    SymAccess thisAccess = SymAccess(impl.get_data(), impl.get_permutation(), allIterators);
    thisAccess.isSparse = true;
    SymAccessList l = SymAccessList(thisAccess);
    return l ;
  }
  template <typename...Indices>
  RAJA_INLINE
  ElmType  operator() (camp::idx_t i, Indices... is) {
    auto tuple = make_tuple(i,is...);
        return impl.operator()(tuple);
  }

   
  auto get_sparse_iteration_space(std::vector<camp::idx_t> dimOrder) {
    auto outerDim = new SparseSymbolicSegmentLead<NumDims>(*this, dimOrder[0]);
    auto innerDim = new SparseSymbolicSegmentFollow<NumDims>(*this, dimOrder[1], outerDim);
    return camp::tie(*outerDim, *innerDim);
  }

  size_t get_hits() {return impl.get_hits();}
  size_t get_misses() {return impl.get_misses();}
  void reset_counters() {impl.reset_counters();}
  double get_miss_rate() { return impl.get_miss_rate();}
  double get_hit_rate() { return impl.get_hit_rate();}
  void set_permutation(std::vector<idx_t> perm) {impl.set_permutation(perm);}

 
}; // SparseView<NumDims>


template <typename Tuple, camp::idx_t...Is>
auto tuple_to_vector(Tuple t, camp::idx_seq<Is...>) {
  auto vec = std::vector<std::vector<camp::idx_t>>{{camp::get<Is>(t)...}};
  return vec;
}

template <camp::idx_t...Is> 
auto seq_to_vector(camp::idx_seq<Is...>) {
  return std::vector<idx_t>{{Is...}};
}

template <typename ElmType, idx_t NumDims, template <typename, idx_t> typename Format, typename...Vectors>
auto make_sparse_view_permuted(Vectors...vecs) {
  auto tuple = camp::make_tuple(vecs...);

  //static_assert(NumDims == sizeof...(Vectors) - 2, 
    //"Mismatch between number of dimensions and number of provided vectors.\n");
 
  auto val = camp::get<NumDims>(tuple);
  auto permutation = camp::get<NumDims+1>(tuple);
  auto indices = tuple_slice<0,NumDims>(tuple);
  
  auto indicesVector = tuple_to_vector(indices, idx_seq_for(indices));

  Format<ElmType, NumDims> impl(indicesVector, val, permutation);  

  return SparseView<ElmType, NumDims, Format<ElmType, NumDims>>(impl);

}

template <typename ElmType, idx_t NumDims, template <typename, idx_t> typename Format, typename...Vectors>
auto make_sparse_view(Vectors...vecs) {
  auto tuple = camp::make_tuple(vecs...);
  static_assert( NumDims == sizeof...(Vectors) - 1, "Mismatch between NumDims and number of vectors provided.");

  auto seq = idx_seq_from_to<0,NumDims>();
  auto perm = seq_to_vector(seq);
  return make_sparse_view_permuted<ElmType, NumDims, Format>(vecs..., perm);
}

template <typename ElmType, idx_t NumDims, template <typename, idx_t> typename Format, idx_t...Is>
auto make_random_sparse_view_helper(std::vector<std::vector<idx_t>> dims, std::vector<ElmType> vals,camp::idx_seq<Is...>) {
  return make_sparse_view<ElmType, NumDims, Format>(dims[Is]..., vals);
}
template <typename ElmType, idx_t NumDims, template <typename, idx_t> typename Format = COO >
auto make_random_sparse_view(idx_t N, double density, bool fullDiagonal=false) {

  std::vector<std::vector<idx_t>> dims;
  for(int i = 0; i < NumDims; i++) {
    dims.push_back(std::vector<idx_t>());
  }
  std::vector<ElmType> vals;

  
  idx_t denseSize = std::pow(N,NumDims);
  size_t nnz = denseSize * density;
  
  // create a random list of dense offsets without duplicates
  std::unordered_set<idx_t> denseOffsets;
  while(denseOffsets.size() < nnz) {
    idx_t i = std::rand() % denseSize;
    denseOffsets.insert(i);
  }
  if (fullDiagonal) {
    for(int i = 0; i < N; i++) {
      idx_t ithDiag = 0;
      for(int j = 0; j < NumDims; j++) {
        ithDiag += i * std::pow(N,j);
      }
      denseOffsets.insert(ithDiag);
    }
  }

  //convert those offsets to index values
  for(idx_t denseOffset : denseOffsets) {
    for(int d = 0; d < NumDims; d++) {
      idx_t thisDim = denseOffset % N;
      dims[d].push_back(thisDim);
      denseOffset /= N;
    }
  }
  
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0,10.0);
  for(size_t i = 0; i < dims[0].size(); i++) {
    vals.push_back((ElmType) dis(gen));
  }
  return make_random_sparse_view_helper<ElmType, NumDims, Format>(dims, vals, idx_seq_from_to<0,NumDims>());
} 


} //namespace RAJA
#endif
