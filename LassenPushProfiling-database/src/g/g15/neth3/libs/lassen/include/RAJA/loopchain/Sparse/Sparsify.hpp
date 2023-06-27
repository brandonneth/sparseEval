#ifndef _RAJA_SPARSIFY_HPP
#define _RAJA_SPARSIFY_HPP

#include "RAJA/loopchain/Utils.hpp"

namespace RAJA {



std::vector<camp::idx_t> invert_permutation_unsafe(std::vector<camp::idx_t> p) {
  std::vector<camp::idx_t> q = std::vector<camp::idx_t>(p.size());
  for(size_t i = 0; i < p.size(); i++) {
    q[p[i]] = i;
  }
  return q;
}

std::vector<camp::idx_t> apply_permutation(std::vector<camp::idx_t> p, std::vector<camp::idx_t> v) {
  std::vector<camp::idx_t> pp = std::vector<camp::idx_t>(v.size());

  for(size_t i = 0; i < p.size(); i++) {
    pp[p[i]] = v[i];
  }

  return pp;
}
/*
template <typename ElmType, std::size_t ViewDims, std::size_t CurrIdx, std::size_t MaxIdx, typename ViewFormat>
auto sparsification_follow_dimensions( SparseView<ElmType,ViewDims,ViewFormat> & A, std::vector<camp::idx_t> argumentOrderInverse, LeadSegmentImpl * lead) {
  if constexpr (CurrIdx == MaxIdx) {
    camp::sink(A,argumentOrderInverse,lead);
    return make_tuple();
  } else {
    
    auto currDimension = make_follow_segment(A.impl.dim(argumentOrderInverse[CurrIdx]), lead);
    auto remainingDims = sparsification_follow_dimensions<ElmType, ViewDims, CurrIdx+1, MaxIdx>(A, argumentOrderInverse, lead);
    return tuple_cat(camp::tuple(currDimension), remainingDims);
  }
}
*/
template <typename ElmType,std::size_t ViewDims,typename ViewFormat, typename... DenseTupleTypes>
auto sparsify_iteration_space_equal_dims(camp::tuple<DenseTupleTypes...> denseSegs, SparseView<ElmType,ViewDims,ViewFormat> & A, std::vector<camp::idx_t> argumentOrder, std::vector<camp::idx_t> nestingOrder) {
  camp::sink(denseSegs);
  if constexpr (ViewDims != sizeof...(DenseTupleTypes)) {
    std::cerr << "Warning in sparsify_iteration_space_equal_dims: Iteration space and View dimensionality mismatch.\n";
  }

  return denseSegs;
}

template <camp::idx_t CurrIdx, typename...TupleTypes>
auto trim_segment_tuple_impl(camp::tuple<TupleTypes...> segments, std::vector<idx_t> argumentOrder) {
  using EntryTypeRef = decltype(camp::get<0>(segments));
  using EntryType = typename std::remove_reference<EntryTypeRef>::type;
  if constexpr (CurrIdx >= sizeof...(TupleTypes)) { 
    return std::vector<EntryType>();
  } else {
    std::vector<EntryType> thisDim;
    if (std::find(argumentOrder.begin(), argumentOrder.end(), CurrIdx) != argumentOrder.end()) {
      thisDim.push_back(camp::get<CurrIdx>(segments));
    }
    auto remaining = trim_segment_tuple_impl<CurrIdx+1>(segments, argumentOrder);
    thisDim.insert(thisDim.end(), remaining.begin(), remaining.end());
    return thisDim;
  }
}

template <typename T, camp::idx_t... Is>
auto tuplify(std::vector<T> v, camp::idx_seq<Is...>) {
  
  return make_tuple(v[Is]...);
}

template <typename TupleType, typename...TupleTypes, camp::idx_t...Is>
auto vectorify(camp::tuple<TupleType,TupleTypes...> t, camp::idx_seq<Is...>) {
  using EntryType = typename std::remove_reference<TupleType>::type;
  std::vector<EntryType> v{camp::get<Is>(t)...};
  return v;
}


template <camp::idx_t DataDims, typename...TupleTypes>
auto trim_segment_tuple(camp::tuple<TupleTypes...> segments, std::vector<idx_t> argumentOrder) {
  auto trimmedSegmentVector = trim_segment_tuple_impl<0>(segments, argumentOrder);
  using TupleType = decltype(camp::get<0>(segments));
  using EntryType = typename std::remove_reference<TupleType>::type;
  auto trimmedSegmentTuple = tuplify<EntryType>(trimmedSegmentVector,idx_seq_from_to<0,DataDims>());

  return trimmedSegmentTuple;
}

auto shrink(std::vector<idx_t> v) {
  std::vector<idx_t> shrunk = std::vector<idx_t>(v.size());
  for(size_t i = 0; i < v.size(); i++) {
    idx_t biggerThan = 0;
    for(size_t j = 0; j < v.size(); j++) {
      if(v[i] > v[j]) {biggerThan++;}
    
    }
    shrunk[i] = biggerThan;
  }
  return shrunk;
}

template <typename DenseTupleType, camp::idx_t...Is>
auto range_segments_to_wrapper_vector(DenseTupleType t, camp::idx_seq<Is...>) {
  std::vector<SparseSegment> returnVector{make_dense_segment(*camp::get<Is>(t).begin(), *camp::get<Is>(t).end())...};
  return returnVector;
}

template <typename...DenseTupleTypes, typename SparseTuple>
auto integrate_sparse_segments(camp::tuple<DenseTupleTypes...> inSegs, std::vector<idx_t> trimmedInSegsIdxs, SparseTuple sparsePart) {
  auto outSegsVector = range_segments_to_wrapper_vector(inSegs, idx_seq_for(inSegs));
  auto sparseSegsVector = vectorify(sparsePart, idx_seq_for(sparsePart));

  for(size_t i = 0; i < sparseSegsVector.size(); i++) {
    outSegsVector[trimmedInSegsIdxs[i]] = sparseSegsVector[i];
  }

  return tuplify(outSegsVector, idx_seq_for(inSegs));

}

template <typename ElmType,size_t ViewDims, typename ViewFormat, typename... DenseTupleTypes>
auto sparsify_iteration_space_smaller_data(camp::tuple<DenseTupleTypes...> denseSegs, SparseView<ElmType,ViewDims,ViewFormat> & A, std::vector<camp::idx_t> argumentOrder, std::vector<camp::idx_t> nestingOrder) {
  camp::sink(A, argumentOrder);
  if constexpr (ViewDims >= sizeof...(DenseTupleTypes)) {
    std::cerr << "Warning in sparsify_iteration_space_smaller_data:  View dimensionality greater than or equal to iteration space dimensionality.\n";
  }
  //trimmedNestingOrder = [idx for idx in nesting if idx in access]
  std::vector<idx_t> trimmedNestingOrder;
  for(auto i : nestingOrder) {
    if (std::find(argumentOrder.begin(), argumentOrder.end(), i) != argumentOrder.end()) {
      trimmedNestingOrder.push_back(i);
    }
  }
  //trimmedInSegs = [seg for seg in inSegs if seg.idx in access]
  auto trimmedInSegs = trim_segment_tuple<ViewDims,DenseTupleTypes...>(denseSegs, argumentOrder);

  //trimmedInSegsIdxs = [seg.idx for seg in trimmedInSegs]
  auto trimmedInSegsIdxs = std::vector<idx_t>(trimmedNestingOrder.begin(), trimmedNestingOrder.end());
  std::sort(trimmedInSegsIdxs.begin(), trimmedInSegsIdxs.end());

  //accessOrderOrder = shrink(accessOrder)
  auto argumentOrderOrder = shrink(argumentOrder);

  //accessOrderOrderMap = {access[i] : accessOrderOrder[i] for i in len(access)}
  std::map<idx_t, idx_t> argumentOrderOrderMap;
  for(size_t i = 0; i < argumentOrder.size(); i++) {
    argumentOrderOrderMap[argumentOrder[i]] = argumentOrderOrder[i];
  }

  //trimmedNestingOrderOrder = [accessOrderOrderMap[idx] for idx in trimmedNestingOrder]
  std::vector<idx_t> trimmedNestingOrderOrder;
  for(auto i : trimmedNestingOrder) {
    trimmedNestingOrderOrder.push_back(argumentOrderOrderMap[i]); 
  }

  //sparsePart = sparsify(nesting=trimmedNestingOrderOrder, access=accessOrderOrder, segments=trimmedInSegs)
  auto sparsePart = sparsify_iteration_space_equal_dims(trimmedInSegs, A, argumentOrderOrder, trimmedNestingOrderOrder);
  
  //outSegs = inSegs
  //for idx in 0..len(sparsePart):
  //  outSegs[trimmedInSegsIdxs[idx]] = sparsePart[idx]
  auto outSegs = integrate_sparse_segments(denseSegs, trimmedInSegsIdxs, sparsePart);

  return outSegs;
}

template <typename KernelPolicy, typename ElmType, size_t ViewDims, typename ViewFormat, typename... DenseTupleTypes>
auto sparsify_iteration_space_larger_data(camp::tuple<DenseTupleTypes...> denseSegs, SparseView<ElmType,ViewDims,ViewFormat> & A, std::vector<camp::idx_t> argumentOrder) {
  camp::sink(A, argumentOrder);
  if constexpr (ViewDims <= sizeof...(DenseTupleTypes)) {
    std::cerr << "Warning in sparsify_iteration_space_larger_data:  View dimensionality less than or equal to iteration space dimensionality.\n";
  }

  auto nestingOrder = policy_to_nesting_order<KernelPolicy>();
  auto nestingOrderInverse = invert_permutation_unsafe(nestingOrder);

  camp::sink(nestingOrderInverse);
  return denseSegs;
}


template <typename KernelPolicy, typename ElmType,std::size_t ViewDims,typename ViewFormat,typename... DenseTupleTypes>
auto sparsify_iteration_space(camp::tuple<DenseTupleTypes...> denseSegs, SparseView<ElmType,ViewDims,ViewFormat> & A, std::vector<camp::idx_t> argumentOrder) {
  auto nestingOrder = policy_to_nesting_order<KernelPolicy>();
  if constexpr (ViewDims == sizeof...(DenseTupleTypes)) {
    return sparsify_iteration_space_equal_dims(denseSegs, A, argumentOrder, nestingOrder);
  } else if constexpr (ViewDims < sizeof...(DenseTupleTypes)) {
    return sparsify_iteration_space_smaller_data(denseSegs, A, argumentOrder, nestingOrder);
  } else {
    return sparsify_iteration_space_larger_data<KernelPolicy>(denseSegs, A, argumentOrder); 
  }
}



template <typename ElmType, std::size_t ViewDims, typename ViewFormat, std::size_t CurrIdx, std::size_t MaxIdx>
auto sparsification_follow_dimensions_compressed( SparseView<ElmType,ViewDims,ViewFormat> & A, std::vector<camp::idx_t> argumentOrderInverse) {
  if constexpr (CurrIdx == MaxIdx) {
    camp::sink(A,argumentOrderInverse);
    return make_tuple();
  } else {
    
    auto currDimension = make_compressed_follow_segment(A.impl.dim(argumentOrderInverse[CurrIdx]));
    auto remainingDims = sparsification_follow_dimensions_compressed<ElmType, ViewDims, ViewFormat, CurrIdx+1, MaxIdx>(A, argumentOrderInverse);
    return tuple_cat(camp::tuple(currDimension), remainingDims);
  }
}


template <typename KernelPolicy, typename ElmType,size_t ViewDims, typename ViewFormat, typename... DenseTupleTypes>
auto sparsify_iteration_space_compressed(camp::tuple<DenseTupleTypes...> denseSegs, SparseView<ElmType,ViewDims,ViewFormat> & A, std::vector<camp::idx_t> argumentOrder) {
  auto nestingOrder = policy_to_nesting_order<KernelPolicy>();

  camp::sink(denseSegs);
  if constexpr (ViewDims != sizeof...(DenseTupleTypes)) {
    std::cerr << "Warning in sparsify_iteration_space_equal_dims: Iteration space and View dimensionality mismatch.\n";
  }

  
  auto nestingOrderInverse = invert_permutation_unsafe(nestingOrder);

  auto q = invert_permutation_unsafe(argumentOrder);
 
  
  auto followDimensions = sparsification_follow_dimensions_compressed<double,ViewDims,ViewFormat,1,ViewDims>(A, q);
  
  auto leadDimension = make_compressed_lead_segment(A.impl.dim(q[0]), followDimensions);

  auto sparseDimensions = tuple_cat(camp::tuple(leadDimension), followDimensions);

  return sparseDimensions;
}

}


#endif
