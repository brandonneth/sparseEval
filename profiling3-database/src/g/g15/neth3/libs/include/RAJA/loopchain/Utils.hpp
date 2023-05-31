//Contains camp utilities mostly
#ifndef RAJA_LoopChainUtils_HPP
#define RAJA_LoopChainUtils_HPP

#include "RAJA/config.hpp"
#include "RAJA/pattern/kernel.hpp"
#include "RAJA/pattern/kernel/Lambda.hpp"

namespace RAJA
{

// camp::idx_seq utility functions
//   - idx_seq_cat
//   - idx_seq_shift
//   - idx_seq_for
//   - idx_seq_from_to

// idx_seq_cat: Concatenates an arbitrary number of sequences

template <camp::idx_t...Is, camp::idx_t...Js>
auto idx_seq_cat(camp::idx_seq<Is...>, camp::idx_seq<Js...>) {
  return camp::idx_seq<Is...,Js...>{};
}

template <camp::idx_t...Is, typename...Seqs>
auto idx_seq_cat(camp::idx_seq<Is...> s1, Seqs&&...seqs) {
  return idx_seq_cat(s1, idx_seq_cat(std::forward<Seqs>(seqs)...));
}

// idx_seq_shift: Shifts each value in the sequence by the amount

template <camp::idx_t amount, camp::idx_t...Is>
auto idx_seq_shift(camp::idx_seq<Is...>) {
  return camp::idx_seq<(Is+amount)...>{};
}

// idx_seq_for: Returns the idx seq for a tuple

template <typename... ElemTypes>
auto idx_seq_for(camp::tuple<ElemTypes...>) {
  return camp::make_idx_seq_t<sizeof...(ElemTypes)>{};
}

// idx_seq_from_to: Returns an idx seq from the start value up to the end value

template <camp::idx_t start, camp::idx_t end>
auto idx_seq_from_to() {
  auto zeroSeq = camp::make_idx_seq_t<end-start>{};
  
  return idx_seq_shift<start>(zeroSeq);
}





// camp::tuple utility functions
//  - tuple_cat
//  - tuple_slice
//  - tuple_len 
//  - tuple_reverse
//  - tuple_zip

// tuple_cat: Concatenates an arbitrary number of tuples

template <typename Tuple1>
auto tuple_cat(Tuple1 t1) {
  return t1;
}
template <typename Tuple1, typename Tuple2>
auto tuple_cat(Tuple1 t1, Tuple2 t2) {
  return camp::tuple_cat_pair(t1,t2);
}

template <typename Tuple1, typename...Tuples>
auto tuple_cat(Tuple1 t1, Tuples&&...tuples) {
  return tuple_cat(t1, tuple_cat(std::forward<Tuples>(tuples)...));
}

// tuple_slice: Returns a slice of a tuple, with [) bounds as template arguments or with a idx sequence

template <typename Tuple, camp::idx_t...Is>
auto tuple_slice(Tuple t, camp::idx_seq<Is...>) {
  return camp::make_tuple(camp::get<Is>(t)...);
}

template <camp::idx_t start, camp::idx_t end, typename Tuple>
auto tuple_slice(Tuple t) {
  if constexpr (start >= end) {
    camp::sink(t); // quiets use def warning
    return camp::make_tuple();
  } else {
    return tuple_slice(t, idx_seq_from_to<start,end>());
  }
}

// tuple_len: Returns the length of a tuple
template <typename...Ts>
RAJA_INLINE
constexpr auto tuple_len(camp::tuple<Ts...>) {
  return sizeof...(Ts);
}

template <typename T, typename...Ts>
auto tuple_reverse(camp::tuple<T,Ts...> t) {
  if constexpr (sizeof...(Ts) == 0) {
    return t;
  } else {
    auto endTuple = camp::make_tuple(camp::get<0>(t));
    auto subTuple = tuple_reverse(tuple_slice<1,sizeof...(Ts) + 1>(t));

    return tuple_cat(subTuple, endTuple);
  }
}


// tuple_zip : given a tuple of tuples that are all the same length,
//  creates a new tuple of tuples where the first tuple is all the first
//  elements of the original tuples, the second element is all the second
//  elements of the original tuples, etc...


//returns the DimNumth slice of the zip. This means, the DimNumth element of each of the original tuples
template <camp::idx_t DimNum, typename...InnerTupleTypes, camp::idx_t...Is>
auto tuple_zip_slice(camp::tuple<InnerTupleTypes...> t, camp::idx_seq<Is...>) {
  //return make_tuple(camp::get<DimNum>(camp::get<Is>(t))...);
  return camp::tuple{camp::get<DimNum>(camp::get<Is>(t))...};
}

template <typename T>
void doPrint(T t) {
  std::cout << t;
}

template <typename T>
void doPrint(std::vector<T> v) {
  std::cout << "[";
  for(size_t i = 0; i < v.size(); i++) {
    std::cout << v[i] << ", ";
  }
  std::cout << "]";
}
template <typename T, typename... Args>
void doPrint(T t, Args... args) {
  doPrint(t);
  doPrint(args...);
}


template <typename...InnerTupleTypes, camp::idx_t...OuterDims, camp::idx_t... InnerDims>
auto tuple_zip_helper(camp::tuple<InnerTupleTypes...> t, 
                      camp::idx_seq<OuterDims...> oSeq,
                      camp::idx_seq<InnerDims...> iSeq) {
  //return make_tuple((tuple_zip_slice<InnerDims>(t, oSeq))...);
  //std::cout << "zip slices:\n";
  //doPrint(tuple_zip_slice<InnerDims>(t, oSeq) ...) ;
  //std::cout << "\n";
  auto t_extra = camp::tuple{tuple_zip_slice<InnerDims>(t, oSeq)..., camp::make_tuple(-100)};

  auto sliced = tuple_slice(t_extra, iSeq);
  return sliced;
 // return camp::tuple{tuple_zip_slice<0>(t, oSeq)};

}
template <typename InnerTupleType, typename...InnerTupleTypes, camp::idx_t...Is>
auto tuple_zip(camp::tuple<InnerTupleType, InnerTupleTypes...> t, camp::idx_seq<Is...> seq) {
  constexpr auto innerTupleLength = camp::tuple_size<InnerTupleType>::value;
  
  return tuple_zip_helper(t, seq, idx_seq_from_to<0,innerTupleLength>());
}

template <typename...InnerTupleTypes>
auto tuple_zip(camp::tuple<InnerTupleTypes...> t) {
  return tuple_zip(t, idx_seq_for(t));
}

template <idx_t N, typename T>
auto tuple_repeat(T element) {
  if constexpr (N <= 0) {
    camp::sink(element); // quiets use-def warning
    return camp::tuple{};
  } else {
    return tuple_cat(camp::tuple{element}, tuple_repeat<N-1>(element));
  }
}

// Vararg utilities, implemented for both tuples and parameter packs
//   - max
//   - min

template <typename T>
auto vmax(T val) {
  return val;
}

template <typename T, typename...Ts>
auto vmax(T val, Ts...rest) {
  return std::max(val, vmax(rest...));
}

template <typename T>
auto vmin(T val) {
  return val;
}

template <typename T, typename...Ts>
auto vmin(T val, Ts...rest) {
  return std::min(val, vmin(rest...));
}

template <typename T>
auto vmult(T val) {
  return val;
}

template <typename T, typename...Ts>
auto vmult(T val, Ts...rest) {
  return val * vmult(rest...);
}


}

#endif
