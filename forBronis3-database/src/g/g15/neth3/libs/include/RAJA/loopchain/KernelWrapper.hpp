//Contains class definitions and constructor functions for the kernel wrapper object

#ifndef RAJA_KernelWrapper_HPP
#define RAJA_KernelWrapper_HPP

#include "RAJA/config.hpp"
#include "RAJA/loopchain/SymExec.hpp"
#include "RAJA/loopchain/SymbolicSegment/SymbolicSegment.hpp"
#include "RAJA/pattern/kernel.hpp"

#include "RAJA/pattern/kernel/TiledLambda.hpp"

#include "RAJA/loopchain/PolicyToNestingOrder.hpp"
#include <vector>
#include <string>

#include "RAJA/util/all-isl.hpp"
#include <barvinok/barvinok.h>
namespace RAJA
{

template <typename T>
T get_array_name(SymAccess a);

template <typename T>
void mergeVectors(std::vector<T> &v1, std::vector<T> &v2) {
  for(long unsigned int i = 0; i < v2.size(); i++) {
    auto t = v2.at(i);
    auto search = std::find(v1.begin(), v1.end(), t);
    if(search == v1.end()) {      
      v1.push_back(t);
    }
  }
}

//KernelWrapper wraps a kernel execution so it can be transformed before execution
//The type and constructor parameters are the same as the kernel function.
template <typename KernelPol, typename SegmentTuple, typename... Bodies>
struct KernelWrapper {
  using KPol = KernelPol;
  using BodyTuple = camp::tuple<Bodies...>;
  
  //these fields come from the kernel function
  const SegmentTuple segments;
  const BodyTuple bodies;
 
  //these fields are extra. they exist to enable runtime transformation
  // instead of compile time, like tile sizes.
  std::vector<camp::idx_t> overlapAmounts;
  std::vector<camp::idx_t> tileSizes;
  std::vector<SymAccess> accesses;
  static constexpr int numArgs = camp::tuple_size<SegmentTuple>::value;
  static constexpr int NumLambdas = sizeof...(Bodies);
  KernelWrapper(SegmentTuple  _segments, const Bodies&... _bodies) : 
    segments(_segments), bodies(_bodies...) {
     overlapAmounts = std::vector<camp::idx_t>();
     tileSizes = std::vector<camp::idx_t>();
     //accesses = _execute_symbolically();
  }

  KernelWrapper(SegmentTuple  _segments, tuple<Bodies...> _bodies) : 
    segments(_segments), bodies(_bodies) {
     overlapAmounts = std::vector<camp::idx_t>();
     tileSizes = std::vector<camp::idx_t>();
     //accesses = _execute_symbolically();
  }

  
  KernelWrapper(const KernelWrapper &) = default;
  KernelWrapper(KernelWrapper &&) = default;
  KernelWrapper & operator=(const KernelWrapper &) = default;
  KernelWrapper & operator=(KernelWrapper &&) = default;


  template <camp::idx_t Idx>
  RAJA_INLINE
  SymIterator make_sym_iterator() {
    std::string iteratorName = "i" + std::to_string(Idx);
    return SymIterator(iteratorName);
  }
  template <camp::idx_t Idx>
  RAJA_INLINE
  SymIterator make_sym_iterator(idx_t symVal) {
    std::string iteratorName = "i" + std::to_string(Idx);
    return SymIterator(iteratorName, symVal);
  }
  template <typename T, typename...Ts>
  std::string segment_names(T seg, Ts... segs) {
    if constexpr (sizeof...(Ts) == 0) {
      return seg.name;
    } else {
      return seg.name + std::string(",") + segment_names(segs...);
    } 
  }
  
  template <typename T, typename...Ts>
  std::string segment_bounds(T seg, Ts... segs) {
    if constexpr(sizeof...(Ts) == 0) {
      std::stringstream s;
      s << seg;
      return s.str();
    } else {
      std::stringstream s;
      s << seg << " and " << segment_bounds(segs...);
      return s.str();
    }
  }


  template <typename TupleType, idx_t...Is>
  size_t symbolic_size(TupleType segs, camp::idx_seq<Is...>) {
    //std::cout << "symbolic size start\n";
    std::stringstream s;
    s << "{";
    s <<  "[" << segment_names(camp::get<Is>(segs)...) << "]";
    s << " : ";
    s << segment_bounds(camp::get<Is>(segs)...);
    s << "}";
    //std::cout << "symbolic bounds expression: " << s.str() << "\n"; 
    isl_ctx * ctx = isl_ctx_alloc();
    isl_set * ispace = isl_set_read_from_str(ctx, s.str().c_str());
    isl_pw_qpolynomial * cardinality = isl_set_card(ispace);

    auto valueExtractor = [](isl_set*, isl_qpolynomial * qp, void * user) {
      isl_val * value = isl_qpolynomial_get_constant_val(qp);
      long size = isl_val_get_num_si(value);
      *((long*)user) = size;
      return isl_stat_ok;
    };
    
    long size;
    isl_pw_qpolynomial_foreach_piece(cardinality, valueExtractor, (void*) (&size));
    isl_pw_qpolynomial_free(cardinality); 
    //isl_ctx_free(ctx);
    //std::cout << "symbolic size end\n";
    return (size_t) size;
  }

  template <typename TupleType>
  size_t calculate_iteration_space_size(TupleType segs) {
    return calculate_iteration_space_size(segs, idx_seq_for(segs));
  }
  template <typename TupleType, idx_t...Is>
  size_t calculate_iteration_space_size(TupleType segs, camp::idx_seq<Is...> seq) {
    if constexpr (sizeof...(Is) == 0) {
      camp::sink(seq); // quiets use-def warning
      return 0;
    } else {
      using SegType = decltype(camp::get<0>(segs));
      if constexpr (std::is_same<SymbolicSegment&,SegType>::value) {
        return symbolic_size(segs, seq);
     } else {
        camp::sink(seq); // quiets use-def warning
        return vmult(camp::get<Is>(segs).size() ...);
      }
    }
  }

  template <camp::idx_t... Is>
  RAJA_INLINE
  auto make_iterator_tuple(camp::idx_seq<Is...>) { 
    auto iterators = camp::make_tuple((make_sym_iterator<Is>())...);
    return iterators;
  }
  
  template <typename...T, camp::idx_t... Is>
  RAJA_INLINE
  auto make_iterator_tuple(camp::tuple<T...> symVals, camp::idx_seq<Is...>) { 
    auto iterators = camp::make_tuple(make_sym_iterator<Is>(camp::get<Is>(symVals))...);
    return iterators;
  }
  

  std::vector<SymAccess> collect_accesses() {
    return std::vector<SymAccess>();
  }

  template <typename... Iterators>
  RAJA_INLINE
  std::vector<SymAccess> collect_accesses(SymIterator iterator, Iterators&&... rest) {
    std::vector<SymAccess> accesses = collect_accesses(std::forward<Iterators>(rest)...);

    for(long unsigned int i = 0; i < iterator.accesses->size(); i++) {
      auto access = iterator.accesses->at(i);
      auto search = std::find(accesses.begin(), accesses.end(), access);
      if(search == accesses.end()) {
        accesses.push_back(access);
      }
    }
    
    return accesses;
  }

  template <typename T1, camp::idx_t... Is>
  RAJA_INLINE
  auto collect_accesses_from_iterators(T1 iterators, camp::idx_seq<Is...>) {
    return collect_accesses(camp::get<Is>(iterators)...);
  }

  template <typename T1, typename T2, camp::idx_t...Is>
  std::vector<SymAccess> es_helper(T1 function, T2 iterators, camp::idx_seq<Is...> seq) {
   
    function(camp::get<Is>(iterators)...);
    auto accesses = collect_accesses_from_iterators(iterators, seq);
   
    auto segsForThisLambda = make_tuple(camp::get<Is>(segments)...);

    size_t iterationSpaceSize = calculate_iteration_space_size(segsForThisLambda);
    
    for(auto & access : accesses ) {
      access.numIterations = iterationSpaceSize;
    }
    
    camp::sink(camp::get<Is>(iterators).clear()...);

    return accesses;
  }  

  template <idx_t SearchIdx, typename CurrTupleType, typename FullTupleType, idx_t I, typename ExecPol, typename...NestedPol>
  std::vector<SymAccess> es_traverse(const CurrTupleType, FullTupleType fullTuple, statement::For<I,ExecPol,NestedPol...>) {

    //expand the current tuple
    constexpr long int currSize = camp::tuple_size<CurrTupleType>::value;//tuple_len(currTuple);
    constexpr idx_t newSize = currSize+1;
    auto newTuple = tuple_slice<0,newSize>(fullTuple);
 
    //recurse
    auto policyTuple = make_tuple(NestedPol{}...);
    return es_traverse<SearchIdx>(newTuple, fullTuple, policyTuple);
  }

  template <idx_t SearchIdx, typename CurrTupleType, typename FullTupleType, typename Pol1, typename...NestedPol>
  std::vector<SymAccess> es_traverse(CurrTupleType currTuple, FullTupleType fullTuple, camp::tuple<Pol1, NestedPol...> ) {
    
    //execute the first one
    std::vector<SymAccess> accesses = es_traverse<SearchIdx>(currTuple, fullTuple, Pol1{});
    //recurse
    if constexpr (sizeof...(NestedPol)) {
      std::vector<SymAccess> moreAccesses = es_traverse<SearchIdx>(currTuple, fullTuple, make_tuple(NestedPol{}...));
      mergeVectors(accesses, moreAccesses);
      
    }
    return accesses;
  }

  template <idx_t SearchIdx, typename CurrTupleType, typename FullTupleType, idx_t LamNum, typename...LamDetails>
  std::vector<SymAccess> es_traverse(CurrTupleType currTuple, FullTupleType, statement::Lambda<LamNum,LamDetails...> ) {
    if constexpr (SearchIdx == -1 || LamNum == SearchIdx) {
      return es_helper(camp::get<LamNum>(bodies), currTuple, idx_seq_for(currTuple));
    } else {
      return std::vector<SymAccess>{};
    }
  }

  template <idx_t SearchIdx, typename CurrTupleType, typename FullTupleType, typename Statement, typename...Statements>
  std::vector<SymAccess> es_traverse(CurrTupleType currTuple, FullTupleType fullTuple, camp::list<Statement, Statements...>) {
    return es_traverse<SearchIdx>(currTuple, fullTuple, make_tuple(Statement{}, Statements{}...));
  }

  
  template <idx_t LambdaIdx = -1>
  std::vector<SymAccess> execute_symbolically() {
    
    auto iterators = make_iterator_tuple(camp::make_idx_seq_t<numArgs>());
    return es_traverse<LambdaIdx>(make_tuple(), iterators, KPol{});
  }

  template <idx_t LambdaIdx = -1, typename... T>
  std::vector<SymAccess> execute_symbolically(T... symVals) {
    
    auto iterators = make_iterator_tuple(make_tuple(symVals...), camp::make_idx_seq_t<numArgs>());
    return es_traverse<LambdaIdx>(make_tuple(), iterators, KPol{});
  }
  template <idx_t LambdaIdx = -1, typename... T>
  std::vector<SymAccess> execute_symbolically(camp::tuple<T...> symVals) {
    
    auto iterators = make_iterator_tuple(symVals, camp::make_idx_seq_t<numArgs>());
    return es_traverse<LambdaIdx>(make_tuple(), iterators, KPol{});
  }


  /*std::vector<SymAccess> execute_symbolically() {
    auto iterators = make_iterator_tuple(camp::make_idx_seq_t<numArgs>());
    return es_traverse<-1>(make_tuple(), iterators, KPol{});
  }*/

  // Traditional execution. For normal kernels, this resolves to a call to kernel.
  // If the tile sizes or overlap tile amounts are specified at runtime,
  //  those values are added to the loop data before executing.
  template <camp::idx_t... Is>
  RAJA_INLINE
  void execute(camp::idx_seq<Is...>) const { 
  
    if(overlapAmounts.size() != 0 && tileSizes.size() != 0) {
      util::PluginContext context{util::make_context<KernelPol>()};
     
      using segment_tuple_t = typename IterableWrapperTuple<camp::decay<SegmentTuple>>::type;

      auto params = RAJA::make_tuple();
      using param_tuple_t = camp::decay<decltype(params)>;

      auto res = resources::get_default_resource<KPol>();
      using Resource = decltype(res);
      using loop_data_t = internal::LoopData<segment_tuple_t, param_tuple_t, Resource, camp::decay<Bodies>...>;
   
      loop_data_t loop_data(overlapAmounts, tileSizes, 
                            make_wrapped_tuple(segments), params, res, camp::get<Is>(bodies)...);

      util::callPostCapturePlugins(context);

      using loop_types_t = internal::makeInitialLoopTypes<loop_data_t>;

      util::callPreLaunchPlugins(context);

      RAJA_FORCEINLINE_RECURSIVE
      internal::execute_statement_list<KernelPol, loop_types_t>(loop_data);

      util::callPostLaunchPlugins(context);
    } else {
      RAJA::kernel<KernelPol>(segments, camp::get<Is>(bodies)...);
    }

    reset_symbolic_segments(segments);

  } //execute
  
  
  
  RAJA_INLINE
  void operator() () const {
    if constexpr (numArgs == 1) {
      using ForType = camp::first<KPol>;
      using StatementTypes = typename ForType::enclosed_statements_t;
      using StatementType = camp::first<StatementTypes>;
      
      if constexpr (std::is_same<StatementType,statement::TiledLambda<0>>::value) {
        auto seq = camp::make_idx_seq_t<sizeof...(Bodies)>{};
        execute(seq);
      } else {

        using ExecPol = typename ForType::execution_policy_t;
        RAJA::forall<ExecPol>(camp::get<0>(segments), camp::get<0>(bodies));
      }
    } else {
      auto seq = camp::make_idx_seq_t<sizeof...(Bodies)>{};
      execute(seq);
    }
  }

  RAJA_INLINE
  void operator() (SegmentTuple s) {
    //TODO: Enable the kernel to be executed with a different segment
    kernel<KPol>(segments, bodies);
  }


  template <camp::idx_t I, camp::idx_t...Is>
  std::string segment_string_helper(camp::idx_seq<I, Is...>) {
    auto currSeg = camp::get<I>(segments);
    std::stringstream s;
    s << "(" << *currSeg.begin() << "," << *currSeg.end() << ") ";

    if constexpr (sizeof...(Is) == 0) {
      return s.str();
    } else {
      return s.str() + segment_string_helper(camp::idx_seq<Is...>{});
    }
  }

  std::string segment_string() {

    return segment_string_helper(idx_seq_for(segments));
  }


  std::string model_data() {

    const char * separator = "";
    std::stringstream o;
    //JSON opening bracket
    o << "{\n";
    


    // NestingORder key and list start
    o << " \"NestingOrder\" : [";

    
    std::vector<camp::idx_t> nestingOrder = policy_to_nesting_order<KPol>();
    //NestingOrder list contents
    for(auto i : nestingOrder) {
      o << separator << i;
      separator = ",";
    }
    //NestingOrder list end
    o << "]";
    //Next entry
    o << ",\n";

    
    //LambdaParameters key and list start
    o << "\"LambdaParameters\" : [";
    
    //LambdaParameters contents
    separator = "";
    for(int i = 0; i < numArgs; i++) {
      o << separator << "\"i" << i <<"\"";
      separator = ",";
    }

    //LambdaParameters list end
    o << "]";

    //Next entry
    o << ",\n";

    //AccessArguments key and list start
    o << "\"AccessArguments\" : [";

    auto accesses = execute_symbolically();
    separator = "";
    for(auto access: accesses) {
      o << separator;
      o << "[";
      o << "\"" << get_array_name(access) << "\"" << ",";
      char * _s = "";
      for(auto iterator : access.iterators) {
        o << _s << "\"" << iterator << "\"";
        _s = ",";
      }
      o << "]";
      separator = ",";
    }
    //list end
    o << "]";
    

    o << ",\n";

    //DataLayouts key
    o << "\"DataLayouts\" : ";

    //DataLayouts map start
    o << "{\n";
    
    std::map<std::string, std::vector<size_t>> m{};

    for(auto access: accesses) {
      m[get_array_name(access)] = access.layout_permutation;
    }     

    separator = "";
    for(const auto& [key,value] : m) {
      o << separator << "\"" << key << "\"" << ": [";
      const char * s2 = "";
      for(auto v : value) {
        o << s2 << v;
        s2 = ",";
      }
      o << "]";
      separator = ",\n";
    }

    //Map end
    o << "}";

    
    // JSON closing bracket
    o << "}";
    return o.str();
  }

    std::vector<camp::idx_t> normalize_access(SymAccess a) {
    auto layout_perm = a.layout_permutation;
    auto policy_order = policy_to_nesting_order<KPol>();

    auto access_args = a.iterators;

    std::vector<camp::idx_t> access_indices;
   
    for (auto arg : access_args) {
      auto digits = arg.name.substr(1);
      auto index = std::stoi(digits, nullptr);
      access_indices.push_back(index);
    }

    std::vector<camp::idx_t> normalized_logical_access;

    for(auto index : access_indices) {
      auto indexof = std::find(policy_order.begin(), policy_order.end(), index) - policy_order.begin();
      normalized_logical_access.push_back(indexof);
    }


    std::vector<camp::idx_t> normalized_access;
    for(auto index : layout_perm) {
      normalized_access.push_back(normalized_logical_access.at(index));
    }
    return normalized_access; 
  }

  template <idx_t I>
  auto ni_helper() {
    if constexpr (I == numArgs) { return 1;} else
    {
      auto s = get<I>(segments);
      return s.size() * ni_helper<I+1>();
    }
  }
 
  auto num_iterations() {
    return ni_helper<0>();
  }


}; // KernelWrapper

template <typename KernelPol, typename SegmentTuple, typename... Bodies>
KernelWrapper<KernelPol, SegmentTuple, Bodies...> 
make_kernel_tuple(const SegmentTuple & segment,  tuple<Bodies...> bodies) {
  return KernelWrapper<KernelPol,SegmentTuple,Bodies...>(segment, bodies);
}


//creates a kernel object using the same interface as the kernel function
template <typename KernelPol, typename SegmentTuple, typename... Bodies>
KernelWrapper<KernelPol, SegmentTuple, Bodies...> 
make_kernel(const SegmentTuple & segment,   Bodies const &... bodies) {
  return KernelWrapper<KernelPol,SegmentTuple,Bodies...>(segment, bodies...);
}
//creates a kernel object using the same interface as the forall function
template <typename ExecPol, typename Segment, typename Body> 
auto make_forall(Segment segment, const Body & body) {
  using KernPol = 
    RAJA::KernelPolicy<
      statement::For<0,ExecPol,
        statement::Lambda<0>
      >
    >;     
  
  return KernelWrapper<KernPol, camp::tuple<Segment>, Body>(camp::make_tuple(segment), body);
}

template <typename KernPol, typename SegmentTuple, typename...Bodies>
auto change_segment_tuple(KernelWrapper<KernPol, SegmentTuple, Bodies...> knl, SegmentTuple newSeg) {
  
  return make_kernel<KernPol>(newSeg, camp::get<0>(knl.bodies));
}





} //namespace RAJA


#endif

