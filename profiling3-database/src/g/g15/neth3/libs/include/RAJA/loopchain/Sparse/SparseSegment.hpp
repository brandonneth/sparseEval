#ifndef _RAJA_SparseSegment_HPP
#define _RAJA_SparseSegment_HPP

#include "RAJA/loopchain/SymbolicSegment/SymbolicSegment.hpp"

namespace RAJA {

//interator and segment interfaces and wrappers
struct SparseIteratorImpl {
  virtual idx_t access(size_t i) = 0;
  virtual idx_t index() = 0;
};

struct SparseIterator {
  SparseIteratorImpl * impl;
  SparseIterator (SparseIteratorImpl * _impl) : impl(_impl) {}
  idx_t operator[](size_t i ) {
    return impl->access(i);
  }
  idx_t operator - (SparseIterator other) {
    return impl->index() - other.impl->index();
  }
};

struct SparseSegmentImpl {
  virtual SparseIterator begin() = 0;
  virtual SparseIterator end() = 0;
};
struct SparseSegment {
  using iterator = SparseIterator;
  using IndexType = idx_t;
  SparseSegmentImpl * impl;
  SparseSegment(SparseSegmentImpl * i) : impl(i) {}
  SparseIterator begin() const {return impl->begin();}
  SparseIterator end() const {return impl->end();}
};

//implementations for dense segments/iterators
struct DenseIteratorImpl : SparseIteratorImpl {
  idx_t value;
  DenseIteratorImpl(idx_t v) : value(v) {}
  idx_t access(size_t i) { return value + i;}
  idx_t index() {return value;}
};

struct DenseSegmentImpl : SparseSegmentImpl {
  idx_t low, high;
  DenseIteratorImpl * iterLow;
  DenseIteratorImpl * iterHigh;
  DenseSegmentImpl(idx_t l, idx_t h) : low(l), high(h) {
    iterLow = new DenseIteratorImpl(low);
    iterHigh = new DenseIteratorImpl(high);
  }
  SparseIterator begin() {
    return SparseIterator(iterLow);
  }
  SparseIterator end() {
    return SparseIterator(iterHigh);
  }
};

SparseSegment make_dense_segment(idx_t low, idx_t high) {
  auto impl = new DenseSegmentImpl(low, high);
  return SparseSegment(impl);
}

//implementations for lead segments/iterators
struct LeadIteratorImpl : SparseIteratorImpl {
  std::vector<idx_t> dimension;
  idx_t offset;
  size_t * currVal;
  LeadIteratorImpl(std::vector<idx_t> dim, idx_t _offset, size_t * _currVal) :
    dimension(dim), offset(_offset), currVal(_currVal) {}
  idx_t access(size_t i) {
    *currVal = i;
    return dimension[offset+i];
  }
  idx_t index() {return offset;}
};

struct LeadSegmentImpl : SparseSegmentImpl {
  std::vector<idx_t> dimension;
  LeadIteratorImpl * iterLow;
  LeadIteratorImpl * iterHigh;
  size_t currVal;
  
  LeadSegmentImpl(std::vector<idx_t> dim) : dimension(dim), currVal(0) {
    iterLow = new LeadIteratorImpl(dimension, 0, &currVal);
    iterHigh = new LeadIteratorImpl(dimension, dimension.size(), &currVal);
  }
  
  SparseIterator begin() {return SparseIterator(iterLow);}
  SparseIterator end() {return SparseIterator(iterHigh);}
};

SparseSegment make_lead_segment(std::vector<idx_t> dim) {
  auto impl = new LeadSegmentImpl(dim);
  return SparseSegment(impl);
}

//implementations for follow segments/iterators
struct FollowIteratorImpl : SparseIteratorImpl {
  std::vector<idx_t> dimension;
  idx_t offset;
  LeadSegmentImpl * leader;

  FollowIteratorImpl(std::vector<idx_t> dim, idx_t _offset, LeadSegmentImpl * lead) :
    dimension(dim), offset(_offset), leader(lead) {}

  idx_t access(size_t) {
    return dimension[offset + leader->currVal];
  }
  idx_t index() {return offset;}
};

struct FollowSegmentImpl : SparseSegmentImpl {
  std::vector<idx_t> dimension;
  LeadSegmentImpl * leader;
  FollowIteratorImpl * iterLow;
  FollowIteratorImpl * iterHigh;

  FollowSegmentImpl(std::vector<idx_t> _dimension, LeadSegmentImpl * _leader) 
    : dimension(_dimension), leader(_leader) {
    iterLow = new FollowIteratorImpl(dimension, 0, leader);
    iterHigh = new FollowIteratorImpl(dimension, 1, leader);
  }
  SparseIterator begin() { return SparseIterator(iterLow);}
  SparseIterator end() { return SparseIterator(iterHigh);}

};

SparseSegment make_follow_segment(std::vector<idx_t> dim, LeadSegmentImpl * leader) {
  auto impl = new FollowSegmentImpl(dim, leader);
  return SparseSegment(impl);
}

/*



template <camp::idx_t NumDims>
struct SparseIterator {
  virtual idx_t operator[](idx_t i) const;
};
template <camp::idx_t NumDims>
struct SparseIteratorWrapper {
  
  SparseIterator<NumDims> * iter;
 
  SparseIteratorWrapper(SparseIterator<NumDims> * i) : iter(i) {}

  idx_t operator[](idx_t i) const {
    return iter->operator[](i);
  }

  virtual idx_t operator - (const SparseIteratorWrapper<NumDims> & other) {
    return iter->operator[](0) - other.iter->operator[](0);
  }

};


template <camp::idx_t NumDims>
struct SparseSegment {
  using iterator = SparseIteratorWrapper<NumDims>;
  using IndexType = idx_t;
  virtual iterator begin();
  virtual iterator end();

};
template <camp::idx_t NumDims>
struct SparseSegmentWrapper {
  using iterator = SparseIteratorWrapper<NumDims>;

  using segment = SparseSegment<NumDims>*;
  
  segment seg;
  SparseSegmentWrapper(segment s) : seg(s) {}

  iterator begin() {
    return iterator(seg->begin());
  }

  iterator end() {
    return iterator(seg->end());
  }
};

template <camp::idx_t NumDims>
struct SparseIteratorDense : SparseIterator<NumDims> {
  idx_t val;
  SparseIteratorDense(idx_t v) : val(v) {}

  idx_t operator[](idx_t i) const{
    return val + i;
  }
  
};

template <camp::idx_t NumDims>
struct SparseSegmentDense : SparseSegment<NumDims> {
  using typename SparseSegment<NumDims>::iterator;

  idx_t low, high;
  SparseSegmentDense(camp::idx_t l, camp::idx_t h) : low(l), high(h) {}

  iterator begin() {
    return SparseIteratorWrapper(new SparseIteratorDense<NumDims>(low));
  }

  iterator end() {
    return SparseIteratorWrapper(new SparseIteratorDense<NumDims>(high));
  }
};








template <camp::idx_t NumDims>
struct SparseSymbolicSegmentLead;

template <camp::idx_t NumDims>
struct SparseIteratorWrapperLead {

  SparseSymbolicSegmentLead<NumDims> * seg;
  using iterator = std::vector<idx_t>::const_iterator; 
  iterator iter;

  SparseIteratorWrapperLead(SparseSymbolicSegmentLead<NumDims> * s, iterator i) : seg(s), iter(i) {}

  
  idx_t operator[] (idx_t i) {
    seg->currVal = i;
    return iter[i];
  }

  idx_t operator - (const SparseIteratorWrapperLead<NumDims> & other) {
    auto i = iter - other.iter;
    return i;
  } 
};


template <camp::idx_t NumDims>
struct SparseSymbolicSegmentLead {
using iterator = SparseIteratorWrapperLead<NumDims>;
using value_type = Index_type;
  using IndexType = Index_type;

const SparseView<double, NumDims> & view;
idx_t dimIdx;
std::vector<idx_t> dim;
idx_t currVal = 0;

SparseSymbolicSegmentLead(SparseView<double, NumDims> & v, idx_t d) :
  view(v), dimIdx(d) {
  dim = view.impl->dim(dimIdx);
}


iterator begin() {
  return SparseIteratorWrapperLead<NumDims>(this, dim.begin());
}

iterator end() {
  return SparseIteratorWrapperLead<NumDims>(this, dim.end());
}

};


template <camp::idx_t NumDims>
struct SparseIteratorWrapperFollow {

  SparseSymbolicSegmentLead<NumDims> * seg;
  using iterator = std::vector<idx_t>::const_iterator; 
  iterator iter;

  SparseIteratorWrapperFollow(SparseSymbolicSegmentLead<NumDims> * s, iterator i) : seg(s), iter(i) {}

  
  idx_t operator[] (idx_t i) {
    return iter[i + seg->currVal];
  }

  idx_t operator - (const SparseIteratorWrapperFollow<NumDims> & other) {
    auto i = iter - other.iter;
    return i;
  } 
};




template <camp::idx_t NumDims>
struct SparseSymbolicSegmentFollow {
using iterator = SparseIteratorWrapperFollow<NumDims>;
using value_type = Index_type;
using IndexType = Index_type;

const SparseView<double, NumDims> & view;
idx_t dimIdx;
std::vector<idx_t> dim;
SparseSymbolicSegmentLead<NumDims> * leadSeg;

SparseSymbolicSegmentFollow(SparseView<double, NumDims> & v, idx_t d, SparseSymbolicSegmentLead<NumDims> * l) :
  view(v), dimIdx(d), leadSeg(l) {
  dim = view.impl->dim(dimIdx);
}


iterator begin() const {
  return SparseIteratorWrapperFollow<NumDims>(leadSeg, dim.begin());
}

iterator end() const {
  return SparseIteratorWrapperFollow<NumDims>(leadSeg, dim.begin() + 1);
}

};
*/


} //namespace RAJA

template <>
struct std::iterator_traits<RAJA::SparseIterator> {
  typedef camp::idx_t value_type;
  typedef camp::idx_t difference_type;
  typedef camp::idx_t* pointer;
  typedef camp::idx_t& reference;
  typedef std::random_access_iterator_tag iterator_category;
};


template <>
struct std::iterator_traits<RAJA::SparseIteratorImpl> {
  typedef camp::idx_t value_type;
  typedef camp::idx_t difference_type;
  typedef camp::idx_t* pointer;
  typedef camp::idx_t& reference;
  typedef std::random_access_iterator_tag iterator_category;
};


#endif
