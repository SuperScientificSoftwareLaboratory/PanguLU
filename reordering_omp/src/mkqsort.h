#ifndef MKQSORT_H
#define MKQSORT_H

/* Swap two items pointed to by A and B using temporary buffer t. */
#define _GKQSORT_SWAP(a, b, t) ((void)((t = *a), (*a = *b), (*b = t)))

/* Discontinue quicksort algorithm when partition gets below this size.
   This particular magic number was chosen to work best on a Sun 4/260. */
#define _GKQSORT_MAX_THRESH 4

/* The next 4 #defines implement a very fast in-line stack abstraction. */
#define _GKQSORT_STACK_SIZE	    (8 * sizeof(size_t))
#define _GKQSORT_PUSH(top, low, high) (((top->_lo = (low)), (top->_hi = (high)), ++top))
#define	_GKQSORT_POP(low, high, top)  ((--top, (low = top->_lo), (high = top->_hi)))
#define	_GKQSORT_STACK_NOT_EMPTY	    (_stack < _top)


/* The main code starts here... */
#define GK_MKQSORT(GKQSORT_TYPE,GKQSORT_BASE,GKQSORT_NELT,GKQSORT_LT)   \
{									\
  GKQSORT_TYPE *const _base = (GKQSORT_BASE);				\
  const size_t _elems = (GKQSORT_NELT);					\
  GKQSORT_TYPE _hold;							\
									\
  if (_elems == 0)                                                      \
    return;                                                             \
                                                                        \
  /* Don't declare two variables of type GKQSORT_TYPE in a single	\
   * statement: eg `TYPE a, b;', in case if TYPE is a pointer,		\
   * expands to `type* a, b;' wich isn't what we want.			\
   */									\
									\
  if (_elems > _GKQSORT_MAX_THRESH) {					\
    GKQSORT_TYPE *_lo = _base;						\
    GKQSORT_TYPE *_hi = _lo + _elems - 1;				\
    struct {								\
      GKQSORT_TYPE *_hi; GKQSORT_TYPE *_lo;				\
    } _stack[_GKQSORT_STACK_SIZE], *_top = _stack + 1;			\
									\
    while (_GKQSORT_STACK_NOT_EMPTY) {					\
      GKQSORT_TYPE *_left_ptr; GKQSORT_TYPE *_right_ptr;		\
									\
      /* Select median value from among LO, MID, and HI. Rearrange	\
         LO and HI so the three values are sorted. This lowers the	\
         probability of picking a pathological pivot value and		\
         skips a comparison for both the LEFT_PTR and RIGHT_PTR in	\
         the while loops. */						\
									\
      GKQSORT_TYPE *_mid = _lo + ((_hi - _lo) >> 1);			\
									\
      if (GKQSORT_LT (_mid, _lo))					\
        _GKQSORT_SWAP (_mid, _lo, _hold);				\
      if (GKQSORT_LT (_hi, _mid))					\
        _GKQSORT_SWAP (_mid, _hi, _hold);				\
      else								\
        goto _jump_over;						\
      if (GKQSORT_LT (_mid, _lo))					\
        _GKQSORT_SWAP (_mid, _lo, _hold);				\
  _jump_over:;								\
									\
      _left_ptr  = _lo + 1;						\
      _right_ptr = _hi - 1;						\
									\
      /* Here's the famous ``collapse the walls'' section of quicksort.	\
         Gotta like those tight inner loops!  They are the main reason	\
         that this algorithm runs much faster than others. */		\
      do {								\
        while (GKQSORT_LT (_left_ptr, _mid))				\
         ++_left_ptr;							\
									\
        while (GKQSORT_LT (_mid, _right_ptr))				\
          --_right_ptr;							\
									\
        if (_left_ptr < _right_ptr) {					\
          _GKQSORT_SWAP (_left_ptr, _right_ptr, _hold);			\
          if (_mid == _left_ptr)					\
            _mid = _right_ptr;						\
          else if (_mid == _right_ptr)					\
            _mid = _left_ptr;						\
          ++_left_ptr;							\
          --_right_ptr;							\
        }								\
        else if (_left_ptr == _right_ptr) {				\
          ++_left_ptr;							\
          --_right_ptr;							\
          break;							\
        }								\
      } while (_left_ptr <= _right_ptr);				\
									\
     /* Set up pointers for next iteration.  First determine whether	\
        left and right partitions are below the threshold size.  If so,	\
        ignore one or both.  Otherwise, push the larger partition's	\
        bounds on the stack and continue sorting the smaller one. */	\
									\
      if (_right_ptr - _lo <= _GKQSORT_MAX_THRESH) {			\
        if (_hi - _left_ptr <= _GKQSORT_MAX_THRESH)			\
          /* Ignore both small partitions. */				\
          _GKQSORT_POP (_lo, _hi, _top);				\
        else								\
          /* Ignore small left partition. */				\
          _lo = _left_ptr;						\
      }									\
      else if (_hi - _left_ptr <= _GKQSORT_MAX_THRESH)			\
        /* Ignore small right partition. */				\
        _hi = _right_ptr;						\
      else if (_right_ptr - _lo > _hi - _left_ptr) {			\
        /* Push larger left partition indices. */			\
        _GKQSORT_PUSH (_top, _lo, _right_ptr);				\
        _lo = _left_ptr;						\
      }									\
      else {								\
        /* Push larger right partition indices. */			\
        _GKQSORT_PUSH (_top, _left_ptr, _hi);				\
        _hi = _right_ptr;						\
      }									\
    }									\
  }									\
									\
  /* Once the BASE array is partially sorted by quicksort the rest	\
     is completely sorted using insertion sort, since this is efficient	\
     for partitions below MAX_THRESH size. BASE points to the		\
     beginning of the array to sort, and END_PTR points at the very	\
     last element in the array (*not* one beyond it!). */		\
									\
  {									\
    GKQSORT_TYPE *const _end_ptr = _base + _elems - 1;			\
    GKQSORT_TYPE *_tmp_ptr = _base;					\
    register GKQSORT_TYPE *_run_ptr;					\
    GKQSORT_TYPE *_thresh;						\
									\
    _thresh = _base + _GKQSORT_MAX_THRESH;				\
    if (_thresh > _end_ptr)						\
      _thresh = _end_ptr;						\
									\
    /* Find smallest element in first threshold and place it at the	\
       array's beginning.  This is the smallest array element,		\
       and the operation speeds up insertion sort's inner loop. */	\
									\
    for (_run_ptr = _tmp_ptr + 1; _run_ptr <= _thresh; ++_run_ptr)	\
      if (GKQSORT_LT (_run_ptr, _tmp_ptr))				\
        _tmp_ptr = _run_ptr;						\
									\
    if (_tmp_ptr != _base)						\
      _GKQSORT_SWAP (_tmp_ptr, _base, _hold);				\
									\
    /* Insertion sort, running from left-hand-side			\
     * up to right-hand-side.  */					\
									\
    _run_ptr = _base + 1;						\
    while (++_run_ptr <= _end_ptr) {					\
      _tmp_ptr = _run_ptr - 1;						\
      while (GKQSORT_LT (_run_ptr, _tmp_ptr))				\
        --_tmp_ptr;							\
									\
      ++_tmp_ptr;							\
      if (_tmp_ptr != _run_ptr) {					\
        GKQSORT_TYPE *_trav = _run_ptr + 1;				\
        while (--_trav >= _run_ptr) {					\
          GKQSORT_TYPE *_hi; GKQSORT_TYPE *_lo;				\
          _hold = *_trav;						\
									\
          for (_hi = _lo = _trav; --_lo >= _tmp_ptr; _hi = _lo)		\
            *_hi = *_lo;						\
          *_hi = _hold;							\
        }								\
      }									\
    }									\
  }									\
									\
}

#endif