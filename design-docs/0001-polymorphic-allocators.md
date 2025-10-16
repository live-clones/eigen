- Feature Name: Adding Polymorphic Allocators to Eigen types
- Start Date: April 29th, 2024
- Eigen Issue: [#2354](https://gitlab.com/libeigen/eigen/-/issues/2354)

# Summary
[summary]: #summary

Dynamically allocated Eigen storage types should be able to allocate their memory using a [polymorphic allocator](https://en.cppreference.com/w/cpp/memory/polymorphic_allocator/polymorphic_allocator).
Adding a polymorphic allocator inside of Eigen types would allow matrices to have user defined memory with different allocation strategies.

```cpp
Eigen::monotonic_buffer_resource mono_buff(2 << 16);
Eigen::polymorphic_allocator arena_alloc(&mono_buff)
Eigen::Index N = 10;
Eigen::Index M = 10;
// Constructor with sizes
Eigen::MatrixXd X(N, M, arena_alloc);
Eigen::MatrixXd Y(N, M, arena_alloc);
// Fill in X and Y

// Constructor with no size
Eigen::MatrixXd Z(arena_alloc);
Z = X * Y;

// Assign and use allocator on one line
Eigen::MatrixXd H = (X * Y).with_allocator(arena_alloc);

alloc.resource()->release();

```

## Motivation
[motivation]: #motivation

> Why are we doing this? What use cases does it support? What is the expected outcome?

Polymorphic memory resources allow developers to manage the memory of their program via allocators such as stacks, pools, or arenas.
Using memory management patterns efficiently can greatly improve performance. For example, if the user repeatedly allocates many small dynamically allocated matrices they can use `Eigen::monotonic_buffer_resource` to repeatedly reuse the same memory over and over.

Custom allocators are also important in real time or embedded systems that can minimize latency or memory fragmentation.
A polymorphic allocator scheme would allow users to use memory allocation strategies that suite their own needs.

## Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

> Explain the proposal as if it was already included in the project and you were teaching it to another Eigen programmer in the manual. That generally means:

> - Introducing new named concepts.
> - Explaining the feature largely in terms of examples.
> - Explaining how Eigen programmers should *think* about the feature, and how it should impact the way they use the relevant package. It should explain the impact as concretely as possible.
> - If applicable, provide sample error messages, deprecation warnings, or migration guidance.
> - If applicable, describe the differences between teaching this to existing Eigen programmers and new Eigen programmers.

> For implementation-oriented RFCs this section should focus on how contributors should think about the change, and give examples of its concrete impact. For policy RFCs, this section should provide an example-driven introduction to the policy, and explain its impact in concrete terms.

I'll be writing the guide level explanation as proposed documentation, which means some of the above will be reiterated.

--------------

### Allocator Aware Eigen Types

Since Eigen X.X, users are able to incorporate their own allocators to manage the memory of dense dynamic Eigen types.
When incorporating polymorphic allocators into your projects, think of them as a way to optimize and control memory usage finely.
By selecting appropriate allocators for the task at hand, you can reduce overhead and improve performance significantly.
The use of these allocators should be considered particularly in performance-critical applications where allocation/deallocation overhead can become a bottleneck.

### New Concepts

#### Memory resource

`Eigen::memory_resource` is an abstract class that encapsulates memory resources.
User defined memory resources must inherit from this class. See the information below on custom memory resources for an example memory resource class.
`Eigen::memory_resource` is directly based on [`std::pmr::memory_resource`](https://en.cppreference.com/w/cpp/memory/memory_resource) and many of the docs there will also be useful when working with Eigen's memory management.

#### Polymorphic Allocator

`Eigen::polymorphic_allocator` is an interface class, allowing dynamic memory allocation to be customized by user-defined strategies.
The design of `Eigen::polymorphic_allocator` is based on [`std::pmr::polymorphic_allocator`](https://en.cppreference.com/w/cpp/memory/polymorphic_allocator), but with type erasure.
Unlike, `std::pmr::polymorphic_allocator`, there is no template to specify what this allocator should allocate. Instead, types are provided when performing the allocation inside of Eigen types.

```c++

Eigen::monotonic_buffer mem;
Eigen::polymorphic_allocator x(&mem);
double* p = x.template allocate<double>(10);
x.deallocate(p, 10);
```

#### Allocator-aware Eigen Types

- Matrix and vector types in Eigen that can be constructed with a custom allocator, allowing them to use user-specified memory resources.

### Example: Monotonic Allocator

Below is an example program that performs gradient descent for a least squares problem.
It uses `Eigen::monotonic_buffer_resource` to reuse the memory for each gradient step.

```c++
#include <Eigen/Dense>
#include <Eigen/Memory>


// Usage of the custom allocator
int main() {
  MatrixXd X(5, 2);
  X << 1, 1,
        2, 1,
        3, 1,
        4, 1,
        5, 1;
  VectorXd y(5);
  y << 1, 2, 3, 4, 5;

  // Parameters for gradient descent
  Vector<double, 2, 1> theta;      // [m, b] assuming y = mx + b
  theta << 0, 0;          // Initial guess
  double alpha = 0.01;    // Learning rate
  int iterations = 1000;  // Number of iterations
  // Set the initial size for the monotonic buffer
  Eigen::monotonic_buffer_resource mono_buff(2 << 16);
  Eigen::polymorphic_allocator alloc(&mono_buff)
  Eigen::monotonic_buffer_resource mono_buff(2 << 16);
  Eigen::polymorphic_allocator scratch_alloc(&mono_buff)
  Eigen::VectorXd res_err = std::numeric_limits<double>::infinity()
  double abs_err = 1e-8;

  // Gradient descent
  for (int i = 0; i < iterations; ++i) {
    // Use mono buffer for intermediate memory
    VectorXd predictions(5, alloc);
    predictions = X * theta;
    // Can also use `with_allocator()` to set the allocator
    VectorXd errors = (predictions - y).with_allocator(alloc, scratch_alloc);
    VectorXd gradient = (X.transpose() * errors).with_allocator(alloc, scratch_alloc);
    theta -= (alpha * gradient).with_allocator(alloc, scratch_alloc);
    // Reset the monotonic buffer to reuse the memory
    alloc.resource()->release();
  }
  return 0;
}
```

## Custom Allocators

Eigen supports runtime customizable memory management strategies for dynamically allocated Eigen types.
This feature allows developers to specify how memory for Eigen's dynamic storage types is allocated and deallocated.
Using custom allocators enables more control over memory usage patterns and optimizations specific to application requirements.

### Memory Resources

Memory resources are defined using an abstract class to manage access to the underlying memory.
Custom Memory resource classes should inherit from `Eigen::memory_resource` and have the following structure.

```c++
class my_memory_resource : public Eigen::memory_resource {
  public:
  /**
   * Allocates storage with a size of at least bytes bytes,
   * aligned to the specified alignment.
   * Equivalent to `do_allocate(bytes, alignment);``.
   */
  void* allocate(std::size_t bytes,
    std::size_t alignment = std::max(alignof(std::max_align_t), EIGEN_ALIGN_MAX));
  /**
   * Deallocates the storage pointed to by p. p shall have been
   * returned by a prior call to `allocate(bytes, alignment)`
   * on a `memory_resource`` that compares equal to `*this`,
   * and the storage it points to shall not yet have been
   * deallocated.
   * Equivalent to do_deallocate(p, bytes, alignment);
   */
  void deallocate(void* p,
    std::size_t bytes,
    std::size_t alignment = std::max(alignof(std::max_align_t), EIGEN_ALIGN_MAX) );

  /**
   * Compares `*this` with `other` for equality.
   * Two memory_resources compare equal if and only if memory allocated
   * from one memory_resource can be deallocated from the other
   * and vice versa.
   */
  bool is_equal( const my_memory_resource& other ) const noexcept;
  private:

  /**
   * Allocates storage with a size of at least bytes bytes, aligned to * the specified alignment. Alignment shall be a power of two.
   * @throw Throws an exception if storage of the requested size and alignment cannot be obtained.
   */
  virtual void* do_allocate(std::size_t bytes,
    std::size_t alignment );

  /**
   * Deallocates the storage pointed to by p.
   * p must have been returned by a prior call to
   * `allocate(bytes, alignment)` on a memory_resource that compares
   * equal to *this, and the storage it points to must not yet have
   * been deallocated, otherwise the behavior is undefined.
   */
  virtual void do_deallocate(void* p, std::size_t bytes,
    std::size_t alignment );

  /**
   * Compares *this for equality with other.
   * Two memory_resources compare equal if and only if memory allocated
   * from one memory_resource can be deallocated from the other and vice
   * versa.
   */
  virtual bool do_is_equal(const my_memory_resource& other ) const noexcept;
};

  /**
   * Compares the memory_resources a and b for equality.
   * Two memory_resources compare equal if and only if memory allocated
   * from one memory_resource can be deallocated from the other
   * and vice versa.
   */
  bool operator==(const my_memory_resource& a,
    const my_memory_resource& b ) noexcept;
  bool operator!=(const my_memory_resource& a,
    const my_memory_resource& b ) noexcept;

```

For a more direct example see `Eigen::monotonic_buffer_resource` which is a special purpose memory resource class that releases the allocated memory only when the resource is destroyed.

### Impact on Usage

For existing Eigen users, the introduction of polymorphic allocators is backward compatible but offers a new dimension of control over matrix and vector memory management. Users should start by understanding the default behavior and then incrementally integrate custom allocators where beneficial.

For new users, understanding the allocator model early on can provide a deeper insight into how Eigen manages memory and how it can be tailored to meet specific performance and memory usage requirements.

### Error Messages and Migration Guidance
Error messages related to incorrect usage of allocators will typically involve type mismatches or allocation failures.

For an allocator that does not conform to the Eigen polymorphic allocator interface (tested at compile time given a `is_eigen_allocator_compatible` type trait), a compile time error will be thrown such as

```
Error: Allocator does not have the necessary member {NAME_OF_MEMBER}
```

If you encounter this, ensure that your custom allocator correctly inherits from `Eigen::polymorphic_allocator` and implements all required functions.

In general, allocator errors should conform to the same errors given by `std::pmr::polymorphic_allocator`


# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

> This is the technical portion of the RFC. Explain the design in sufficient detail that:

> - Its interaction with other features is clear.
> - It is reasonably clear how the feature would be implemented.
> - Corner cases are dissected by example.

> The section should return to the examples given in the previous section, and explain more fully how the detailed proposal makes those examples work.

There's a good bit of things that need to be added

1. `Eigen::memory_resource`
    - Will be inherited by all user defined memory resources.
2. `Eigen::polymorphic_allocator`
    - Will be inherited by all user defined allocators. If the user is compiling with C++17 enabled this will be an alias for `std::pmr::polymorphic_allocator`. Otherwise a custom implementation of `Eigen::polymorphic_allocator` will be available that complies with the `std::pmr::polymorphic_allocator` definition in the C++17 standard.

An `Eigen::polymorphic_allocator` will be added inside of `DenseStorage`

```cpp
// purely dynamic matrix.
template <typename T, int Options_>
class DenseStorage<T, Dynamic, Dynamic, Dynamic, Options_> {
  Eigen::polymorphic_allocator m_allocator;
// ...
```

`DenseStorage` can either have new constructors or we can add `Eigen::polymorphic_allocator` to the current constructors with a default value of `Eigen::default_allocator()`, which is an `Eigen::polymorphic_allocator` that uses Eigen's `malloc` and `free` implementations.

```cpp
 public:
  EIGEN_DEVICE_FUNC constexpr DenseStorage() : m_allocator(Eigen::default_allocator{}),
    m_data(0), m_rows(0), m_cols(0) {}

  EIGEN_DEVICE_FUNC explicit constexpr DenseStorage(internal::constructor_without_unaligned_array_assert)
      : m_allocator(Eigen::default_allocator{}), m_data(0), m_rows(0), m_cols(0) {}

  EIGEN_DEVICE_FUNC DenseStorage(Index size, Index rows, Index cols, Eigen::polymorphic_allocator& allocator)
      : m_allocator(allocator),
        m_data(internal::allocate<T, Options_>(size, m_allocator)),
        m_rows(rows),
        m_cols(cols) {
    EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN({})
    eigen_internal_assert(size == rows * cols && rows >= 0 && cols >= 0);
  }

  EIGEN_DEVICE_FUNC DenseStorage(const DenseStorage& other)
      : m_alloactor(other.m_allocator),
        m_data(internal::allocate<T, Options_>(other.m_rows * other.m_cols, m_allocator)),
        m_rows(other.m_rows),
        m_cols(other.m_cols) {
    EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN(Index size = m_rows * m_cols)
    internal::smart_copy(other.m_data, other.m_data + other.m_rows * other.m_cols, m_data);
  }

```

There should not be any changes need in functions such as `call_assignment` and friends as the left hand side will have already been allocated via an allocator or the left hand side will be initialized by the right hand side.

Eigen's `polymorphic_allocator` will have an interface that follows the C++17 standards `std::pmr::polymorphic_allocator`, but with the exception of not having a template parameter for the class. Using type erasure will make it easier to allow mixed scalar type operations.

```c++
class polymorphic_allocator {
  /**
   * Constructs a polymorphic_allocator using the return
   * value of Eigen::get_default_resource() as the
   * underlying memory resource.
   */
  polymorphic_allocator() noexcept;

  /**
   * Constructs a polymorphic_allocator using
   * `other.resource()` as the underlying memory resource.
   */
  polymorphic_allocator( const polymorphic_allocator& other ) = default;

  /**
   * Constructs a polymorphic_allocator using `r`` as the
   * underlying memory resource. This constructor
   * provides an implicit conversion from
   * `Eigen::memory_resource*`.
   */
  polymorphic_allocator(Eigen::memory_resource* r );

  /**
   * Allocates storage for n objects of type T using the underlying memory resource.
   * @param n	the number of objects to allocate storage for
   * @return A pointer to the allocated storage.
   */
  template <typename T>
  T* allocate(std::size_t n, std::size_t alignment = std::max(alignof(std::max_align_t), EIGEN_ALIGN_MAX));

  /**
   * Deallocates the storage pointed to by p, which must have
   * been allocated from a std::pmr::memory_resource x that
   * compares equal to `*resource()`` using
   *  `x.allocate(n * sizeof(T), alignof(T))`.
   */
  template <typename T>
  void deallocate( T* p, std::size_t n,
    std::size_t alignment = std::max(alignof(std::max_align_t), EIGEN_ALIGN_MAX));

  /**
   * Creates an object of the given type U by means of
   * uses-allocator construction at the uninitialized memory
   * location indicated by p, using *this as the allocator.
   * @tparam U type of allocated storage
   * @tparam Args Types of constructor arguments for `U`
   * @param p pointer to allocated, but not initialized storage
   * @param args the constructor arguments to pass to the
   * constructor of T
   */
  template<typename U, typename... Args >
  void construct(U* p,
    std::size_t alignment = std::max(alignof(std::max_align_t), EIGEN_ALIGN_MAX),
    Args&&... args );

  /**
   * Destroys the object pointed to by p, as if by calling p->~U().
   * @tparam U type of allocated storage
   * @tparam Args Types of constructor arguments for `U`
   * @param p	pointer to the object being destroyed
   */
  template< class U >
  void destroy(U* p );

  /**
   * Allocates nbytes bytes of storage at specified alignment alignment using the
   * underlying memory resource. Equivalent to
   * `return resource()->allocate(nbytes, alignment);`
   * @param nbytes The number of bytes to allocate
   * @param alignment The alignment to use
   */
  void* allocate_bytes( std::size_t nbytes,
        std::size_t alignment = std::max(alignof(std::max_align_t), EIGEN_ALIGN_MAX));

  /**
   * Deallocates the storage pointed to by p, which must have been
   * allocated from a `Eigen::memory_resource` x that compares equal
   * to `*resource()`, using `x.allocate(nbytes, alignment)`,
   * typically through a call to `allocate_bytes(nbytes, alignment)`.
   * `Equivalent to resource()->deallocate(p, nbytes, alignment);`.
   *
   * @param p pointer to memory to deallocate
   * @param nbytes The number of bytes originally allocated
   * @param alignment The alignment originally allocated
   */
  void deallocate_bytes(void* p,
    std::size_t nbytes,
    std::size_t alignment = std::max(alignof(std::max_align_t), EIGEN_ALIGN_MAX));
};
```

A new member function `with_allocator` will be added that accepts either one or two allocators. In the of one argument, any temporaries made in the expression will also be allocated with the allocator and the end destination matrix will be allocated in the allocator.

```c++
VectorXd gradient = (X.transpose() * errors).with_allocator(alloc);
```

For the two argument version of `with_allocator`, the second argument will be an allocator used for temporary allocations and will be cleared after the expression and assignment have finished.

```c++
VectorXd gradient = (X.transpose() * errors).with_allocator(alloc, scratch_allocator);
```


# Drawbacks
[drawbacks]: #drawbacks

> Why should we *not* do this?

In theory users can already do something very similar to this with `Eigen::Map<>`

```c++
Eigen::Map<Eigen::MatrixXd> x(arena_allocator.allocate(M * N), M, N);
```

We could just formalize that form and allow maps to generate other maps using the map expressions allocator.

```c++
// Allocates M * N space
Eigen::Map<Eigen::MatrixXd> x(arena_allocator, M, N);
Eigen::Map<Eigen::MatrixXd> y(arena_allocator, N, M);
// let map operations make a new map using the allocator
Eigen::Map<Eigen::MatrixXd> z = x * y
```

# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

> - Why is this design the best in the space of possible designs?

I like this design because the polymorphic scheme matches up nicely with `std::pmr` which C++ users should already be used to.

> - What other designs have been considered and what is the rationale for not choosing them?

Instead of using polymorphic allocators, Eigen could use compile time known allocators. Eigen could also choose a mix of both.
A new template `Allocator` could be added to Eigen types with a default of `Eigen::polymorphic_alloctor`.
Standard library containers have a default of `std::allocator` so that is why they need to have the `std::pmr` namespace.
But Eigen could just use a polymorphic allocator as it's default template type that users could choose to overload.

> - What is the impact of not doing this?

Users will have to do some hacky'ish things to get non-standard memory allocation.

# Prior art
[prior-art]: #prior-art

> Discuss prior art, both the good and the bad, in relation to this proposal.
A few examples of what this can include are:

> This section is intended to encourage you as an author to think about the lessons from other languages, provide readers of your RFC with a fuller picture.
> If there is no prior art, that is fine - your ideas are interesting to us whether they are brand new or if it is an adaptation from other languages.

> Note that while precedent set by other languages is some motivation, it does not on its own motivate an RFC.
> Please also take into consideration that rust sometimes intentionally diverges from common language features.

The main precedent here is c++17's `std::pmr` namespace. This design doc is pretty directly based on that.

# Unresolved questions
[unresolved-questions]: #unresolved-questions

> - What parts of the design do you expect to resolve through the RFC process before this gets merged?
> - What parts of the design do you expect to resolve through the implementation of this feature before stabilization?
> - What related issues do you consider out of scope for this RFC that could be addressed in the future independently of the solution that comes out of this RFC?

-------------------

Do we want this for sparse matrices as well?
