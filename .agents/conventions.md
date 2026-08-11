# Local Conventions For New Code

Use this guide when writing new declarations anywhere in the tree. It records the forms review asks for, so that a
patch does not spend a round on them. It does not authorize rewriting code the task is not otherwise changing: rule 5
in the repository-root `AGENTS.md` still governs untouched lines.

Eigen predates most of these forms, so a tree-wide count is not the convention. `enum` blocks still outnumber
`static constexpr` members, and `typedef` still outnumbers `using`. New code uses the current form, and a file you are
already editing heavily should come out uniform rather than half converted.

## Declarations

- Traits constants are `static constexpr` members, not `enum` blocks. Give each one the type it is used as:
  `Flags` is a bitmask, so `static constexpr unsigned int Flags`, and a predicate is `static constexpr bool`.
- Prefer `using` to `typedef` in new aliases. Where a file is uniformly `typedef`, matching it is reasonable for a few
  added lines; say so rather than mixing both in one block.
- Use `nullptr`, not `NULL` or `0`.
- Use `= default` and default member initializers instead of an empty constructor body that assigns each member.
- Reach for Eigen's own metaprogramming aliases before the standard ones spelled out: `bool_constant<C>` over
  `std::integral_constant<bool, C>`, `void_t`, `remove_all_t`, `internal::is_arithmetic`. They are in
  `Eigen/src/Core/util/Meta.h`.
- Put SFINAE in a defaulted template parameter rather than in the return type, following the local spelling of the
  surrounding file. When a constrained overload set covers the negative case too, constrain both overloads: a single
  constrained function plus an unconstrained one can bind a converted temporary and return a reference to it.
- An in-class function definition is already implicitly `inline`, so a bare `inline` there carries no meaning. Use
  `EIGEN_STRONG_INLINE` or `EIGEN_ALWAYS_INLINE` when the inlining matters, and nothing otherwise.
- A function-local `constexpr` scalar does not need `static`.
- Spell names out. `scratch`, not `scr`. Name a trait or alias for the property it asserts, so that
  `require_host_scalar_convertible_t` reads at the use site where `enable_scalar_arg_t` does not.

## The C++14 baseline

Supported headers compile as C++14, which rules out several forms that a review suggestion may reach for:

- `if constexpr` is C++17. Use `EIGEN_IF_CONSTEXPR(...)`, which lowers to a plain `if` on older standards. Use it
  wherever the condition is a compile-time constant, both for the codegen and because it documents the intent.
- Designated initializers are C++20 and warn under `-pedantic` earlier. Use aggregate assignment with `/*name=*/`
  comments. When a field's value derives from another field of the same object, compute both from locals: the braced
  initializer is built before the assignment, so it cannot read the fields it is about to set.
- `std::span` and the rest of C++17/20 library additions are unavailable. `Eigen/src/Core/util/Meta.h` and
  `Eigen/src/Core/MathFunctions.h` carry the backfills Eigen relies on.

A guarded backend with a documented newer requirement may use newer forms inside its guard. The SYCL configurations
select C++17 (`cmake/SyclConfigureTesting.cmake`, `cmake/FindDPCPP.cmake`, `cmake/FindTriSYCL.cmake`), so
`EIGEN_IF_CONSTEXPR` genuinely discards the untaken branch there and can be used to keep an unsupported primitive from
being instantiated for that backend.

## Comments

The comment rules in the repository-root `AGENTS.md` are part of review, not a suggestion, and they are the most
repeated style finding in this repository's history. Before publishing a diff, reread each comment it adds and delete
the ones that narrate the code, restate an identifier, or repeat a lifetime rule already stated at the declaration.
Keep the ones that record mathematics, an invariant, a compatibility constraint, provenance, or the reason a slower
form was chosen; a comment that stops the next contributor from "simplifying" a deliberate construct is worth its
lines. State that reason where the construct is, not in the merge request.
