# bindrcpp 0.2 (2017-06-15)

- Fixed very rare segmentation fault due to missing protection of function arguments in autogenerated boilerplate code.
- Fix compilation errors on FreeBSD due to use of nonstandard Make features (#5).
- Native symbol registration added by Rcpp.


# bindrcpp 0.1 (2016-12-08)

Initial CRAN release.

## Exported C++ functions

- `create_env_string()` creates an environment with active bindings, with names given as a character vector.  Access of these bindings triggers a call to a C++ function with a fixed signature (`GETTER_FUNC_STRING`); this call contains the name of the binding (as character) and an arbitrary payload (`PAYLOAD`, essentially a wrapped `void*`).
- `create_env_symbol()` is similar, the callback function accepts the name of the binding as symbol instead of
  character (`GETTER_FUNC_SYMBOL`).
- `populate_env_string()` and `populate_env_symbol()` populate an existing environment instead of creating a new one.
- Use `LinkingTo: bindrcpp` and `#include <bindrcpp.h>` to access these functions from your package.

## Exported R functions

- Reexported from `bindr`: `create_env()` and `populate_env()`.
