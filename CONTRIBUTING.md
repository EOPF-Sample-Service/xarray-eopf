# How to contribute

The xarray-eopf welcomes contributions of any form
as long as you respect our [code of conduct](CODE_OF_CONDUCT.md) and stay 
in line with the following instructions and guidelines.

If you have suggestions, ideas, feature requests, or if you have identified
a malfunction or error, then please 
[post an issue](https://github.com/EOPF-Sample-Service/xarray-eopf/issues). 

If you'd like to submit code or documentation changes, we ask you to provide a 
pull request (PR) 
[here](https://github.com/EOPF-Sample-Service/xarray-eopf/pulls). 
For code and configuration changes, your PR must be linked to a 
corresponding issue. 

To ensure that your code contributions are consistent with our project’s
coding guidelines, please make sure all applicable items of the following 
checklist are addressed in your PR.  

**PR checklist**

* Format and check code using [ruff](https://docs.astral.sh/ruff/) with 
  default settings: `ruff format` and `ruff check`. See also section 
  [code style](#code-style) below.
* Your change shall not break existing unit tests.
  `pytest` must run without errors.
* Add unit tests for any new code not yet covered by tests.
* Make sure test coverage stays close to 100% for any change.
  Use `pytest --cov=xarray_eopf --cov-report=html` to verify.
* If your change affects the current project documentation,
  please adjust it and include the change in the PR.
  Run `mkdocs serve` to verify. 

## Code style

The code style of xarray-eopf equals the default settings 
of [black](https://black.readthedocs.io/). Since black is 
un-opinionated regarding the order of imports, we group and 
sort imports statements according to the default settings of 
[isort](https://pycqa.github.io/isort/) which boils down to

0. Future imports
1. Python standard library imports, e.g., `os`, `typing`, etc
2. 3rd-party imports, e.g., `xarray`, `zarr`, etc
3. 1st-party library module imports using absolute paths, 
   e.g., `from xarray_eopf.a.b.c import d`. 
4. 1st-party library module imports from local modules: 
   Relative imports such as `from .c import d` are ok
   while `..c import d` are not ok.

Use `typing.TYPE_CHECKING` to resolve forward references 
and effectively avoid circular dependencies.

Package classes, functions, constants, type aliases considered public API 
should have docstrings according to the 
[Google Style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
