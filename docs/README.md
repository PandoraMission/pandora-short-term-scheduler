# Documentation Guide

This folder contains the documentation for the project, built using [MkDocs](https://www.mkdocs.org/). `mkdocs` and all the required dependencies for building the docs are installed when you install this project with dev dependencies.

!!! warning

    To make this work in your repo you will need to update your repo settings under `Settings/Pages/` so that docs are deployed from the`gh-pages` branch.

!!! info

    If your docs aren't building in the GitHub action, you should make sure your Settings -> Actions -> Workflow permissions are set to read and write.

## Writing Documentation

- Documentation is written in **Markdown (`.md`)** files.
- The main configuration file is `mkdocs.yml`, where you can define site settings, navigation structure, themes, and extensions.
- Place new documentation pages inside this `docs/` folder and link them in `mkdocs.yml`.

You should update the `mkdocs.yml` file to change the name, icon and URL for your repository, instead of this one.

## Building and Serving the Docs Locally

To preview the documentation locally, run:

```sh
poetry run mkdocs serve
```

or

```sh
make serve
```

from the main project directory (one above `docs/`).

This starts a local web server, typically at `http://127.0.0.1:8000/`, where you can view the documentation live.

## Deploying the Docs

MkDocs is set up to deploy to GitHub Pages via a GitHub Action. On each push to `main`, the documentation will be built and published to the `gh-pages` branch.

To manually deploy, run:

```sh
poetry run mkdocs gh-deploy --force
```

or

```sh
make deploy
```

This will build and push the site to GitHub Pages.

## Adding Code Documentation

This project uses `mkdocstrings` to generate API documentation directly from the source code. To document Python functions and classes:

- Ensure docstrings follow a consistent format (NumPy-style is recommended).
- Use `::: packagename.module` syntax inside Markdown to auto-generate API docs.

For more details, see the [MkDocs documentation](https://www.mkdocs.org/) and the [MkDocStrings plugin](https://mkdocstrings.github.io/).

### Documenting functions

You should add docstrings to your functions, and I recommend you use the `numpydoc` style for docstrings. You should fill out the sections as needed. In addition, I recommend you include type hints to help users understand your functions. This is the example function that is given in the package example module.

``` py
def example_function(
    x: Union[int, float, npt.NDArray],
) -> Union[int, float, npt.NDArray]:
    """Squares the input value.

    This function computes the square of a given number or NumPy array.

    Parameters
    ----------
    x : int, float, or npt.NDArray
        The input value(s) to be squared.

    Returns
    -------
    result : int, float, or npt.NDArray
        The squared result of `x`.

    Raises
    ------
    TypeError
        If `x` is not an int, float, or NumPy array.

    Notes
    -----
    This function uses `np.square(x)`, which is optimized for NumPy arrays.

    Examples
    --------
    Square an integer:
    >>> example_function(3)
    9

    Square a float:
    >>> example_function(2.5)
    6.25

    Square a NumPy array:
    >>> import numpy as np
    >>> arr = np.array([1, 2, 3])
    >>> example_function(arr)
    array([1, 4, 9])

    See Also
    --------
    numpy.square : Equivalent NumPy function for squaring elements.

    References
    ----------
    .. [1] NumPy Documentation: https://numpy.org/doc/stable/reference/generated/numpy.square.html
    """
    if not isinstance(x, (int, float, np.ndarray)):
        raise TypeError("Input must be an int, float, or NumPy array.")

    return np.square(x)
```
