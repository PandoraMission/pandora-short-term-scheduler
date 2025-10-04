"""Example package module"""

# Standard library
from typing import Union

# Third-party
import numpy as np
import numpy.typing as npt


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
