"""Functions to define basic CAD building blocks."""

from typing import Union
import math

from madcad import Axis, Circle, Segment, extrusion, flatsurface, thicken, vec3, web
from stl import Mesh


def build_plate(x, y, z) -> Mesh:
    """Build plate.
    Parameters
    ----------
    x : float
        Dimension in x direction.
    y : float
        Dimension in y direction.
    z :  float
        Dimension in z direction.
    Returns
    -------
    mesh
        A plate object.
    """
    A = vec3(0, 0, 0)
    B = vec3(x, 0, 0)
    line = web([Segment(A, B)])
    face = extrusion(vec3(0, y, 0), line)
    plate = thicken(face, z)
    return plate


def build_cylinder(x, y, h, r) -> Mesh:
    """Build cylinder.
    Parameters
    ----------
    x : float
        Position in x.
    y : float
        Position in y.
    h : float
        Cylinder height.
    r : float
        Cylinder radius.
    Returns
    -------
    mesh
        A cylinder object.
    """
    A = vec3(x, y, h)
    B = vec3(0.0, 0.0, 1.0)
    axis = Axis(A, B)
    circle = flatsurface(Circle(axis, r))
    cylinder = thicken(circle, 3 * h)
    return cylinder


def is_on_circle(
    p_x: Union[int, float],
    p_y: Union[int, float],
    c_x: Union[int, float],
    c_y: Union[int, float],
    r: Union[int, float]
) -> bool:
    """Returns whether a 2D-point is on a circle
    Args:
        p_x: x-coordinate of the point
        p_y: y-coordinate of the point
        c_x: x_coordinate of the center of the circle
        c_y: y-coordinate of the center of the circle
        r: radius of the circle
    Returns:
        Whether the point is on the circle
    """
    d_x = abs(p_x - c_x)
    d_y = abs(p_y - c_y)
    d = math.sqrt(d_x ** 2 + d_y ** 2)

    return d <= r
