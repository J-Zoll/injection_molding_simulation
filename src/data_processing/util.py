from typing import Iterable
import math


def discretize(step_size: float, xs: Iterable[float]) -> Iterable[float]:
    return [math.ceil(x / step_size) * step_size for x in xs]


def binary_aggregate(step_size: float, xs: Iterable[float], condition: str) -> Iterable[Iterable[bool]]:
    x_min = min(xs)
    x_max = max(xs)

    agg_xs = []
    step_value = x_min
    while step_value < x_max + step_size:
        if condition == "smaller_or_equal":
            xs_step = [math.isclose(x, step_value) or x < step_value for x in xs]
        else:
            raise NotImplementedError(f"Condition '{condition}' is not yet implemented")
        agg_xs.append(xs_step)
        step_value += step_size

    return agg_xs


def componentwise_distance(p1: Iterable[float], p2: Iterable[float]) -> Iterable[float]:
    return [abs(c1 - c2) for c1, c2 in zip(p1, p2)]


def parse_to_float_list(string: str) -> Iterable[float]:
    string = string[1:-1] # remove parenthesis
    str_numbers = string.split(", ")
    numbers = [float(f) for f in str_numbers]
    return tuple(numbers)
