from abc import ABC, abstractmethod
import random
import madcad
import geometry as g


class Study:
    def __init__(
        self,
        name,
        geometry,
        injection_locations
    ) -> None:
        self.name = name
        self.geometry = geometry
        self.injection_locations = injection_locations


class StudyGenerator (ABC):
    @abstractmethod
    def generate_study(self) -> Study:
        pass


class PlateWithHole (StudyGenerator):
    PLATE_WIDTH, PLATE_HEIGHT, PLATE_THICKNESS = 100, 100, 5
    HOLE_RADIUS, HOLE_DEPTH, HOLE_PADDING = 15, PLATE_THICKNESS, 10
    NAME_TEMPLATE = "plate_{study_index:06d}"
    INJECTION_DIRECTION = (0, 0, 1)

    def __init__(self, start_index=0) -> None:
        super().__init__()
        self._study_index = start_index


    def generate_study(self) -> Study:
        # set name
        name = self.NAME_TEMPLATE.format(study_index=self._study_index)
        self._study_index += 1

        # get random hole position
        max_hole_x = self.PLATE_WIDTH - 2 * self.HOLE_PADDING - 2 * self.HOLE_RADIUS
        hole_x = self.HOLE_PADDING + self.HOLE_RADIUS + random.randint(0, max_hole_x)
        max_hole_y = self.PLATE_HEIGHT - 2 * self.HOLE_PADDING - 2 * self.HOLE_RADIUS
        hole_y = self.HOLE_PADDING + self.HOLE_RADIUS + random.randint(0, max_hole_y)

        # generate geometry
        plate = g.build_plate(self.PLATE_WIDTH, self.PLATE_HEIGHT, self.PLATE_THICKNESS)
        hole = g.build_cylinder(hole_x, hole_y, self.HOLE_DEPTH, self.HOLE_RADIUS)
        geometry = madcad.difference(plate, hole)

        # get random injection location
        while True:
            inj_x = random.randint(0, self.PLATE_WIDTH)
            inj_y = random.randint(0, self.PLATE_HEIGHT)

            if not g.is_on_circle(inj_x, inj_y, hole_x, hole_y, self.HOLE_RADIUS):
                break

        inj_z = 0
        inj_locations = [((inj_x, inj_y, inj_z), self.INJECTION_DIRECTION)]

        return Study(name, geometry, inj_locations)


generators = {
    "PlateWithHole": PlateWithHole()
}
