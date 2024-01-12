from zmxtools.optical_design.header import Medium
from zmxtools.optical_design.light import Wavefront, LightPath


class Source:
    def __init__(self, medium: Medium, wavefront: Wavefront):
        self.medium: Medium = medium
        self.wavefront = wavefront

    def emit(self) -> LightPath:
        return LightPath(self.wavefront)
