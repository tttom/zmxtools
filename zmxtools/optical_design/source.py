from zmxtools.optical_design.header import Medium
from zmxtools.optical_design.light import LightPath, Wavefront


class Source:
    """A base class for light-sources."""

    def __init__(self, medium: Medium, wavefront: Wavefront):
        """
        Constructs a new light source in the specified medium, that emits the specified wavefront.

        :param medium: The medium that the light-source is embedded in.
        :param wavefront: The wavefront it emits.
        """
        self.medium: Medium = medium
        self.wavefront = wavefront

    def emit(self) -> LightPath:
        """Returns the emitted light-path, let by the light-source wavefront."""
        return LightPath(self.wavefront)
