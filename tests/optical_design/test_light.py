import numpy as np
import pytest
from numpy import testing as npt

from zmxtools.optical_design.geometry import IDENTITY, Translation
from zmxtools.optical_design.light import LightPath, Wavefront
from zmxtools.optical_design.material import VACUUM
from zmxtools.optical_design.medium import HomogeneousMedium, InhomogeneousMedium
from zmxtools.optical_design.optic import CompoundElement, Detector, OpticalDesign, SurfaceDetector
from zmxtools.optical_design.source import Source
from zmxtools.optical_design.surface import (
    AnalyticSurface,
    Aperture,
    DiskAperture,
    Interface,
    PlanarSurface,
    SnellInterface,
    SphericalSurface,
    Surface,
)


def make_wavefront():
    """Create a simple wavefront of a single ray traveling along z."""
    return Wavefront(
        electric_field=(1, 0, 0),
        magnetizing_field=(0, 1, 0),
        position=(0, 0, 0),
        direction=(0, 0, 1),
        k0=2 * np.pi / 550e-9,
    )


class TestWavefront:
    def setup_method(self):
        self.w = make_wavefront()

    def test_properties(self):
        npt.assert_allclose(self.w.position, [0, 0, 0])
        npt.assert_allclose(self.w.direction, [0, 0, 1])
        npt.assert_allclose(self.w.electric_field, [1, 0, 0])
        npt.assert_allclose(self.w.magnetizing_field, [0, 1, 0])
        npt.assert_allclose(self.w.k, [0, 0, 2 * np.pi / 550e-9])
        npt.assert_allclose(self.w.k0, 2 * np.pi / 550e-9)
        npt.assert_allclose(self.w.ct, 0)
        npt.assert_allclose(self.w.refractive_index, 1)
        npt.assert_array_equal(self.w.shape, ())
        npt.assert_equal(self.w.size, 1)
        npt.assert_equal(self.w.ndim, 0)

    def test_derived_direction(self):
        w = Wavefront(electric_field=(1, 0, 0), magnetizing_field=(0, 1, 0), position=(0, 0, 0), k=(0, 0, 2.0))
        npt.assert_allclose(w.direction, [0, 0, 1])
        npt.assert_allclose(w.k0, 2.0)

    def test_to(self):
        w = self.w.to(Translation((1, 2, 3)))
        npt.assert_allclose(w.position, [1, 2, 3])
        npt.assert_allclose(w.direction, [0, 0, 1])


class TestLightPath:
    def setup_method(self):
        self.lp = LightPath(make_wavefront())

    def test_wavefronts(self):
        npt.assert_equal(len(self.lp.wavefronts), 1)
        npt.assert_equal(self.lp.wavefront, self.lp.wavefronts[-1])

    def test_append(self):
        w2 = make_wavefront()
        self.lp.append(w2)
        npt.assert_equal(len(self.lp.wavefronts), 2)

    def test_propagate(self):
        self.lp.propagate(2.0)
        npt.assert_equal(len(self.lp.wavefronts), 2)
        npt.assert_allclose(self.lp.wavefront.position, [0, 0, 2])
        npt.assert_allclose(self.lp.wavefront.ct, 2.0)

    def test_interact(self):
        self.lp.interact(direction=(0, 0, 1))
        npt.assert_equal(len(self.lp.wavefronts), 2)
        npt.assert_allclose(self.lp.wavefront.direction, [0, 0, 1])
        self.lp.interact(k=(0, 0, 1.0), direction=(0, 0, 1), electric_field=(2, 0, 0), magnetizing_field=(0, 2, 0))
        npt.assert_allclose(self.lp.wavefront.electric_field, [2, 0, 0])

    def test_interact_k_from_direction(self):
        self.lp.interact(direction=(0, 0, 2))
        npt.assert_allclose(self.lp.wavefront.k / np.linalg.norm(self.lp.wavefront.k), [0, 0, 1])

    def test_to(self):
        self.lp.to(Translation((0, 0, 5)))
        npt.assert_allclose(self.lp.wavefront.position, [0, 0, 5])


class TestInterfaces:
    def test_interface(self):
        i = Interface()
        with pytest.raises(NotImplementedError):
            i.refract_into(LightPath(make_wavefront()), HomogeneousMedium(VACUUM))
        npt.assert_equal(repr(i).startswith("Interface("), True)
        i.to(Translation((1, 0, 0)))
        npt.assert_array_equal(i.transform.matrix[1:, 0], [1, 0, 0])

    def test_interface_reflect(self):
        i = Interface()
        lp = LightPath(make_wavefront())
        reflected = i.reflect(lp)
        npt.assert_allclose(reflected.wavefront.direction, [0, 0, -1])

    def test_snell_interface(self):
        si = SnellInterface()
        lp = LightPath(make_wavefront())
        medium = HomogeneousMedium(VACUUM)
        refracted = si.refract_into(lp, medium)
        npt.assert_allclose(refracted.wavefront.direction, [0, 0, 1], atol=1e-12)

    def test_aperture(self):
        a = Aperture()
        lp = LightPath(make_wavefront())
        npt.assert_equal(a.transmit(lp), lp)
        a.to(IDENTITY)
        npt.assert_equal(repr(a).startswith("Aperture("), True)

    def test_disk_aperture(self):
        da = DiskAperture(outer_radius=1.0, inner_radius=0.1)
        npt.assert_allclose(da.outer_radius, 1.0)
        npt.assert_allclose(da.inner_radius, 0.1)
        lp = LightPath(make_wavefront())
        transmitted = da.transmit(lp)
        npt.assert_allclose(transmitted.wavefront.electric_field, [0, 0, 0])  # on-axis point is inside the hole
        npt.assert_equal(repr(da).startswith("DiskAperture("), True)


class TestSurfaces:
    def test_surface(self):
        s = Surface(interface=SnellInterface())
        lp = LightPath(make_wavefront())
        medium = HomogeneousMedium(VACUUM)
        out = s.transmit_to(lp, medium)
        npt.assert_allclose(out.wavefront.direction, [0, 0, 1], atol=1e-12)
        npt.assert_equal(repr(s).startswith("Surface("), True)

    def test_analytic_surface(self):
        a = AnalyticSurface(interface=SnellInterface())
        with pytest.raises(NotImplementedError):
            a.distance(make_wavefront())

    def test_planar_surface(self):
        ps = PlanarSurface(interface=SnellInterface())
        w = make_wavefront()
        npt.assert_allclose(ps.distance(w), 0)
        w2 = Wavefront(electric_field=(1, 0, 0), magnetizing_field=(0, 1, 0), position=(0, 0, -2), direction=(0, 0, 1))
        npt.assert_allclose(ps.distance(w2), 2)

    def test_spherical_surface(self):
        ss = SphericalSurface(curvature=0.0)
        npt.assert_equal(ss.radius_of_curvature, np.inf)
        w = make_wavefront()
        npt.assert_allclose(ss.distance(w), 0, atol=1e-12)

        ss2 = SphericalSurface(curvature=0.01, interface=SnellInterface())
        npt.assert_allclose(ss2.radius_of_curvature, 100)
        d = ss2.distance(w)
        npt.assert_allclose(d, 0, atol=1e-12)

        # Default interface
        ss3 = SphericalSurface(curvature=0.0)
        npt.assert_equal(isinstance(ss3.interface, Interface), True)


class TestMedia:
    def test_homogeneous_medium(self):
        m = HomogeneousMedium(VACUUM)
        npt.assert_allclose(
            m.complex_refractive_index(wavenumber=1e6, position=0, electric_field=0, magnetizing_field=0),
            1,
        )
        npt.assert_equal(repr(m).startswith("HomogeneousMedium("), True)

    def test_propagate_to(self):
        m = HomogeneousMedium(VACUUM)
        lp = LightPath(
            Wavefront(
                electric_field=(1, 0, 0),
                magnetizing_field=(0, 1, 0),
                position=(0, 0, -1),
                direction=(0, 0, 1),
                k0=1.0,
            ),
        )
        ps = PlanarSurface(interface=SnellInterface())
        out = m.propagate_to(lp, ps)
        npt.assert_allclose(out.wavefront.position, [0, 0, 0], atol=1e-12)

    def test_inhomogeneous_medium(self):
        m = InhomogeneousMedium()
        npt.assert_equal(m.transform is not None, True)


class TestOptic:
    def setup_method(self):
        self.surface1 = PlanarSurface(interface=SnellInterface(), transform=Translation((0, 0, 0)))
        self.surface2 = PlanarSurface(interface=SnellInterface(), transform=Translation((0, 0, 1)))
        self.medium = HomogeneousMedium(VACUUM)

    def test_compound_element(self):
        ce = CompoundElement(self.surface1, self.medium, self.surface2)
        npt.assert_equal(len(ce.elements), 2)
        npt.assert_equal(len(ce.media), 1)
        w = Wavefront(electric_field=(1, 0, 0), magnetizing_field=(0, 1, 0), position=(0, 0, 0), direction=(0, 0, 1))
        npt.assert_allclose(ce.distance(w), 0)
        npt.assert_equal(repr(ce).startswith("CompoundElement("), True)

    def test_compound_element_transmit(self):
        ce = CompoundElement(self.surface1, self.medium, self.surface2)
        lp = LightPath(
            Wavefront(
                electric_field=(1, 0, 0),
                magnetizing_field=(0, 1, 0),
                position=(0, 0, 0),
                direction=(0, 0, 1),
                k0=1.0,
            ),
        )
        out = ce.transmit_to(lp, self.medium)
        npt.assert_equal(len(out.wavefronts) >= 3, True)

    def test_detector(self):
        m = HomogeneousMedium(VACUUM)
        d = Detector(medium=m)
        npt.assert_equal(d.medium, m)
        s = PlanarSurface(interface=SnellInterface())
        sd = SurfaceDetector(medium=m, surface=s)
        npt.assert_equal(sd.surface, s)

    def test_optical_design(self):
        m = HomogeneousMedium(VACUUM)
        source = Source(medium=m, wavefront=make_wavefront())
        s = PlanarSurface(interface=SnellInterface())
        od = OpticalDesign(source=source, optic=s, detector=Detector(medium=m))
        npt.assert_equal(od.optic, s)
        npt.assert_array_equal(source.emit().wavefront.direction, [0, 0, 1])


class TestHeader:
    def test_element_not_implemented(self):
        from zmxtools.optical_design.header import Element

        e = Element()
        lp = LightPath(make_wavefront())
        with pytest.raises(NotImplementedError):
            e.transmit_to(lp, HomogeneousMedium(VACUUM))
        with pytest.raises(NotImplementedError):
            e.reflect(lp)
        with pytest.raises(NotImplementedError):
            e.distance(make_wavefront())
        npt.assert_equal(repr(e).startswith("Element("), True)

    def test_medium_not_implemented(self):
        from zmxtools.optical_design.header import Medium

        m = Medium()
        lp = LightPath(make_wavefront())
        with pytest.raises(NotImplementedError):
            m.propagate_to(lp, PlanarSurface(interface=SnellInterface()))
        with pytest.raises(NotImplementedError):
            m.complex_refractive_index(wavenumber=1, position=0, electric_field=0, magnetizing_field=0)
        npt.assert_equal(repr(m).startswith("Medium("), True)
