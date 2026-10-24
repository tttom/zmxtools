import json
import os
from pathlib import Path

import numpy as np
import pytest
from numpy import testing as npt

from tests.zmx import test_directory
from zmxtools import gui as oshow

IN_GITHUB_ACTION = os.getenv("GITHUB_ACTIONS") == "true"

ZMX_FILE = test_directory / "LA1116-Zemax.zmx"
ZAR_FILE = test_directory / "352240-1064-Zemax.zar"
INF_ZAR_FILE = test_directory / "TRH254-040-A-ML-Zemax.zar"

SYNTHETIC = """VERS 24.1
NAME Two lens system
AUTH Tester
NOTE 0 A synthetic two-lens system
MODE SEQ
UNIT MM
WAVL 0.5875618
WWGT 1.0
SURF 0
  TYPE STANDARD
  CURV 0.0
  DISZ 0.0
  GLAS __BLANK
SURF 1
  TYPE STANDARD
  CURV 0.02
  DISZ 3.0
  GLAS 5168 6417
SURF 2
  TYPE STANDARD
  CURV 0.0
  DISZ 2.0
  GLAS 6200 3600
SURF 3
  TYPE STANDARD
  CURV -0.01
  DISZ 0.0
  STOP
SURF 4
  TYPE STANDARD
  CURV 0.0
  DISZ 0.0
"""


WATER_GAP = """VERS 24.1
NAME Water gap system
AUTH Tester
NOTE 0 Glass, water, glass
MODE SEQ
UNIT MM
WAVL 0.5875618
WWGT 1.0
SURF 0
  TYPE STANDARD
  CURV 0.0
  DISZ 0.0
  GLAS __BLANK
SURF 1
  TYPE STANDARD
  CURV 0.02
  DISZ 2.0
  GLAS 5168 6417
SURF 2
  TYPE STANDARD
  CURV 0.0
  DISZ 2.0
  GLAS WATER 1.33
SURF 3
  TYPE STANDARD
  CURV -0.02
  DISZ 0.0
  GLAS 1700 5000
SURF 4
  TYPE STANDARD
  CURV 0.0
  DISZ 0.0
"""


BICONVEX = """VERS 24.1
NAME Strong biconvex
AUTH Tester
NOTE 0 R=2mm biconvex, 2mm thick, 10mm aperture
MODE SEQ
UNIT MM
WAVL 0.5875618
WWGT 1.0
SURF 0
  TYPE STANDARD
  CURV 0.0
  DISZ 0.0
  GLAS __BLANK
SURF 1
  TYPE STANDARD
  CURV 0.5
  DISZ 2.0
  DIAM 10.0
  STOP
  GLAS 5168 6417
SURF 2
  TYPE STANDARD
  CURV -0.5
  DISZ 0.0
  DIAM 10.0
SURF 3
  TYPE STANDARD
  CURV 0.0
  DISZ 0.0
"""


INFINITY = """VERS 24.1
NAME Infinity system
AUTH Tester
NOTE 0 Object and image at infinity
MODE SEQ
UNIT MM
WAVL 0.5875618
WWGT 1.0
SURF 0
  TYPE STANDARD
  CURV 0.0
  DISZ INFINITY
  GLAS __BLANK
SURF 1
  TYPE STANDARD
  CURV 0.05
  DISZ 3.0
  DIAM 10.0
  STOP
  GLAS 5168 6417
SURF 2
  TYPE STANDARD
  CURV -0.05
  DISZ INFINITY
  DIAM 10.0
SURF 3
  TYPE STANDARD
  CURV 0.0
  DISZ 0.0
"""


def synthetic_design() -> oshow.ZmxOpticalDesign:
    return oshow.ZmxOpticalDesign.from_str(SYNTHETIC)


class TestGlassFilter:
    def test_vacuum_is_not_glass(self):
        from zmxtools.optical_design.material import VACUUM

        npt.assert_equal(oshow._is_glass(VACUUM), False)

    def test_air_and_water_are_not_glass(self):
        from zmxtools.optical_design.material import ModelGlassMaterial, SimpleAir

        npt.assert_equal(oshow._is_glass(SimpleAir()), False)
        npt.assert_equal(oshow._is_glass(ModelGlassMaterial("water", 1.33)), False)
        npt.assert_equal(oshow._is_glass(ModelGlassMaterial("air", 1.0)), False)

    def test_glass_is_glass(self):
        from zmxtools.optical_design.material import ModelGlassMaterial

        npt.assert_equal(oshow._is_glass(ModelGlassMaterial("N-BK7", 1.5168, 64.17)), True)

    def test_water_section_not_drawn(self):
        scene = oshow.build_scene(oshow.ZmxOpticalDesign.from_str(WATER_GAP), nb_rays=1)
        npt.assert_equal(len(scene["lenses"]), 2)  # The water section between surfaces 2 and 3 is skipped.

    def test_lens_caps_clipped(self):
        import math

        scene = oshow.build_scene(oshow.ZmxOpticalDesign.from_str(BICONVEX), nb_rays=1)
        npt.assert_equal(len(scene["lenses"]), 1)
        r_max = max(math.hypot(v[0], v[1]) for mesh in scene["lenses"][0]["meshes"] for v in mesh["vertices"])
        npt.assert_equal(r_max < 3e-3, True)  # Clipped well below the 5 mm aperture, where the caps meet.
        npt.assert_equal(r_max > 1e-3, True)

    def test_object_at_infinity(self):
        scene = oshow.build_scene(oshow.load_design(INF_ZAR_FILE), nb_rays=7)
        points = np.array([p for ray in scene["rays"] for p in ray["points"]])
        npt.assert_equal(np.all(np.isfinite(points)), True)
        npt.assert_equal(np.isfinite(scene["bounds"]["zmin"]), True)


class TestLoad:
    def test_load_zmx(self):
        design = oshow.load_design(ZMX_FILE)
        npt.assert_equal(len(design.surfaces) >= 3, True)

    def test_load_zar(self):
        design = oshow.load_design(ZAR_FILE)
        npt.assert_equal(len(design.surfaces) >= 3, True)

    def test_load_str_path(self):
        design = oshow.load_design(str(ZMX_FILE))
        npt.assert_equal(len(design.surfaces) >= 3, True)

    def test_load_bad_extension(self):
        with pytest.raises(ValueError):
            oshow.load_design("some/file.txt")


class TestUnzarOutput:
    def test_load_zip(self):
        import tempfile

        from zmxtools import zar

        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "design.zip"
            zar.repack(ZAR_FILE, zip_path)
            design = oshow.load_design(zip_path)
            npt.assert_equal(len(design.surfaces) >= 3, True)

    def test_load_directory(self):
        import tempfile

        from zmxtools import zar

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "design"
            zar.extract(ZAR_FILE, output)
            design = oshow.load_design(output)
            npt.assert_equal(len(design.surfaces) >= 3, True)

    def test_load_directory_without_design(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ValueError):
                oshow.load_design(tmp)


class TestInfinityRays:
    def setup_method(self):
        self.scene = oshow.build_scene(oshow.ZmxOpticalDesign.from_str(INFINITY), nb_rays=7)
        self.diameter = 2.0 * self.scene["aperture_stop"]["inner_radius"]

    def test_input_rays_parallel(self):
        for ray in self.scene["rays"]:
            points = np.array(ray["points"])
            direction = points[1] - points[0]
            npt.assert_allclose(direction[:2], [0.0, 0.0], atol=1e-12)  # Parallel to the axis.
            npt.assert_equal(direction[2] >= 5.0 * self.diameter - 1e-12, True)

    def test_output_rays_extended(self):
        for ray in self.scene["rays"]:
            points = np.array(ray["points"])
            length = float(np.linalg.norm(points[-1] - points[-2]))
            npt.assert_equal(length >= 5.0 * self.diameter - 1e-12, True)

    def test_bounds_include_extensions(self):
        npt.assert_equal(self.scene["bounds"]["zmin"] <= -5.0 * self.diameter + 1e-12, True)
        npt.assert_equal(self.scene["bounds"]["zmax"] >= 0.003 + 5.0 * self.diameter - 1e-12, True)

    def test_object_at_infinity_fixture(self):
        scene = oshow.build_scene(oshow.load_design(INF_ZAR_FILE), nb_rays=7)
        for ray in scene["rays"]:
            points = np.array(ray["points"])
            npt.assert_allclose(points[1][:2], points[0][:2], atol=1e-12)  # Parallel input.


NEW_REAL_WORLD_DESIGNS = (
    "circular_aperture_example_with_aperture",  # 100 deg wide-angle lens, inches.
    "complicated_fold_mirrors_setup_v2",  # Coordinate breaks and fold mirrors.
    "curved_reflective_grating",
    "flat_transmission_grating",
    "inactive_wavelengths",
    "lens1",
    "lens2",  # Aspheric F/1.5.
    "lens_floa",
    "lens_offset_aperture",
    "one_mirror_up_45deg",
    "simple_fold_mirror_up",
    "thorlabs_lj1598l1",  # Cylindrical lens.
)


class TestRealWorldDesigns:
    def test_build_scene(self, subtests):
        for stem in NEW_REAL_WORLD_DESIGNS:
            with subtests.test(stem=stem):
                scene = oshow.build_scene(oshow.load_design(test_directory / f"{stem}.zmx"), nb_rays=7)
                points = np.array([p for ray in scene["rays"] for p in ray["points"]])
                npt.assert_equal(np.all(np.isfinite(points)), True)
                npt.assert_equal(len(scene["rays"]) >= 7, True)


class TestScene:
    def setup_method(self):
        self.scene = oshow.build_scene(synthetic_design(), nb_rays=7)

    def test_structure(self):
        npt.assert_equal(self.scene["version"], 2)
        npt.assert_equal(self.scene["name"], "Two lens system")
        npt.assert_equal(len(self.scene["lenses"]) >= 1, True)
        npt.assert_equal(len(self.scene["rays"]) >= 7, True)
        json.loads(oshow.scene_json(self.scene))

    def test_axis_line(self):
        line = self.scene["axis_line"]
        npt.assert_equal(len(line), 2)
        npt.assert_allclose(line[0][2], self.scene["bounds"]["zmin"])
        npt.assert_allclose(line[1][2], self.scene["bounds"]["zmax"])

    def test_rays_start_at_source(self):
        z_min = self.scene["bounds"]["zmin"]
        for ray in self.scene["rays"]:
            npt.assert_allclose(ray["points"][0][2], z_min)

    def test_ray_colors_by_start_point(self):
        starts: dict[tuple, str] = {}
        for ray in self.scene["rays"]:
            key = tuple(round(v, 9) for v in ray["points"][0])
            if key in starts:
                npt.assert_equal(starts[key], ray["color"])
            starts[key] = ray["color"]

    def test_aperture_stop(self):
        stop = self.scene["aperture_stop"]
        npt.assert_equal(stop["inner_radius"] > 0, True)
        npt.assert_equal(stop["outer_radius"] > stop["inner_radius"], True)
        npt.assert_allclose(stop["center"][2], 0.0)

    def test_stop_centered(self):
        bounds = self.scene["bounds"]
        npt.assert_equal(bounds["zmin"] <= 0.0, True)
        npt.assert_equal(bounds["zmax"] >= 0.0, True)

    def test_lens_colors_distinct(self):
        colors = [lens["color"] for lens in self.scene["lenses"]]
        npt.assert_equal(len(set(colors)), len(colors))
        npt.assert_equal(all(lens["opacity"] == 0.5 for lens in self.scene["lenses"]), True)
        npt.assert_equal(all(len(lens["rim_lines"]) > 0 for lens in self.scene["lenses"]), True)

    def test_ray_points(self):
        for ray in self.scene["rays"]:
            points = np.asarray(ray["points"])
            npt.assert_equal(points.shape[0] >= 2, True)
            npt.assert_equal(np.all(np.isfinite(points)), True)

    def test_axes_length(self):
        npt.assert_equal(self.scene["axes"]["length"] > 0, True)

    def test_image_plane_center(self):
        npt.assert_allclose(self.scene["image_plane"]["center"][2], self.scene["bounds"]["zmax"])


class TestViewerDescription:
    def setup_method(self):
        self.scene = oshow.build_scene(synthetic_design(), nb_rays=1)

    def test_css_design_constants(self):
        npt.assert_equal([lens["color"] for lens in self.scene["lenses"]], ["#008000", "#008080"])
        npt.assert_equal(all(lens["opacity"] == 0.5 for lens in self.scene["lenses"]), True)
        npt.assert_equal(self.scene["aperture_stop"]["opacity"], 0.5)

    def test_lenses_have_two_ends_and_rims(self):
        for lens in self.scene["lenses"]:
            npt.assert_equal(len(lens["surfaces"]), 2)
            npt.assert_equal(len(lens["rims"]), 2)
            surface_ids = [mesh.get("surface_id") for mesh in lens["meshes"] if mesh.get("surface_id")]
            expected = [f"surf-{lens['surfaces'][0]}", f"surf-{lens['surfaces'][1]}", f"rim-{lens['id'].split('-')[-1]}"]
            npt.assert_equal(sorted(surface_ids), sorted(expected))

    def test_table_block_order(self):
        types = [block["type"] for block in self.scene["table"]]
        npt.assert_equal(
            types,
            ["medium", "surface", "material", "surface", "material", "surface", "medium", "image"],
        )
        npt.assert_equal(self.scene["table"][0]["target"], {"kind": "rays"})
        npt.assert_equal(self.scene["table"][-1]["target"], {"kind": "image"})

    def test_surface_targets_have_caps(self):
        surface_ids = {
            mesh["surface_id"] for lens in self.scene["lenses"] for mesh in lens["meshes"] if mesh.get("surface_id")
        }
        for block in self.scene["table"]:
            target = block.get("target", {})
            if target.get("kind") == "surface":
                npt.assert_equal(target["id"] in surface_ids, True)
                npt.assert_equal(len(block["details"]) >= 2, True)

    def test_material_colors_match_rims(self):
        lens_colors = {lens["id"]: lens["color"] for lens in self.scene["lenses"]}
        for block in self.scene["table"]:
            target = block.get("target", {})
            if block["type"] == "material" and target.get("kind") == "rim":
                npt.assert_equal(block["color"], lens_colors[target["id"]])

    def test_stop_surface_is_marked(self):
        stops = [block for block in self.scene["table"] if block.get("stop")]
        npt.assert_equal(len(stops), 1)
        npt.assert_equal(stops[0]["stop"], True)


class TestRender:
    def test_html_contains_scene(self):
        scene = oshow.build_scene(synthetic_design(), nb_rays=1)
        html = oshow.render_html(scene)
        npt.assert_equal("<!DOCTYPE html>" in html, True)
        npt.assert_equal("Two lens system" in html, True)
        npt.assert_equal('"version": 2' in html, True)
        npt.assert_equal("const SERVED = false;" in html, True)
        npt.assert_equal("&#9680;" in html, True)  # The light/dark mode toggle symbol.
        npt.assert_equal("prefers-color-scheme" in html, True)  # OS-default theme.
        npt.assert_equal("0x808088" in html, True)  # Dark silver aperture stop.

    def test_html_served_flag(self):
        scene = oshow.build_scene(synthetic_design(), nb_rays=1)
        html = oshow.render_html(scene, served=True)
        npt.assert_equal("const SERVED = true;" in html, True)


class TestCli:
    def test_version(
        self,
    ):
        assert oshow.oshow(["-V"]) == 0

    def test_write_outputs(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            html_path = Path(tmp) / "view.html"
            json_output_path = Path(tmp) / "scene.json"
            code = oshow.oshow(
                ["-i", str(ZMX_FILE), "--no-serve", "-o", str(html_path), "-j", str(json_output_path), "--no-open"],
            )
            npt.assert_equal(code, 0)
            npt.assert_equal(html_path.exists(), True)
            npt.assert_equal(json_output_path.exists(), True)
            json.loads(json_output_path.read_text(encoding="utf-8"))

    def test_missing_input(self):
        code = oshow.oshow(["-i", "definitely/missing.zmx", "--no-serve"])
        npt.assert_equal(code, 1)

    def test_positional_input(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            html_path = Path(tmp) / "view.html"
            code = oshow.oshow([str(ZMX_FILE), "--no-serve", "-o", str(html_path)])
            npt.assert_equal(code, 0)
            npt.assert_equal(html_path.exists(), True)

    def test_positional_and_flag_conflict(self):
        code = oshow.oshow(["-i", str(ZMX_FILE), str(ZAR_FILE), "--no-serve"])
        npt.assert_equal(code, 2)

    def test_no_input(self):
        code = oshow.oshow(["--no-serve"])
        npt.assert_equal(code, 2)


class TestServe:
    def test_serve_endpoints(self):
        import threading
        import urllib.request

        scene = oshow.build_scene(synthetic_design(), nb_rays=1)
        oshow._ViewerHandler.state = {"html": "", "scene": "{}", "version": "0"}
        thread = threading.Thread(target=oshow.serve, args=(scene,), kwargs={"port": 8199, "open_browser": False})
        thread.start()
        try:
            version = urllib.request.urlopen("http://127.0.0.1:8199/version", timeout=5).read().decode()
            npt.assert_equal(version, "0")
            page = urllib.request.urlopen("http://127.0.0.1:8199/", timeout=5).read().decode()
            npt.assert_equal("<!DOCTYPE html>" in page, True)
            body = urllib.request.urlopen("http://127.0.0.1:8199/scene.json", timeout=5).read().decode()
            json.loads(body)
            missing = urllib.request.urlopen("http://127.0.0.1:8199/nope", timeout=5).read().decode()
            npt.assert_equal(missing, "Not found")
        finally:
            oshow._ViewerHandler.state["stop"] = True
            thread.join(timeout=5)
        npt.assert_equal(thread.is_alive(), False)

    @pytest.mark.skipif(IN_GITHUB_ACTION, reason="Accessing URLs does not work in GitHub workflow.")
    def test_serve_exit_after_open(self):
        import threading
        import time
        import urllib.request

        scene = oshow.build_scene(synthetic_design(), nb_rays=1)
        thread = threading.Thread(
            target=oshow.serve,
            args=(scene,),
            kwargs={"port": 8200, "open_browser": False, "exit_after_open": True},
        )
        thread.start()
        page = b""
        for _ in range(50):
            try:
                page = urllib.request.urlopen("http://127.0.0.1:8200/", timeout=2).read()
                break
            except OSError:
                time.sleep(0.1)
        npt.assert_equal(b"<!DOCTYPE html>" in page, True)
        thread.join(timeout=5)
        npt.assert_equal(thread.is_alive(), False)
