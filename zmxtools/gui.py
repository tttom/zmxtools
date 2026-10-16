"""
A sub-module to view optical designs in an interactive 3D browser window.

Loads an optical design from a `.zar` archive, a `.zmx` file, or the output of
`unzar` (a `.zip` archive or a directory with the `.zmx` and `.agf` files), builds
a JSON scene (lenses, aperture-stop-centered axes, traced ray bundle, image plane),
and serves it as a self-contained Three.js page with live reload over a local
`http.server`.

The scene JSON is also the model save format: writing it to disk stores the exact
scene that the viewer renders. See the oshow section of the documentation.
"""

from __future__ import annotations

import argparse
import importlib.resources
import itertools
import json
import math
import threading
import time
import webbrowser
import zipfile
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import numpy as np

from zmxtools import log, zar
from zmxtools.agf import AgfMaterialLibrary
from zmxtools.optical_design import material as material_module
from zmxtools.utils.array import array_type, asarray
from zmxtools.utils.io import BytesFile
from zmxtools.zmx import ZmxOpticalDesign, ZmxSurface

log = log.getChild(__name__)

__all__ = ["build_scene", "load_design", "oshow", "render_html", "scene_json", "serve"]

PALETTE = ("#4fc3f7", "#ff8a65", "#aed581", "#ba68c8", "#4db6ac", "#f06292", "#fff176", "#90a4ae")
RAY_PALETTE = ("#ffeb3b", "#ff5252", "#40c4ff", "#69f0ae", "#e040fb", "#ffd740")

D_LINE_WAVENUMBER = 2 * np.pi / 587.5618e-9

BACKGROUND_MATERIAL_NAMES = {"vacuum", "air", "water"}


def _design_from_named_files(
    named_files: Sequence[tuple[str, bytes]],
    material_libraries: Sequence[str | Path],
    origin: Path,
) -> ZmxOpticalDesign:
    """
    Load the first design from a collection of (name, contents) pairs.

    Any `.agf` files among them are used as glass catalogs, on top of the
    optionally provided material library paths.

    Args:
        named_files: The (file name, contents) pairs to search for a design.
        material_libraries: Optional paths to extra `.agf` glass catalogs.
        origin: The archive or directory the files came from, for error messages.

    Returns:
        The first optical design found.

    Raises:
        ValueError: If no `.zmx` file is present.
    """
    libraries = list[AgfMaterialLibrary]([AgfMaterialLibrary.from_file(str(_)) for _ in material_libraries])
    libraries += [
        AgfMaterialLibrary.from_file(BytesFile(name, contents))
        for name, contents in named_files
        if name.lower().endswith(".agf")
    ]
    zmx_files = [(name, contents) for name, contents in named_files if name.lower().endswith(".zmx")]
    if len(zmx_files) == 0:
        raise ValueError(f"No .zmx optical design file found in {origin}.")
    name, contents = zmx_files[0]
    return ZmxOpticalDesign.from_file(BytesFile(name, contents), libraries)


def load_design(input_path: str | Path, material_libraries: Sequence[str | Path] = ()) -> ZmxOpticalDesign:
    """
    Load an optical design from a `.zar` archive, a `.zmx` file, or `unzar` output.

    The `unzar` output can be a `.zip` archive (from `unzar -z`) or a directory
    (from `unzar`) containing the `.zmx` design file and any `.agf` glass catalogs.

    Args:
        input_path: Path to the `.zar`, `.zmx`, or `.zip` file, or to a directory.
        material_libraries: Optional paths to `.agf` glass catalogs.

    Returns:
        The first optical design found in the input.

    Raises:
        ValueError: If the input is not a `.zar`, `.zmx`, or `.zip` file or a
            directory containing a `.zmx` file, or if the archive is empty.
    """
    input_path = Path(input_path)
    suffix = input_path.suffix.lower()
    if input_path.is_dir():
        named_files = [(p.name, p.read_bytes()) for p in sorted(input_path.rglob("*")) if p.is_file()]
        return _design_from_named_files(named_files, material_libraries, input_path)
    if suffix == ".zar":
        designs = zar.load(input_path)
        if len(designs) == 0:
            raise ValueError(f"No optical design found in {input_path}.")
        return designs[0]
    if suffix == ".zmx":
        return ZmxOpticalDesign.from_file(input_path, list(material_libraries))
    if suffix == ".zip":
        with zipfile.ZipFile(input_path) as archive:
            named_files = [
                (Path(info.filename).name, archive.read(info)) for info in archive.infolist() if not info.is_dir()
            ]
        return _design_from_named_files(named_files, material_libraries, input_path)
    raise ValueError(f"Unsupported input {input_path}, expected a .zar, .zmx, or .zip file or a directory.")


def _surface_zs(surfaces: Sequence[ZmxSurface]) -> list[float]:
    """Cumulative axial positions of the surfaces, in meters. ``DISZ`` of surface *i* is the distance to surface *i+1*."""
    zs = [0.0]
    for surface in surfaces[:-1]:
        spacing = float(surface.spacing)
        if not np.isfinite(spacing):
            spacing = 0.0  # Object/image at infinity: rays are parallel.
        zs.append(zs[-1] + spacing)
    return zs


def _max_radius(surfaces: Sequence[ZmxSurface]) -> float:
    """A sensible drawing radius: the largest finite aperture radius, with a fallback."""
    finite = [float(_.radius) for _ in surfaces if np.isfinite(_.radius) and _.radius > 0]
    if len(finite) == 0:
        return 5e-3
    return max(finite)


def _sag_function(surface: ZmxSurface):
    """The sag function of a surface; unimplemented surface types are drawn as planes."""
    sag = getattr(surface, "sag", None)
    if sag is None:
        return lambda position: np.zeros_like(np.asarray(position, dtype=float)[..., 0])
    return sag


def _sag_at(surface: ZmxSurface, x: float, y: float) -> float:
    """The sag of a surface at a transverse position (x, y)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(asarray(_sag_function(surface)(np.array([x, y, 0.0]))))


def _sample_sag(surface: ZmxSurface, radius: float, nb_r: int = 12, nb_theta: int = 48) -> array_type:
    """
    Sample the surface sag on a polar grid.

    Args:
        surface: The surface to sample.
        radius: The sampling radius, in meters.
        nb_r: The number of radial samples.
        nb_theta: The number of azimuthal samples.

    Returns:
        An (nb_r, nb_theta, 3) array of points on the surface.
    """
    radii = np.linspace(0.0, radius, nb_r)
    thetas = np.linspace(0.0, 2.0 * np.pi, nb_theta, endpoint=False)
    r_grid, t_grid = np.meshgrid(radii, thetas, indexing="ij")
    x = r_grid * np.cos(t_grid)
    y = r_grid * np.sin(t_grid)
    z = np.zeros_like(x)
    with np.errstate(invalid="ignore", divide="ignore"):
        sag = asarray(_sag_function(surface)(np.stack((x, y, z), axis=-1)), float)
    return np.stack((x, y, sag), axis=-1)


def _cap_mesh(points: array_type, z_offset: float) -> tuple[list[list[float]], list[list[int]]]:
    """Triangulated mesh of a sampled cap, lifted to z_offset."""
    nb_r, nb_theta, _ = points.shape
    vertices = [
        [float(points[i, j, 0]), float(points[i, j, 1]), float(points[i, j, 2]) + z_offset]
        for i in range(nb_r)
        for j in range(nb_theta)
    ]
    triangles = list[list[int]]()
    for i in range(nb_r - 1):
        for j in range(nb_theta):
            jn = (j + 1) % nb_theta
            a = i * nb_theta + j
            b = i * nb_theta + jn
            c = (i + 1) * nb_theta + jn
            d = (i + 1) * nb_theta + j
            triangles.append([a, b, c])
            triangles.append([a, c, d])
    return vertices, triangles


def _rim(points: array_type, z_offset: float) -> list[list[float]]:
    """The closed rim polyline of a sampled cap, lifted to z_offset."""
    outer = points[-1]
    rim = [[float(p[0]), float(p[1]), float(p[2]) + z_offset] for p in outer]
    rim.append(list(rim[0]))
    return rim


def _side_mesh(rim_a: list[list[float]], rim_b: list[list[float]]) -> tuple[list[list[float]], list[list[int]]]:
    """The cylinder side connecting two rim polylines with matching angular sampling."""
    nb = len(rim_a) - 1
    vertices = [*rim_a[:-1], *rim_b[:-1]]
    triangles = list[list[int]]()
    for j in range(nb):
        jn = (j + 1) % nb
        triangles.append([j, jn, nb + jn])
        triangles.append([j, nb + jn, nb + j])
    return vertices, triangles


def _is_glass(mat: material_module.Material) -> bool:
    """Whether a material is a lens glass, as opposed to a background medium."""
    if mat is material_module.VACUUM:
        return False
    if str(mat.name).lower() in BACKGROUND_MATERIAL_NAMES:
        return False
    try:
        refractive_index = float(np.real(mat.complex_refractive_index(wavenumber=D_LINE_WAVENUMBER)))
    except Exception:
        return True  # Unknown optical properties: draw it.
    return refractive_index > 1.001  # Excludes vacuum and air.


def _group_lenses(surfaces: Sequence[ZmxSurface]) -> list[list[int]]:
    """Group surface indices into lens elements: consecutive surfaces sharing a glass."""
    lenses = list[list[int]]()
    current = list[int]()
    for idx in range(1, len(surfaces) - 1):
        mat = surfaces[idx].material
        is_glass = _is_glass(mat) and not getattr(surfaces[idx], "is_mirror", False)
        if is_glass and len(current) > 0 and surfaces[current[-1]].material.name == mat.name:
            current.append(idx)
            continue
        if is_glass:
            if len(current) > 0:
                lenses.append(current)
            current = [idx]
        elif len(current) > 0:
            current.append(idx)
            lenses.append(current)
            current = []
    if len(current) > 0:
        lenses.append(current)
    return lenses


def _sag_profile(surface: ZmxSurface, radii: array_type) -> array_type:
    """The sag of a surface along a radial profile (x = r, y = 0)."""
    positions = np.stack((radii, np.zeros_like(radii), np.zeros_like(radii)), axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return asarray(_sag_function(surface)(positions), float)


def _lens_radius(surfaces: Sequence[ZmxSurface], zs: Sequence[float], indices: Sequence[int], r_max: float) -> float:
    """
    The largest drawing radius at which the lens caps do not intersect.

    The glass thickness between each pair of consecutive lens surfaces,
    ``t(r) = z_j + sag_j(r) - z_i - sag_i(r)``, must stay positive: surfaces are
    clipped at the first radius where they meet, so caps never cross and no
    surface is drawn beyond its intersection with the neighboring one.

    Args:
        surfaces: All design surfaces.
        zs: Axial surface positions.
        indices: The surface indices making up this lens.
        r_max: The requested maximum drawing radius.

    Returns:
        The clipped drawing radius, 0 for a degenerate lens.
    """
    if len(indices) < 2:
        return r_max
    radii = np.linspace(0.0, r_max, 256)
    r_clip = r_max
    for i, j in itertools.pairwise(indices):
        thickness = (zs[j] - zs[i]) + _sag_profile(surfaces[j], radii) - _sag_profile(surfaces[i], radii)
        crossings = np.flatnonzero(thickness <= 0)
        if len(crossings) > 0:
            r_clip = min(r_clip, float(radii[crossings[0]]))
    return r_clip


def _refract(direction: array_type, normal: array_type, eta: float) -> array_type | None:
    """Refract a unit direction across a surface normal; None on total internal reflection."""
    d = direction / np.linalg.norm(direction)
    n = normal / np.linalg.norm(normal)
    cos_i = -float(np.dot(n, d))
    sin2_t = eta * eta * (1.0 - cos_i * cos_i)
    if sin2_t > 1.0:
        return None
    cos_t = math.sqrt(1.0 - sin2_t)
    return eta * d + (eta * cos_i - cos_t) * n


def _refractive_index(mat: material_module.Material) -> float:
    """The d-line refractive index, clamped to a physical range; out-of-range model glasses become air."""
    try:
        refractive_index = float(np.real(mat.complex_refractive_index(wavenumber=D_LINE_WAVENUMBER)))
    except Exception:
        return 1.0
    if not np.isfinite(refractive_index) or refractive_index < 0.5 or refractive_index > 4.0:
        return 1.0  # E.g. a glass whose catalog was not provided.
    return refractive_index


def _append_point(points: list[list[float]], position: array_type) -> None:
    """Append a point to a polyline, skipping duplicates of the previous point."""
    point = [float(position[0]), float(position[1]), float(position[2])]
    if len(points) == 0 or math.dist(points[-1], point) > 1e-9:
        points.append(point)


def _trace_ray(
    position: array_type,
    direction: array_type,
    surfaces: Sequence[ZmxSurface],
    zs: Sequence[float],
    radius: float,
) -> list[list[float]]:
    """Trace one paraxial ray through the surfaces, returning the polyline points."""
    points = [[float(position[0]), float(position[1]), float(position[2])]]
    p = asarray(position, float).copy()
    d = asarray(direction, float)
    d = d / np.linalg.norm(d)
    eps = 1e-6
    with np.errstate(invalid="ignore", divide="ignore"):
        for idx in range(1, len(surfaces)):
            surface = surfaces[idx]
            z_surface = float(zs[idx])
            for _ in range(4):  # Intersect the sag surface by fixed-point iteration.
                if abs(d[2]) < 1e-12:
                    break
                t = (z_surface + _sag_at(surface, *(p + (z_surface - p[2]) / d[2] * d)[:2]) - p[2]) / d[2]
                p = p + t * d
            if not np.all(np.isfinite(p)):
                return points  # The ray left the valid sag domain.
            if math.hypot(p[0], p[1]) > radius:
                _append_point(points, p)
                return points  # Vigneted by the aperture.
            fx = (_sag_at(surface, p[0] + eps, p[1]) - _sag_at(surface, p[0] - eps, p[1])) / (2 * eps)
            fy = (_sag_at(surface, p[0], p[1] + eps) - _sag_at(surface, p[0], p[1] - eps)) / (2 * eps)
            normal = asarray([fx, fy, -1.0], float)
            if not np.all(np.isfinite(normal)):
                normal = asarray([0.0, 0.0, -1.0], float)  # Outside the sag domain: treat as planar.
            if getattr(surfaces[idx - 1], "is_mirror", False):
                n = normal / np.linalg.norm(normal)
                d = d - 2 * float(np.dot(d, n)) * n
            else:
                n_before = _refractive_index(surfaces[idx - 1].material)
                n_after = _refractive_index(surface.material)
                new_d = _refract(d, normal, n_before / n_after)
                if new_d is None:
                    break
                d = new_d / np.linalg.norm(new_d)
                if not np.all(np.isfinite(d)):
                    return points  # Refraction left the valid domain.
            _append_point(points, p)
    return points


def _infinity_flags(surfaces: Sequence[ZmxSurface]) -> tuple[bool, bool]:
    """Whether the distance to the source of the image (the object) and to the image are infinite."""
    object_at_infinity = len(surfaces) >= 2 and not np.isfinite(float(surfaces[0].spacing))
    image_at_infinity = len(surfaces) >= 2 and not np.isfinite(float(surfaces[-2].spacing))
    return object_at_infinity, image_at_infinity


def _trace_bundle(
    surfaces: Sequence[ZmxSurface],
    zs: Sequence[float],
    z_stop: float,
    nb_rays: int,
) -> list[dict]:
    """Rays starting at the source, before the first surface, aimed through the entrance pupil."""
    stop_idx = next((i for i, s in enumerate(surfaces) if s.stop), len(surfaces) // 2)
    stop_radius = float(surfaces[stop_idx].radius) if np.isfinite(surfaces[stop_idx].radius) else _max_radius(surfaces)
    entrance_radius = min(stop_radius, _max_radius(surfaces))
    rays = list[dict]()
    if _infinity_flags(surfaces)[0]:
        # Source at infinity: parallel rays filling the entrance pupil, colored per ring.
        groups = [[(0.0, 0.0)]]
        ring = max(1, (nb_rays - 1) // 6)
        for ring_idx in range(1, ring + 1):
            r = entrance_radius * ring_idx / ring
            nb = 6 * ring_idx
            groups.append([(r * math.cos(2 * math.pi * k / nb), r * math.sin(2 * math.pi * k / nb)) for k in range(nb)])
        for color_idx, group in enumerate(groups):
            color = RAY_PALETTE[color_idx % len(RAY_PALETTE)]
            for sx, sy in group:
                polyline = _trace_ray(
                    np.array([sx, sy, float(zs[0])]),
                    np.array([0.0, 0.0, 1.0]),
                    surfaces,
                    zs,
                    _max_radius(surfaces) * 1.5,
                )
                rays.append(
                    {
                        "color": color,
                        "points": [[p[0], p[1], p[2] - z_stop] for p in polyline],
                    },
                )
        return rays
    source_radius = 0.25 * stop_radius
    source_points = [(0.0, 0.0)]
    source_points += [
        (source_radius * math.cos(k * math.pi / 2), source_radius * math.sin(k * math.pi / 2)) for k in range(4)
    ]
    per_source = max(3, nb_rays // len(source_points))
    targets = [(0.0, 0.0)]
    ring = max(1, (per_source - 1) // 6)
    for ring_idx in range(1, ring + 1):
        r = entrance_radius * ring_idx / ring
        nb = 6 * ring_idx
        targets += [(r * math.cos(2 * math.pi * k / nb), r * math.sin(2 * math.pi * k / nb)) for k in range(nb)]
    rays = list[dict]()
    for color_idx, (sx, sy) in enumerate(source_points):
        color = RAY_PALETTE[color_idx % len(RAY_PALETTE)]
        for tx, ty in targets[:per_source]:  # Rays from the same source point share the color.
            polyline = _trace_ray(
                np.array([sx, sy, float(zs[0])]),
                np.array([tx - sx, ty - sy, 1.0]),
                surfaces,
                zs,
                _max_radius(surfaces) * 1.5,
            )
            rays.append(
                {
                    "color": color,
                    "points": [[p[0], p[1], p[2] - z_stop] for p in polyline],
                },
            )
    return rays


def _extend_parallel_rays(
    rays: list[dict],
    object_at_infinity: bool,
    image_at_infinity: bool,
    aperture_diameter: float,
) -> None:
    """
    Draw the collimated parts of the ray bundle sticking out beyond the surfaces.

    When the source of the image or the image is at infinity, the rays enter the
    first or leave the last surface as parallel rays. They are extended beyond
    that surface by the maximum of the rest of the path length and five times the
    aperture diameter.

    Args:
        rays: The traced rays, modified in place.
        object_at_infinity: Whether the source of the image is at infinity.
        image_at_infinity: Whether the image is at infinity.
        aperture_diameter: The aperture stop diameter, in meters.
    """
    for ray in rays:
        points = ray["points"]
        if len(points) < 2:
            continue
        path_length = sum(math.dist(points[i], points[i + 1]) for i in range(len(points) - 1))
        reach = max(path_length, 5.0 * aperture_diameter)
        if image_at_infinity:
            direction = np.subtract(points[-1], points[-2], dtype=float)
            norm = float(np.linalg.norm(direction))
            if norm > 0.0:
                points.append([float(_) for _ in np.add(points[-1], direction * (reach / norm))])
        if object_at_infinity:
            points.insert(0, [points[0][0], points[0][1], points[0][2] - reach])  # Parallel to the axis.


def build_scene(design: ZmxOpticalDesign, nb_rays: int = 13) -> dict:
    """
    Build the viewer scene from an optical design.

    Args:
        design: The optical design to render.
        nb_rays: The approximate number of rays in the traced bundle.

    Returns:
        The scene dictionary, in the oshow scene format.
    """
    surfaces = list(design.surfaces)
    zs = _surface_zs(surfaces)
    stop_idx = next((i for i, s in enumerate(surfaces) if s.stop), len(surfaces) // 2)
    z_stop = zs[stop_idx]
    radius = _max_radius(surfaces)

    lenses = list[dict]()
    for element_idx, indices in enumerate(_group_lenses(surfaces)):
        lens_radius = _lens_radius(surfaces, zs, indices, radius)
        if lens_radius <= 0:
            log.debug(f"Skipping degenerate lens at surfaces {indices}.")
            continue
        color = PALETTE[element_idx % len(PALETTE)]
        meshes = list[dict]()
        rims = list[list[float]]()
        caps = list[tuple[list[list[float]], list[list[int]]]]()
        sampled = list[list[list[float]]]()
        for idx in indices:
            cap_points = _sample_sag(surfaces[idx], lens_radius)
            caps.append(_cap_mesh(cap_points, zs[idx] - z_stop))
            sampled.append(_rim(cap_points, zs[idx] - z_stop))
        for a, b in itertools.pairwise(sampled):
            side_vertices, side_triangles = _side_mesh(a, b)
            meshes.append({"vertices": side_vertices, "triangles": side_triangles})
        for vertices, triangles in caps:
            meshes.append({"vertices": vertices, "triangles": triangles})
        for rim in sampled:
            rims += rim
        lenses.append({"color": color, "opacity": 0.5, "meshes": meshes, "rim_lines": rims})

    object_at_infinity, image_at_infinity = _infinity_flags(surfaces)
    rays = _trace_bundle(surfaces, zs, z_stop, nb_rays)

    stop_radius = float(surfaces[stop_idx].radius) if np.isfinite(surfaces[stop_idx].radius) else radius
    image_radius = max(
        [math.hypot(ray["points"][-1][0], ray["points"][-1][1]) for ray in rays] + [0.25 * radius],
    )
    _extend_parallel_rays(rays, object_at_infinity, image_at_infinity, 2.0 * stop_radius)

    traced_zs = [p[2] for ray in rays for p in ray["points"]]
    z_min = min([zs[0] - z_stop, *traced_zs])
    z_max = max([zs[-1] - z_stop, *traced_zs])
    extent = max(z_max - z_min, 4 * radius)
    return {
        "version": 2,
        "name": design.name,
        "axes": {"length": 0.15 * extent},
        "axis_line": [[0.0, 0.0, z_min], [0.0, 0.0, z_max]],
        "lenses": lenses,
        "rays": rays,
        "aperture_stop": {
            "center": [0.0, 0.0, 0.0],
            "inner_radius": stop_radius,
            "outer_radius": 2.0 * stop_radius,
        },
        "image_plane": {"center": [0.0, 0.0, z_max], "radius": image_radius},
        "camera": {
            "position": [1.2 * extent, -1.6 * extent, 0.9 * extent],
            "target": [0.0, 0.0, 0.5 * (z_min + z_max)],
        },
        "bounds": {"radius": radius, "zmin": z_min, "zmax": z_max},
    }


def scene_json(scene: dict) -> str:
    """Serialize a scene to its JSON scene-format string."""
    return json.dumps(scene)


def render_html(scene: dict, served: bool = False) -> str:
    """
    Render the self-contained viewer page for a scene.

    Args:
        scene: The scene dictionary.
        served: When True, the page polls the server for live reloads.

    Returns:
        The HTML page string.
    """
    template = importlib.resources.files("zmxtools").joinpath("oshow_template.html").read_text(encoding="utf-8")
    return (
        template.replace("__TITLE__", str(scene.get("name", "optical design")))
        .replace("__SCENE__", json.dumps(scene))
        .replace("__SERVED__", "true" if served else "false")
    )


class _ViewerHandler(BaseHTTPRequestHandler):
    """Request handler serving the viewer page, the scene JSON, and a version stamp."""

    server_version = "oshow"
    state: ClassVar[dict] = {"html": "", "scene": "{}", "version": "0"}

    def do_GET(self) -> None:
        """Serve the page, scene, or version endpoints."""
        payload, content_type = {
            "/": (_ViewerHandler.state["html"], "text/html; charset=utf-8"),
            "/scene.json": (_ViewerHandler.state["scene"], "application/json"),
            "/version": (_ViewerHandler.state["version"], "text/plain"),
        }.get(self.path, ("Not found", "text/plain"))
        body = payload.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        if self.path == "/":
            _ViewerHandler.state["delivered"] = True  # The page embeds the full scene.

    def log_message(self, format: str, *args: object) -> None:
        """Route the request log through the package logger."""
        log.debug("%s - %s", self.address_string(), format % args)


def serve(
    scene: dict,
    port: int = 8010,
    open_browser: bool = True,
    watch: Path | None = None,
    exit_after_open: bool = False,
) -> None:
    """
    Serve the viewer page on a local http.server with live reload.

    The page polls `/version` every second and refetches `/scene.json` when it changes.
    When `watch` is given, the scene is rebuilt whenever that file's timestamp changes.

    Args:
        scene: The scene to serve.
        port: The TCP port to serve on.
        open_browser: Whether to open the default browser on the page.
        watch: An optional input file to watch for live rebuilds.
        exit_after_open: Stop serving as soon as the page has been delivered to the
            browser. The page is self-contained, so it keeps working; only live
            reload stops.
    """
    _ViewerHandler.state = {
        "html": render_html(scene, served=True),
        "scene": scene_json(scene),
        "version": "0",
        "delivered": False,
    }

    def rebuild_loop() -> None:
        """Rebuild the scene when the watched input file changes."""
        if watch is None:
            return
        stamp = watch.stat().st_mtime
        counter = 1
        while not _ViewerHandler.state.get("stop", False):
            time.sleep(1.0)
            try:
                new_stamp = watch.stat().st_mtime
            except OSError:
                continue
            if new_stamp != stamp:
                stamp = new_stamp
                log.info(f"Rebuilding scene after change in {watch}...")
                try:
                    new_scene = build_scene(load_design(watch))
                    _ViewerHandler.state = {
                        "html": render_html(new_scene, served=True),
                        "scene": scene_json(new_scene),
                        "version": str(counter),
                        "delivered": False,
                    }
                    counter += 1
                except Exception as err:
                    log.warning(f"Could not rebuild scene: {err}")

    threading.Thread(target=rebuild_loop, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", port), _ViewerHandler)
    server.timeout = 0.5
    url = f"http://127.0.0.1:{port}/"
    log.info(f"Serving oshow on {url} (Ctrl+C to stop)...")
    if open_browser:
        webbrowser.open(url)
    try:
        deadline = time.monotonic() + 30.0 if exit_after_open else None
        while not _ViewerHandler.state.get("stop", False):
            if exit_after_open and _ViewerHandler.state.get("delivered", False):
                break  # The page is fully delivered; the scene is embedded in it.
            if deadline is not None and time.monotonic() > deadline:
                break
            server.handle_request()
    except KeyboardInterrupt:
        log.info("Stopping oshow server.")
    finally:
        _ViewerHandler.state["stop"] = True
        server.server_close()


def _make_argument_parser() -> argparse.ArgumentParser:
    """Create the `oshow` command line argument parser."""
    parser = argparse.ArgumentParser(prog="oshow", description="View an optical design in 3D.")
    parser.add_argument(
        "-i", "--input", type=str, help="the .zar, .zmx, or .zip input file, or a directory (e.g. the output of unzar)"
    )
    parser.add_argument(
        "input_file", type=str, nargs="?", help="the input file or directory (alternative to -i, e.g. unzar output)"
    )
    parser.add_argument("-l", "--library", type=str, nargs="*", default=[], help="optional .agf glass catalog files")
    parser.add_argument("-o", "--output", type=str, help="write the HTML page to this file")
    parser.add_argument("-s", "--save-scene", type=str, help="write the scene JSON to this file")
    parser.add_argument("--rays", type=int, default=13, help="approximate number of rays to trace (default: 13)")
    parser.add_argument("--port", type=int, default=8010, help="port for the viewer server (default: 8010)")
    parser.add_argument("--no-serve", action="store_true", help="do not start the viewer server")
    parser.add_argument("--no-open", action="store_true", help="do not open a browser window")
    parser.add_argument(
        "--keep",
        action="store_true",
        help="keep serving after the page is delivered (by default the viewer exits once the model is displayed)",
    )
    parser.add_argument(
        "--watch", action="store_true", help="rebuild the scene when the input file changes (keeps the server running)"
    )
    parser.add_argument("-V", "--version", action="store_true", help="show version information")
    return parser


def oshow(argv: Sequence[str] | None = None) -> int:
    """
    Run the `oshow` command line tool.

    Args:
        argv: An optional sequence of input arguments.

    Returns:
        The error code. 0: no error, 1: input problem, 2: incorrect argument.
    """
    parser = _make_argument_parser()
    args = parser.parse_args(argv)
    if args.version:
        from zmxtools import __version__

        print(__version__)
        return 0
    if args.input is not None and args.input_file is not None:
        log.error("Specify the input file with -i or as a positional argument, not both.")
        parser.print_usage()
        return 2
    input_path = args.input if args.input is not None else args.input_file
    if input_path is None:
        log.error("No input file specified.")
        parser.print_usage()
        return 2
    try:
        design = load_design(input_path, args.library)
    except (OSError, ValueError) as err:
        log.error(f"Could not load optical design: {err}")
        return 1
    scene = build_scene(design, nb_rays=args.rays)
    log.info(f"Built scene for '{scene['name']}': {len(scene['lenses'])} lenses, {len(scene['rays'])} rays.")
    if args.save_scene:
        Path(args.save_scene).write_text(scene_json(scene), encoding="utf-8")
        log.info(f"Saved scene to {args.save_scene}.")
    if args.output:
        Path(args.output).write_text(render_html(scene, served=False), encoding="utf-8")
        log.info(f"Wrote viewer page to {args.output}.")
    if not args.no_serve:
        serve(
            scene,
            port=args.port,
            open_browser=not args.no_open,
            watch=Path(input_path) if args.watch else None,
            exit_after_open=not (args.keep or args.watch),
        )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(oshow())
