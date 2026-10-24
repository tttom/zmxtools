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
import logging
import math
import re
import threading
import time
import webbrowser
import zipfile
from collections.abc import Sequence
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar, cast

import numpy as np

import zmxtools
from zmxtools import zar
from zmxtools.agf import AgfMaterialLibrary
from zmxtools.optical_design import material as material_module
from zmxtools.utils.array import array_type, asarray
from zmxtools.utils.io import BytesFile
from zmxtools.zmx import ZmxOpticalDesign, ZmxSource, ZmxSurface

log = zmxtools.config_logger(__name__)

__all__ = ["build_scene", "load_design", "oshow", "render_html", "scene_json", "serve"]

D_LINE_WAVENUMBER = 2 * np.pi / 587.5618e-9

_CSS_VARIABLE = re.compile(r"--([A-Za-z0-9_-]+)\s*:\s*([^;]+);")


@lru_cache(maxsize=1)
def _style_constants() -> dict[str, str]:
    """The design constants declared in ``oshow.css``."""
    css = importlib.resources.files("zmxtools.gui").joinpath("oshow.css").read_text(encoding="utf-8")
    constants = dict[str, str]()
    for match in _CSS_VARIABLE.finditer(css):
        constants.setdefault(match.group(1), match.group(2).strip())
    return constants


def _style_value(name: str, default: str = "") -> str:
    """The value of a custom CSS property, with an optional fallback."""
    return _style_constants().get(name, default)


def _style_number(name: str, default: float) -> float:
    """The numeric value of a custom CSS property, with a fallback."""
    try:
        return float(_style_value(name))
    except ValueError:
        return default


def _lens_colors() -> tuple[str, ...]:
    """The alternating lens surface and rim colors, from ``oshow.css``."""
    colors = list[str]()
    index = 0
    while color := _style_value(f"lens-color-{index}"):
        colors.append(color)
        index += 1
    return tuple(colors) or ("#22d3ee", "#4ade80")


def _ray_colors() -> tuple[str, ...]:
    """The ray colors, from ``oshow.css``."""
    colors = list[str]()
    index = 0
    while color := _style_value(f"ray-color-{index}"):
        colors.append(color)
        index += 1
    return tuple(colors) or ("#ffeb3b", "#ff5252", "#40c4ff", "#69f0ae", "#e040fb", "#ffd740")


BACKGROUND_MATERIAL_NAMES = {"vacuum", "air", "water", "__blank", "blank"}


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


def _cap_mesh(points: array_type, z_offset: float, reverse: bool = False) -> tuple[list[list[float]], list[list[int]]]:
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
            if reverse:
                triangles.append([a, c, b])
                triangles.append([a, d, c])
            else:
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
    return str(mat.name).strip().lower() not in BACKGROUND_MATERIAL_NAMES


def _group_lenses(surfaces: Sequence[ZmxSurface]) -> list[list[int]]:
    """Group surface indices into lens elements: consecutive surfaces sharing a glass."""
    lenses = list[list[int]]()
    last = len(surfaces) - 1
    index = 1
    while index < last:
        surface = surfaces[index]
        mat = surface.material
        if getattr(surface, "type", "") == "COORDBRK" or not (
            _is_glass(mat) and not getattr(surface, "is_mirror", False)
        ):
            index += 1
            continue
        indices = [index]
        while index + 1 < last:
            next_surface = surfaces[index + 1]
            next_mat = next_surface.material
            if getattr(next_surface, "type", "") == "COORDBRK":
                break
            if not (_is_glass(next_mat) and not getattr(next_surface, "is_mirror", False)):
                break
            if str(next_mat.name) != str(mat.name):
                break
            index += 1
            indices.append(index)
        indices.append(index + 1)
        lenses.append(indices)
        index = indices[-1]
    return lenses


def _lens_regions(
    surfaces: Sequence[ZmxSurface], zs: Sequence[float], radius: float
) -> list[tuple[int, list[int], float]]:
    """The visible lens elements with their drawing index, surfaces, and radius."""
    regions = list[tuple[int, list[int], float]]()
    for indices in _group_lenses(surfaces):
        lens_radius = _lens_radius(surfaces, zs, indices, radius)
        if lens_radius <= 0:
            log.debug(f"Skipping degenerate lens at surfaces {indices}.")
            continue
        regions.append((len(regions), indices, lens_radius))
    return regions


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
        crossings = np.flatnonzero(thickness < 0)
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
    ray_palette = _ray_colors()
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
            color = ray_palette[color_idx % len(ray_palette)]
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
        color = ray_palette[color_idx % len(ray_palette)]
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


def _surface_type_name(surface: ZmxSurface) -> str:
    """Human-readable surface type name."""
    names = {
        "STANDARD": "Standard",
        "EVENASPH": "Even asphere",
        "ODDASPH": "Odd asphere",
        "PARAXIAL": "Paraxial",
        "COORDBRK": "Coordinate break",
        "TILTSURF": "Tilted surface",
        "TOROIDAL": "Toroidal",
        "SZERNSAG": "Zernike Standard Sag",
        "FZERNSAG": "Zernike Fringe Sag",
        "SZERNPHA": "Zernike Standard Phase",
        "FZERNPHA": "Zernike Fringe Phase",
        "XPOLYNOM": "Extended polynomial",
        "IRREGULA": "Irregular",
        "BICONICX": "Biconic X",
        "BLACKBOX": "Black box",
        "GRID_SAG": "Grid sag",
    }
    surface_type = str(getattr(surface, "type", "STANDARD")).upper()
    return names.get(surface_type, surface_type.replace("_", " ").title())


_LENGTH_PREFIXES = (
    (1e-9, "nm", "/ nm"),
    (1e-6, "µm", "/ µm"),
    (1e-3, "mm", "/ mm"),
    (1e-2, "cm", "/ cm"),
    (1e-1, "dm", "/ dm"),
    (1.0, "m", "/ m"),
    (1e3, "km", "/ km"),
)


def _length_prefix(design: ZmxOpticalDesign) -> tuple[float, str, str]:
    """The length scale, suffix, and inverse suffix used by a design."""
    unit = float(getattr(design, "unit", 1.0))
    for scale, suffix, inverse in _LENGTH_PREFIXES:
        if math.isclose(unit, scale, rel_tol=1e-9, abs_tol=0.0):
            return scale, suffix, inverse
    if unit < 1e-6:
        return _LENGTH_PREFIXES[0]
    if unit < 1e-3:
        return _LENGTH_PREFIXES[1]
    if unit < 1.0:
        return _LENGTH_PREFIXES[2]
    return _LENGTH_PREFIXES[5]


def _format_length(value: float, prefix: tuple[float, str, str]) -> str:
    """Format a length in the design's consistent SI prefix."""
    scale, suffix, _ = prefix
    if value is None or not np.isfinite(value):
        return "Infinity"
    if abs(value) < 1e-15:
        return f"0 {suffix}"
    return f"{value / scale:.6g} {suffix}"


def _surface_curvature(surface: ZmxSurface, prefix: tuple[float, str, str]) -> tuple[str, str | None]:
    """The formatted curvature and radius of a surface."""
    _, _, inverse = prefix
    curvature = float(getattr(surface, "curvature", 0.0))
    if not np.isfinite(curvature) or abs(curvature) < 1e-18:
        return f"0 {inverse}", None
    radius = 1.0 / curvature
    return f"{curvature * prefix[0]:.4g} {inverse}", _format_length(radius, prefix)


def _surface_summary(
    surface: ZmxSurface,
    prefix: tuple[float, str, str],
    label: str | None = None,
) -> str:
    """A concise one-line surface description."""
    curvature, radius = _surface_curvature(surface, prefix)
    name = label if label is not None else _surface_type_name(surface)
    summary = f"{name} surface: κ = {curvature}"
    if radius is not None:
        summary += f", r = {radius}"
    return summary


def _refractive_index_at_wavelength(mat: material_module.Material, wavelength: float) -> float:
    """Refractive index at a given wavelength, with fallbacks."""
    try:
        wavenumber = 2.0 * np.pi / wavelength
        n = mat.complex_refractive_index(wavenumber=wavenumber)
        real = float(np.real(n))
        if np.isfinite(real) and real > 0.0:
            return real
    except Exception:
        pass
    try:
        real = float(np.real(mat.refractive_index(wavelength=wavelength)))
        if np.isfinite(real) and real > 0.0:
            return real
    except Exception:
        pass
    try:
        real = float(np.real(mat.refractive_index_d))
        if np.isfinite(real) and real > 0.0:
            return real
    except Exception:
        pass
    return 1.0


def _middle_wavelength(wavelengths: Sequence[float]) -> float:
    """The middle modeled wavelength, falling back to the d-line."""
    finite = sorted(float(w) for w in wavelengths if np.isfinite(w) and w > 0.0)
    if not finite:
        return 587.5618e-9
    return finite[len(finite) // 2]


def _abbe_number(mat: material_module.Material) -> float | None:
    """The Abbe number of a glass, if it has one."""
    try:
        value = float(np.real(mat.constringence))
    except Exception:
        return None
    if not np.isfinite(value) or value <= 0.0:
        return None
    return value


def _command_numbers(design: ZmxOpticalDesign, *names: str) -> list[float]:
    """The numbers of the first available command among names."""
    commands = design.commands
    for name in names:
        if name in commands:
            return [float(value) for value in commands.command(name).numbers]
    return []


def _field_positions(
    design: ZmxOpticalDesign,
    prefix: tuple[float, str, str],
) -> tuple[str, list[str]]:
    """The field type and field positions, formatted for display."""
    xs = _command_numbers(design, "XFLD", "XFLN")
    ys = _command_numbers(design, "YFLD", "YFLN")
    count = max(len(xs), len(ys))
    if count == 0:
        return "", []
    xs = [*xs, *(0.0 for _ in range(count - len(xs)))]
    ys = [*ys, *(0.0 for _ in range(count - len(ys)))]
    ftyp = _command_numbers(design, "FTYP")
    field_type = int(ftyp[0]) if len(ftyp) > 0 else 0
    field_count = int(ftyp[-1]) if len(ftyp) > 0 and ftyp[-1] > 0 else 0
    used = max(
        [field_count]
        + [index + 1 for index, (x, y) in enumerate(zip(xs, ys, strict=True)) if abs(x) > 1e-15 or abs(y) > 1e-15],
    )
    used = min(max(used, 0), count)
    if used == 0:
        return "", []
    units = {0: "deg", 1: prefix[1], 2: "image height", 3: "image height"}
    unit = units.get(field_type, "")
    type_names = {0: "Angles", 1: "Object heights", 2: "Paraxial image heights", 3: "Real image heights"}
    type_name = type_names.get(field_type, "Fields")
    positions = []
    for x, y in zip(xs[:used], ys[:used], strict=True):
        if field_type == 0:
            positions.append(f"x={x:g}°, y={y:g}°")
        else:
            scale = float(getattr(design, "unit", 1.0))
            x_str = _format_length(x * scale, prefix)
            y_str = _format_length(y * scale, prefix)
            positions.append(f"x={x_str}, y={y_str}")
    return f"{type_name} ({unit})" if unit else type_name, positions


def _material_name(mat: material_module.Material) -> str:
    """The display name of a material."""
    name = str(getattr(mat, "name", "")).strip()
    if not name:
        return "Air"
    if name.lower() == "vacuum":
        return "Vacuum"
    if name.lower() == "air":
        return "Air"
    if name.lower() == "water":
        return "Water"
    if name.lower() in {"__blank", "blank"}:
        return "Vacuum"
    return name


def _build_table(design: ZmxOpticalDesign, surfaces: Sequence[ZmxSurface]) -> list[dict]:
    """Build the description table for the left panel."""
    table = list[dict]()
    prefix = _length_prefix(design)
    zs = _surface_zs(surfaces)
    radius = _max_radius(surfaces)
    lens_by_surface = dict[int, int]()
    for lens_index, indices, _ in _lens_regions(surfaces, zs, radius):
        for surface_index in indices:
            lens_by_surface[surface_index] = lens_index
    lens_colors = _lens_colors()
    source = cast("ZmxSource", design.source)
    wavelengths = []
    try:
        wavelengths = [float(w) for w in np.asarray(source.wavelengths).flatten()]
    except Exception:
        pass
    weights = []
    try:
        weights = [float(w) for w in np.asarray(source.wavelength_weights).flatten()]
    except Exception:
        pass
    weights = [*weights, *(1.0 for _ in range(len(wavelengths) - len(weights)))][: len(wavelengths)]
    wave_strs = []
    for wavelength, weight in zip(wavelengths, weights, strict=True):
        if not np.isfinite(wavelength):
            continue
        weight_str = f" ({weight:g})" if abs(weight - 1.0) > 1e-12 else ""
        wave_strs.append(f"{wavelength / 1e-9:.1f} nm{weight_str}")
    source_details = [(chr(955), ", ".join(wave_strs) if wave_strs else "d-line")]
    field_label, field_positions = _field_positions(design, prefix)
    if field_positions:
        source_details.append((field_label or "Fields", "; ".join(field_positions)))
    else:
        source_details.append(("Fields", "On-axis"))
    object_at_infinity, _ = _infinity_flags(surfaces)
    if object_at_infinity:
        source_details.append(("Δz", "Infinity"))
    elif len(surfaces) > 1 and np.isfinite(float(surfaces[0].spacing)):
        source_details.append(("Δz", _format_length(float(surfaces[0].spacing), prefix)))
    table.append(
        {
            "type": "medium",
            "title": "",
            "details": source_details,
            "target": {"kind": "rays"},
        }
    )

    middle_wavelength = _middle_wavelength(wavelengths)
    for surface_index in range(1, len(surfaces) - 1):
        surface = surfaces[surface_index]
        if getattr(surface, "type", "") == "COORDBRK":
            continue
        curvature, radius_str = _surface_curvature(surface, prefix)
        surface_details = [("Type", _surface_type_name(surface)), ("Curvature", f"κ = {curvature}")]
        if radius_str is not None:
            surface_details.append(("Radius", f"r = {radius_str}"))
        if getattr(surface, "stop", False):
            surface_details.append(("Stop", "Yes"))
        if getattr(surface, "conic_constant", 0.0) != 0.0:
            surface_details.append(("Conic", f"{surface.conic_constant:g}"))
        if getattr(surface, "type", "") == "PARAXIAL" and getattr(surface, "focal_length", None) is not None:
            surface_details.append(("Focal length", _format_length(float(surface.focal_length), prefix)))
        parameters = getattr(surface, "parameters", None)
        if parameters is not None and len(parameters) > 0:
            visible = [f"{value:g}" for value in parameters if abs(float(value)) > 1e-18]
            if visible:
                surface_details.append(("Parameters", ", ".join(visible)))
        extra_data = getattr(surface, "extra_data", None)
        if extra_data is not None and len(extra_data) > 0:
            visible = [f"{value:g}" for value in extra_data if abs(float(value)) > 1e-18]
            if visible:
                surface_details.append(("Extra data", ", ".join(visible)))
        table.append(
            {
                "type": "surface",
                "title": "",
                "summary": _surface_summary(surface, prefix),
                "details": surface_details,
                "stop": bool(getattr(surface, "stop", False)),
                "target": {"kind": "surface", "id": f"surf-{surface_index}"},
            }
        )

        mat = surface.material
        is_glass = _is_glass(mat) and not getattr(surface, "is_mirror", False)
        thickness = (
            _format_length(float(surface.spacing), prefix)
            if surface_index < len(surfaces) - 1 and np.isfinite(float(surface.spacing))
            else "Infinity"
        )
        material_name = _material_name(mat)
        index = _refractive_index_at_wavelength(mat, middle_wavelength)
        abbe = _abbe_number(mat) if is_glass else None
        summary = f"{material_name}: Δz = {thickness}, n = {index:.4g}"
        if abbe is not None:
            summary += f", V<sub>d</sub> = {abbe:.3f}"
        material_block = {
            "type": "material",
            "title": "",
            "summary": summary,
            "details": [],
        }
        lens_index = lens_by_surface.get(surface_index) if is_glass else None
        if lens_index is not None:
            material_block["color"] = lens_colors[lens_index % len(lens_colors)]
            material_block["target"] = {"kind": "rim", "id": f"lens-{lens_index}"}
        elif material_name in {"Vacuum", "Air", "Water"} or not is_glass:
            material_block["type"] = "medium"
        table.append(material_block)

    image_surface = surfaces[-1]
    image_curvature, image_radius = _surface_curvature(image_surface, prefix)
    image_details = [("Type", _surface_type_name(image_surface)), ("Curvature", f"κ = {image_curvature}")]
    if image_radius is not None:
        image_details.append(("Radius", f"r = {image_radius}"))
    _, image_at_infinity = _infinity_flags(surfaces)
    if image_at_infinity:
        image_details.append(("Image", "At infinity"))
    table.append(
        {
            "type": "image",
            "title": "",
            "summary": _surface_summary(image_surface, prefix, label="Image"),
            "details": image_details,
            "target": {"kind": "image"},
        }
    )
    return table


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

    lens_colors = _lens_colors()
    lenses = list[dict]()
    for element_idx, indices, lens_radius in _lens_regions(surfaces, zs, radius):
        color = lens_colors[element_idx % len(lens_colors)]
        meshes = list[dict]()
        front_index, back_index = indices[0], indices[-1]
        front_points = _sample_sag(surfaces[front_index], lens_radius)
        back_points = _sample_sag(surfaces[back_index], lens_radius)
        front_rim = _rim(front_points, zs[front_index] - z_stop)
        back_rim = _rim(back_points, zs[back_index] - z_stop)
        side_vertices, side_triangles = _side_mesh(front_rim, back_rim)
        meshes.append({"vertices": side_vertices, "triangles": side_triangles, "surface_id": f"rim-{element_idx}"})
        front_vertices, front_triangles = _cap_mesh(front_points, zs[front_index] - z_stop)
        meshes.append(
            {
                "vertices": front_vertices,
                "triangles": front_triangles,
                "surface_id": f"surf-{front_index}",
            },
        )
        back_vertices, back_triangles = _cap_mesh(back_points, zs[back_index] - z_stop, reverse=True)
        meshes.append(
            {
                "vertices": back_vertices,
                "triangles": back_triangles,
                "surface_id": f"surf-{back_index}",
            },
        )
        rims = [*front_rim, *back_rim]
        lenses.append(
            {
                "id": f"lens-{element_idx}",
                "color": color,
                "opacity": _style_number("lens-opacity", 0.5),
                "meshes": meshes,
                "rim_lines": rims,
                "rims": [front_rim, back_rim],
                "surfaces": [front_index, back_index],
            },
        )

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
    table = _build_table(design, surfaces)
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
            "color": _style_value("aperture-color", "#808088"),
            "opacity": _style_number("aperture-opacity", 0.5),
        },
        "image_plane": {"center": [0.0, 0.0, z_max], "radius": image_radius},
        "camera": {
            "position": [extent, 0.0, 0.5 * (z_min + z_max)],
            "target": [0.0, 0.0, 0.5 * (z_min + z_max)],
            "up": [0.0, 0.0, -1.0],
        },
        "bounds": {"radius": radius, "zmin": z_min, "zmax": z_max},
        "table": table,
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
    package = importlib.resources.files("zmxtools.gui")
    template = package.joinpath("oshow_template.html").read_text(encoding="utf-8")
    css = package.joinpath("oshow.css").read_text(encoding="utf-8")
    js = package.joinpath("oshow.js").read_text(encoding="utf-8")
    html = template.replace("__CSS__", css).replace("__JS__", js)
    return (
        html.replace("__TITLE__", str(scene.get("name", "optical design")))
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
    server = ThreadingHTTPServer(("127.0.0.1", port), _ViewerHandler)
    server.timeout = 0.5
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
    parser.add_argument(
        "-m",
        "--materials",
        type=str,
        nargs="*",
        default=[],
        help="extra materials in, for instance, .agf glass catalog files",
    )
    parser.add_argument("-o", "--output", type=str, help="write the HTML page to this file")
    parser.add_argument("-j", "--save-json", type=str, help="write the reandering in JSON format to this file")
    parser.add_argument("--rays", type=int, default=13, help="approximate number of rays to trace (default: %(default))")
    parser.add_argument("--port", type=int, default=8010, help="port for the viewer server (default: %(default))")
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
    parser.add_argument(
        "-v",
        "--verbosity",
        help="the level of verbosity",
        default=0,
        action="count",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        help="suppress output partially or completely (-qqqq)",
        default=0,
        action="count",
    )
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

    verbosity_level: int = args.verbosity - args.quiet
    log_levels = logging.CRITICAL, logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG, logging.NOTSET
    DEFAULT_VERBOSITY: int = 2
    zmxtools.console_log_handler.level = log_levels[max(0, min(len(log_levels) - 1, DEFAULT_VERBOSITY + verbosity_level))]  # Default to WARNING

    version_message: str = f"Running zmxtools oshow version {zmxtools.__version__}."
    log.debug(version_message)
    if args.version and verbosity_level >= -DEFAULT_VERBOSITY:
        print(version_message)
        if args.input is None and args.input_file is None:
            return 0
    if args.input is not None and args.input_file is not None:
        log.error("Specify the input file with -i or as the first argument, not both.")
        if verbosity_level >= -DEFAULT_VERBOSITY:
            parser.print_usage()
        return 2
    input_path = args.input if args.input is not None else args.input_file
    if input_path is None:
        log.error("No input file specified.")
        if verbosity_level >= -DEFAULT_VERBOSITY:
            parser.print_usage()
        return 2
    try:
        design = load_design(input_path, args.materials)
    except (OSError, ValueError) as err:
        log.error(f"Could not load optical design: {err}")
        return 1
    scene = build_scene(design, nb_rays=args.rays)
    log.info(f"Built scene for '{scene['name']}': {len(scene['lenses'])} lenses, {len(scene['rays'])} rays.")
    if args.save_json:
        Path(args.save_json).write_text(scene_json(scene), encoding="utf-8")
        log.info(f"Saved scene to {args.save_json}.")
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
