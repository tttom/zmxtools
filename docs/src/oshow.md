# oshow

View optical designs in an interactive 3D browser window.

## Usage

```console
oshow mylens.zar                             # positional input, like unzar
oshow -i mylens.zar                          # load from a ZAR archive
oshow -i mylens.zmx -l schott.agf            # load a ZMX with extra glass catalogs
oshow -i mylens.zmx -o view.html --no-serve  # write a standalone page
oshow -i mylens.zmx -s scene.json            # also save the scene model
oshow -i mylens.zmx --watch                  # live-rebuild when the input changes
oshow -i mylens.zmx --keep                   # keep serving after the page has loaded
oshow mylens/                                # view a directory (output of `unzar`)
oshow -i mylens.zip                          # view a zip archive (output of `unzar -z`)
unzar mylens.zar --show                      # unpack the archive and view the module
```

The tool starts a local `http.server` on `http://127.0.0.1:8010/` (change with
`--port`) and opens the browser. The page embeds the full scene, so the server
exits by default once the model has been displayed; use `--keep` to keep serving
or `--watch` to also live-rebuild on input changes. Besides `.zar` and `.zmx`
files, oshow loads the output of `unzar` directly: a directory with the extracted
`.zmx` and `.agf` files, or the `.zip` archive written by `unzar -z`. Navigation:
drag to orbit, wheel to zoom, arrow keys to pan, `Home` resets the view, `End`
looks along the x-axis toward negative values, `PageUp`/`PageDown` dolly in and
out. The aperture stop center is the origin; z is the propagation axis, x is
horizontal, y is up. A ◐ button in the top-right corner switches between light
and dark mode; the initial mode follows the OS color scheme when available and
falls back to dark.

## Scene format

The scene is a single JSON object, version 2. All lengths are in meters, with
the aperture stop center at the origin.

```json
{
  "version": 2,
  "name": "<design name>",
  "axes": {"length": 0.0},
  "axis_line": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
  "lenses": [
    {
      "color": "#rrggbb",
      "opacity": 0.5,
      "meshes": [
        {"vertices": [[0.0, 0.0, 0.0]], "triangles": [[0, 1, 2]]}
      ],
      "rim_lines": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    }
  ],
  "rays": [
    {"color": "#rrggbb", "points": [[0.0, 0.0, 0.0]]}
  ],
  "aperture_stop": {"center": [0.0, 0.0, 0.0], "inner_radius": 0.0, "outer_radius": 0.0},
  "image_plane": {"center": [0.0, 0.0, 0.0], "radius": 0.0},
  "camera": {"position": [0.0, 0.0, 0.0], "target": [0.0, 0.0, 0.0]},
  "bounds": {"radius": 0.0, "zmin": 0.0, "zmax": 0.0}
}
```

- `lenses`: one entry per lens element, distinguished by `color`. `meshes` are
  triangulated semi-transparent (`opacity`) caps and cylinder sides; `rim_lines`
  are opaque line-segment pairs marking the cap/cylinder intersections in the
  same color.
- `axis_line`: the optical axis, from the source to the image plane.
- `rays`: traced ray polylines, all starting at the source (before the first
  surface); rays sharing a start point share a `color`. When the source of the
  image or the image is at infinity, the rays are drawn as parallel rays
  sticking out beyond the first or final surface by the maximum of the rest of
  the path length and five times the aperture diameter.
- `aperture_stop`: an opaque dark silver annulus marking the stop: a disk with a
  hole of `inner_radius` (the stop aperture) out to `outer_radius`.
- `image_plane`: an opaque disk at the image, with `radius` equal to the field
  size at the image plane.
- `camera`: the initial vantage point showing the whole design from an angle.
- `bounds`: drawing extents used for camera framing and keyboard navigation.

## Saving models to disk

The scene JSON *is* the save format: `oshow -s scene.json` writes exactly what
the viewer renders, and `render_html(json.load(open('scene.json')))` re-displays
it without re-parsing the optical design. This is intentionally a *view* model:
it stores tessellated geometry, not parametric surfaces. A future parametric
save format should round-trip the design itself (e.g. writing a normalized ZMX
command dictionary), which the existing `parser`/`zmx` modules can already
produce via `str(design)`.

## 2D slices (planned)

Slices are a view over the same scene data, so no new model is required:

- Intersect each lens mesh with a plane (`y = 0` for the classic x-z lens
  cross-section) by clipping triangles at the plane and drawing the resulting
  polylines; rays project onto the plane.
- Implement as a `--slice xz|xy|yz` flag that filters the scene server-side (a
  `/slice.json` endpoint) and a 2D canvas renderer, keeping the same
  live-reload mechanism.
