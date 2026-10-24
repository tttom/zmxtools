# oshow

View optical designs in an interactive 3D browser window.

## Usage

### Command line interface
Open and display a model using
```console
oshow mylens.zar                             # positional input, like unzar
oshow -i mylens.zar                          # load from a ZAR archive
oshow -i mylens.zmx -m schott.agf            # load a ZMX with extra material catalogs
oshow -i mylens.zmx -o view.html --no-serve  # write a standalone page
oshow -i mylens.zmx -j scene.json            # also save the scene model
oshow -i mylens.zmx --watch                  # live-rebuild when the input changes
oshow -i mylens.zmx --keep                   # keep serving after the page has loaded
oshow mylens/                                # view a directory (output of `unzar`)
oshow -i mylens.zip                          # view a zip archive (output of `unzar -z`)
unzar mylens.zar --show                      # unpack the archive and view the module
```

### Function
The tool starts a local `http.server` on `http://127.0.0.1:8010/` (change with
`--port`) and opens the browser. The page embeds the full scene, so the server
exits by default once the model has been displayed; use `--keep` to keep serving
or `--watch` to also live-rebuild on input changes. Besides `.zar` and `.zmx`
files, oshow loads the output of `unzar` directly: a directory with the extracted
`.zmx` and `.agf` files, or the `.zip` archive written by `unzar -z`.

The window is split: a description table on the left lists the source, every
surface/material pair, and the image surface. Clicking a block highlights the
matching rays, surface cap, lens rim, or image plane in the 3D view. The viewer
starts with an x-axis view, z pointing down, the source at the top and the image
at the bottom, and refits the model when the window is resized. 

### Navigation
Drag to rotate, use the two-finger-pinch or mouse wheel to zoom. The keyboard arrow keys move the viewer (camera-relative), while `Home` and `End` reset the camera to look directly at the aperture stop, `PageUp`/`PageDown` rotate the camera 90 degrees. 

The aperture stop center is the origin; z is the propagation axis, x is horizontal, y is up. 
A ◐ button in the top-right corner switches between light and dark mode; the initial mode
follows the OS color scheme when available and falls back to dark. All colors,
opacities, and view constants are defined in `src/zmxtools/gui/oshow.css`.

### Exporting the rendering
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
      "id": "lens-0",
      "color": "#rrggbb",
      "opacity": 0.5,
      "surfaces": [1, 2],
      "meshes": [
        {"vertices": [[0.0, 0.0, 0.0]], "triangles": [[0, 1, 2]], "surface_id": "surf-1"}
      ],
      "rim_lines": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
      "rims": [[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]]
    }
  ],
  "rays": [
    {"color": "#rrggbb", "points": [[0.0, 0.0, 0.0]]}
  ],
  "aperture_stop": {"center": [0.0, 0.0, 0.0], "inner_radius": 0.0, "outer_radius": 0.0,
                    "color": "#808088", "opacity": 0.5},
  "image_plane": {"center": [0.0, 0.0, 0.0], "radius": 0.0},
  "camera": {"position": [0.0, 0.0, 0.0], "target": [0.0, 0.0, 0.0], "up": [0.0, 0.0, -1.0]},
  "bounds": {"radius": 0.0, "zmin": 0.0, "zmax": 0.0},
  "table": [
    {"type": "source", "title": "Source", "details": [["Wavelengths", "587.6 nm"]],
     "target": {"kind": "rays"}},
    {"type": "surface", "title": "Surface 1", "details": [["Type", "Standard"]],
     "target": {"kind": "surface", "id": "surf-1"}},
    {"type": "material", "title": "Material 1", "details": [["Material", "N-BK7"]],
     "color": "#22d3ee", "target": {"kind": "rim", "id": "lens-0"}},
    {"type": "image", "title": "Image surface", "details": [["Type", "Standard"]],
     "target": {"kind": "image"}}
  ]
}
```

- `lenses`: one entry per lens element, distinguished by `color` and `id`.
  `meshes` are triangulated semi-transparent (`opacity`) caps and cylinder sides.
  Only the outer side of each lens is drawn: the first and last surface caps are
  tagged with `surface_id` for selection, and the back cap faces outward. `rims`
  are closed polylines marking the cap/cylinder intersections in the same color;
  `rim_lines` is their flattened concatenation for backwards compatibility.
- `axis_line`: the optical axis, from the source to the image plane.
- `rays`: traced ray polylines, all starting at the source (before the first
  surface); rays sharing a start point share a `color`. When the source of the
  image or the image is at infinity, the rays are drawn as parallel rays
  sticking out beyond the first or final surface by the maximum of the rest of
  the path length and five times the aperture diameter.
- `aperture_stop`: a semi-transparent dark silver annulus marking the stop: a
  disk with a hole of `inner_radius` (the stop aperture) out to `outer_radius`.
- `image_plane`: an opaque disk at the image, with `radius` equal to the field
  size at the image plane.
- `camera`: the initial vantage point, looking along the x-axis with z down.
- `bounds`: drawing extents used for camera framing and keyboard navigation.
- `table`: the left-hand description blocks, in top-to-bottom order. Each block
  has a `type`, `title`, and `details`; selectable blocks carry a `target` with
  `kind` `rays`, `surface`, `rim`, or `image`.
