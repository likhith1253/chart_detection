# Ground-truth format (schema 1.1)

Every generated image `images/<name>.png` has a matching `labels/<name>.json`. The dataset also gets an `annotations.csv` with one row per object, and optionally YOLO labels and instance masks.

## Coordinate convention

- Pixel coordinates are floating point, with **(0, 0) at the centre of the top-left pixel**. x increases to the right and y increases downward.
- Object-space coordinates are in µm, with (0, 0) at the top-left *corner* of the image, so `x_px = x_um / pixel_size_um − 0.5`.
- `z_um` is the height above the background glass. 0 is the background glass and `film_thickness_um` is the cover glass.

## Per-image JSON

```json
{
  "schema_version": "1.1",
  "image": "images/img_00000.png",
  "frame": null,                       // frame index for sequences, null for independent images
  "time_s": 0.0,                       // simulation time (sequences)
  "seed": 0,                           // seed that reproduces this image
  "config_fingerprint": "ae93a499fbf1",// hash of the full resolved config
  "image_size_px": [5600, 5600],
  "bit_depth": 8,
  "pixel_size_um": 2.0,                // TRUE um/px - what calibration should recover
  "film_thickness_um": 200,
  "focus_depth_um": 100.0,
  "optics": {
    "n_liquid": 1.47, "n_bubble": 1.0,
    "objective_na": 0.25, "condenser_na": 0.12,
    "critical_angle_deg": 42.865
  },
  "reference_square": {
    "side_um": 10000,                  // outer edge length
    "line_width_um": 60,
    "rotation_deg": -1.8361,
    "center_px": [2820.044, 2764.968],
    "outer_corners_px": [[241.226, 346.353], [5238.659, 186.15],
                         [5398.862, 5183.583], [401.429, 5343.786]],
    "side_px": 5000.0,
    "um_per_px": 2.0
  },
  "ground_truth_iso": {
    "volume_ml": 0.01952288,           // square interior area x film thickness
    "counts":        {">4um": 80, ">6um": 17, ">14um": 2},
    "counts_per_ml": {">4um": 4097.756, ">6um": 870.773, ">14um": 102.444},
    "iso4406_code": "19/17/14",        // solid particles only, bubbles excluded
    "bubbles_inside_square": 18
  },
  "objects": [
    {
      "id": 0,
      "class": "particle",             // "particle" | "bubble"
      "x_px": 4646.402, "y_px": 55.246,
      "x_um": 9293.805, "y_um": 111.491, "z_um": 74.629,
      "size_um": 4.396,                // max Feret diameter (bubble: diameter)
      "ecd_um": 3.964,                 // equivalent circular diameter of projected area
      "bbox_px": [4645.31, 54.21, 4647.42, 56.24],   // x0, y0, x1, y1
      "inside_reference_square": false,// centre inside the square's interior
      "iso_channels": {">4um": true, ">6um": false, ">14um": false},
      "velocity_um_s": [1871.259, 0.0],
      "transmission": 0.0326,          // particles only
      "outline_px": [[4646.6, 56.24], [4646.3, 56.22], "..."]  // particles only, polygon vertices
    },
    {
      "id": 104,
      "class": "bubble",
      "x_px": 1044.183, "y_px": 5324.291,
      "x_um": 2089.366, "y_um": 10649.582, "z_um": 98.762,
      "size_um": 53.0, "ecd_um": 53.0,
      "bbox_px": [1030.93, 5311.04, 1057.43, 5337.54],
      "inside_reference_square": false,
      "iso_channels": {">4um": true, ">6um": true, ">14um": true},
      "velocity_um_s": [1999.693, 0.0]
    }
  ]
}
```

(Comments are explanatory only. The files are plain JSON.)

Ids are unique within an image and **stay the same across all frames of a sequence**, so they can serve as track ids. Objects can lie partly outside the image. Clip their boxes if you need to.

## annotations.csv

One row per object for every image in the dataset:

| Column | Meaning |
|---|---|
| `image`, `frame` | Relative image path, and frame index (empty for static images) |
| `id`, `class` | Object id and class (`particle` / `bubble`) |
| `x_px`, `y_px`, `x_um`, `y_um`, `z_um` | Position |
| `size_um`, `ecd_um` | Maximum Feret diameter and equivalent circular diameter |
| `bbox_x0`, `bbox_y0`, `bbox_x1`, `bbox_y1` | Bounding box in pixels |
| `inside_reference_square` | True if the object counts towards the ISO code (particles only) |
| `gt4`, `gt6`, `gt14` | ISO channel membership |
| `vx_um_s`, `vy_um_s` | Flow velocity |

## YOLO labels (`--yolo`)

`yolo/<name>.txt` holds one line per object: `class cx cy w h`, normalised to image width and height, with boxes clipped to the image. Class ids are **0 = particle** and **1 = bubble**.

## Instance masks (`--masks`)

`masks/<name>.png` is a 16-bit single-channel PNG: 0 is background and `id + 1` is the object with that id. Particles are drawn as their polygons and bubbles as filled discs. Bubbles are drawn last, so they win where they overlap a particle.

## dataset.json

Holds the generator name, the kind (`images` or `sequence`), the elapsed time, the **fully resolved config**, and a summary: image, particle and bubble counts, the particle size range, and the ISO code of each image.
