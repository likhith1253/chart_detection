# Physics model

This document describes the optical, sensor and population models implemented in `src/fluidsim/`. All lengths are in micrometres (µm).

## 1. Geometry and coordinate system

```
            camera / objective (NA_obj)
                     ▲
   ──────────────────┼──────────────────  cover glass            z = h
     oil film (n_liq = 1.47)   ○ bubble (n = 1.00)
   - - - - - - - - - focal plane - - - -                          z = z_f (default h/2)
         ▬ particle (thin opaque flake at depth z_p)
   ═══════[printed 10 mm square]════════  background glass       z = 0
                     ▲
            condenser cone (NA_cond)
```

- x points right and y points down (image convention). z is the optical axis, with light travelling in +z.
- z = 0 is the top surface of the background glass, where the reference square is printed. z = h is the underside of the cover glass. The default film thickness is h = 200 µm.
- Pixel *(i, j)* covers `[j·p, (j+1)·p) × [i·p, (i+1)·p)` in object space, where p is the pixel size. In the ground truth, pixel coordinate (0, 0) is the **centre** of the top-left pixel, so `x_px = x_um / p − 0.5`.

## 2. Ray model (`raycaster.py`)

Each output pixel is sampled with `rays_per_pixel` stratified, jittered samples.

1. **Focal-plane point.** Each sample picks a point (x_f, y_f) inside the pixel on the focal plane.
2. **Illumination direction.** The direction is sampled uniformly over the disk of direction sines inside the illumination cone. The maximum sine in the oil is `sin θ_max = NA_cond / n_liq`.
3. **Primary ray.** The primary ray is the straight line through (x_f, y_f, z_f) with that direction, started at z = 0. In an empty cell this ray would be imaged straight back onto (x_f, y_f), so an empty pixel receives exactly 1.0. The rendered quantity is a **transmission map**.
4. **Tracing.** The ray is traced through the scene with an explicit stack of up to 64 branches and at most `max_ray_depth` surface events. Branches whose weight falls below `min_ray_weight` are dropped.
   - **Particle flake.** If the ray crosses the flake's plane inside its polygon (even–odd point-in-polygon test), the weight is multiplied by the flake's transmission and the ray continues.
   - **Bubble surface.** At a sphere hit the ray splits into a reflected branch (weight R) and a refracted branch (weight 1 − R) using the Fresnel equations below.
   - **Cover glass.** A ray reaching z = h is **collected** only if `n_liq · sin θ ≤ NA_obj`. Otherwise it misses the objective and its energy is lost.
   - **Background glass.** A ray travelling back to z = 0 leaves the film and is lost.
5. **Imaging.** A collected ray is imaged at the point where its last straight segment in the oil crosses the focal plane: `x_img = o_x + d_x (z_f − o_z)/d_z`. Its weight is splatted into that pixel. This is ideal geometric imaging of the plane z = z_f, so defocus, lens action and caustics come out of the geometry.

Rays are only traced in padded patches around objects. Objects whose padded boxes overlap are merged into one cluster and rendered together, so they interact optically. Each patch is parallelised across CPU threads with private accumulation buffers.

## 3. Fresnel equations

For incidence cosine cos θ_i going from index n₁ to n₂, Snell's law gives `sin θ_t = (n₁/n₂) sin θ_i`. The reflectances are:

```
r_s = (n₁ cos θ_i − n₂ cos θ_t) / (n₁ cos θ_i + n₂ cos θ_t)
r_p = (n₁ cos θ_t − n₂ cos θ_i) / (n₁ cos θ_t + n₂ cos θ_i)
R   = (r_s² + r_p²) / 2          # unpolarised light
T   = 1 − R
```

If `(n₁/n₂)² (1 − cos² θ_i) ≥ 1`, the light is **totally internally reflected**: R = 1 and there is no transmitted branch. The refracted direction is `t = η d + (η cos θ_i − cos θ_t) n` with `η = n₁/n₂`, where n is the surface normal facing the incoming ray.

For oil (1.47) to air (1.00), the critical angle is **θ_c = asin(1/1.47) = 42.86°**.

## 4. Why bubbles look the way they do

Take a vertical ray hitting a bubble of radius R at impact parameter b. The incidence angle is θ_i = asin(b/R).

| Zone | Impact parameter | What happens | Appearance |
|---|---|---|---|
| Centre | b ≪ R | Near-normal incidence, small Fresnel loss. The air sphere acts as a strong *diverging* lens, but rays near the axis are only slightly deviated and stay inside the objective NA | **Bright centre spot** |
| Mid annulus | b/R up to sin θ_c = 0.68 | Refracted rays are strongly deviated and leave the collection cone | Dark |
| Outer annulus | 0.68 < b/R < ~0.99 | **Total internal reflection.** Rays are reflected out of the collection cone (deviation 180° − 2θ_i) | **Thick dark ring** |
| Rim | b → R (grazing) | Still total reflection, but the deviation 180° − 2θ_i → 0. These rays carry their full energy, stay inside the NA and are imaged just at the bubble edge. Background light that grazed past also arrives there | **Edge glow**: a thin rim brighter than the background |

None of these features is hand-made. They follow from n_bubble < n_liq, and the test suite asserts all three. The rim is brighter than the surrounding liquid (T > 1) because energy is redistributed. An opaque particle can only absorb, so its transmission never exceeds 1. That difference is the physical cue a counter uses to reject bubbles.

## 5. Defocus, PSF and the reference square

- **Defocus** comes from the illumination cone. A flake at distance Δz from the focal plane blocks a disk of radius `Δz · tan θ_max` in the image. For the analytic reference-square rendering, this disk is approximated by a Gaussian with σ ≈ radius / 2. At the default settings that is 4.1 µm at the bottom glass.
- **The reference square** sits at z = 0. Its outline is computed analytically as a band between two concentric squares in the square's rotated frame. The band is convolved with the defocus Gaussian combined with the pixel box (σ² = σ_defocus² + p²/12), using `erf`. Its chrome transmission is 0.03. The outer edge measures exactly 10.000 mm.
- **Diffraction** is modelled as a Gaussian approximation of the Airy disk, **σ = 0.21 λ / NA_obj**, applied to the whole irradiance image. With λ = 550 nm and NA 0.25 this gives σ = 0.46 µm. The lateral resolution is 0.61 λ/NA = 1.34 µm and the depth of field is λ n / NA² = 12.9 µm.
- **Motion blur** (optional): each object patch is convolved with a line kernel of length v(z) · t_exposure along the flow direction.

## 6. Illumination and sensor

```
E(x, y)   = L₀ (1 − v·r²) (1 + g·tilt(x, y)) · T_square(x, y) · T_objects(x, y)
electrons = Poisson(E · FW) + Normal(0, σ_read)
DN        = clip(round(electrons / FW · (2^bits − 1)))
```

- L₀ = 0.72 of full scale, v = 0.10 vignetting and g = 0.04 tilt in a random direction.
- FW = 12 000 e⁻ full well and σ_read = 6 e⁻.

## 7. Flow

Pressure-driven laminar flow between parallel plates follows the plane Poiseuille profile:

```
v(z) = 4 v_max · z (h − z) / h²
```

v is zero at both plates and v_max at mid-film. In sequences, every object moves by v(z_obj)·Δt along the flow direction each frame, wrapping around the field. Object ids stay the same across frames, so the ground truth can be used directly to evaluate a tracker.

## 8. ISO 4406 and the size distribution

ISO 4406 reports three scale numbers for the cumulative counts per mL of particles larger than 4, 6 and 14 µm(c). A scale number covers "more than / up to and including" ranges that double at each step (see `iso4406.py`). For example, 17 means 640 to 1300 per mL and 19 means 2500 to 5000 per mL.

The target code becomes a particle population in four steps:

1. Each scale number is replaced by the geometric centre of its range, giving N₄, N₆ and N₁₄ per mL. For 19/17/14 these are 3535.5, 912.1 and 113.1.
2. The cumulative distribution N(> d) is taken as **piecewise power-law** (linear in log–log) through (4, N₄), (6, N₆) and (14, N₁₄). Above 14 µm the 6→14 µm slope is extended up to `particle_size_max_um`.
3. The number of particles in the full field is drawn from Poisson(N₄ · V_fov). Sizes are drawn by inverting the normalised cumulative distribution.
4. Each particle's outline is a random star polygon scaled so that its **maximum Feret diameter equals the drawn size**, following ISO 4407's longest-dimension convention.

The ground-truth code counts only solid particles whose centre lies inside the interior of the reference square. It divides by the counting volume, which is the interior area times the film thickness:

```
(10 mm − 2 × 60 µm)² × 200 µm = 0.01952 mL
```

Because of Poisson sampling, an individual image can land one code step away from the target. The ground-truth record always reports the code that was actually realised.
