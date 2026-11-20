//! `ID2D1Geometry::Tessellate`: the shape's filled figures scanned on the tessellator's 64 grid, the
//! fill region cut into bands and each band triangulated by native's `CTessellator`
//! ([`super::combine::tess`]), the triangles sent to the sink in device f32 with the vertex order native
//! emits them in.

use super::{Geometry, Matrix, Point};

/// One emitted triangle, device-space f32 (the three `D2D1_TRIANGLE` points).
pub type Tri = [Point; 3];

/// Tessellate `geom` under `transform` at `tolerance` into filling triangles, as
/// `ID2D1Geometry::Tessellate` emits them to an `ID2D1TessellationSink`.
pub fn tessellate(geom: &Geometry, transform: &Matrix, tolerance: f32) -> Vec<Tri> {
    super::combine::tessellate(geom, transform, tolerance)
}

#[cfg(test)]
fn tri_area(tris: &[Tri]) -> f64 {
    let area = |a: Point, b: Point, c: Point| ((b[0] - a[0]) as f64 * (c[1] - a[1]) as f64 - (c[0] - a[0]) as f64 * (b[1] - a[1]) as f64) * 0.5;
    tris.iter().map(|t| area(t[0], t[1], t[2]).abs()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{FillMode, IDENTITY};

    // The triangles must tile the filled region: their total area equals the shape's area, within
    // the 1/64 workspace-grid snap the tessellator applies to curves (≈0.3 for this ellipse; the
    // straight shapes are exact). Byte-exact vertex/order parity is checked by the tess-* scenes.
    #[test]
    fn triangles_tile_the_fill_area() {
        for g in [
            Geometry::Rect([20.0, 20.0, 100.0, 70.0]),
            Geometry::Ellipse { center: [60.0, 45.0], radius: [40.0, 28.0] },
            Geometry::Path {
                fill: FillMode::Winding,
                figures: vec![super::super::Figure {
                    start: [60.0, 15.0],
                    filled: true,
                    closed: true,
                    segments: vec![super::super::Segment::Line([100.0, 75.0]), super::super::Segment::Line([20.0, 75.0])],
                    seg_flags: Vec::new(),
                }],
            },
        ] {
            let tris = tessellate(&g, &IDENTITY, 0.25);
            let want = g.compute_area(&IDENTITY, 0.25) as f64;
            assert!((tri_area(&tris) - want).abs() < 0.5, "tiled area {} vs {want}", tri_area(&tris));
        }
    }
}
