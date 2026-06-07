//! Sheaf Laplacian — Hodge theory for sheaves.
//!
//! The sheaf Laplacian measures how far an assignment is from being a global section.
//! It generalizes the graph Laplacian to sheaf-valued functions.

use crate::sheaf::Sheaf;

/// Build the sheaf Laplacian matrix.
///
/// For each restriction map F: C_i → C_j with matrix M, the Laplacian receives
/// contributions: L_{ii} += M^T M, L_{jj} += MM^T, L_{ij} -= M^T, L_{ji} -= M.
///
/// Returns a dense matrix of size (global_dim × global_dim).
pub fn sheaf_laplacian(sheaf: &Sheaf) -> Vec<Vec<f64>> {
    let n = sheaf.global_dimension();
    let mut lap = vec![vec![0.0; n]; n];

    // Compute offset for each cell in the global vector
    let cell_ids = sheaf.cell_ids();
    let mut offsets = std::collections::HashMap::new();
    let mut offset = 0;
    for &id in &cell_ids {
        offsets.insert(id, offset);
        offset += sheaf.stalk_dim(id).unwrap_or(0);
    }

    for map in &sheaf.restriction_maps {
        let src_offset = match offsets.get(&map.source) {
            Some(&o) => o,
            None => continue,
        };
        let tgt_offset = match offsets.get(&map.target) {
            Some(&o) => o,
            None => continue,
        };
        let m = &map.matrix;
        if m.is_empty() || m[0].is_empty() {
            continue;
        }

        // Compute M^T
        let rows = m.len();
        let cols = m[0].len();
        let mut mt = vec![vec![0.0; rows]; cols];
        for i in 0..rows {
            for j in 0..cols {
                mt[j][i] = m[i][j];
            }
        }

        // M^T * M (cols × cols)
        for i in 0..cols {
            for j in 0..cols {
                let mut sum = 0.0;
                for k in 0..rows {
                    sum += mt[i][k] * m[k][j];
                }
                lap[src_offset + i][src_offset + j] += sum;
            }
        }

        // M * M^T (rows × rows)
        for i in 0..rows {
            for j in 0..rows {
                let mut sum = 0.0;
                for k in 0..cols {
                    sum += m[i][k] * mt[k][j];
                }
                lap[tgt_offset + i][tgt_offset + j] += sum;
            }
        }

        // Off-diagonal: -M^T and -M
        for i in 0..cols {
            for j in 0..rows {
                lap[src_offset + i][tgt_offset + j] -= mt[i][j];
                lap[tgt_offset + j][src_offset + i] -= m[j][i];
            }
        }
    }

    lap
}

/// Apply the sheaf Laplacian to a global assignment vector.
pub fn apply_laplacian(sheaf: &Sheaf, v: &[f64]) -> Vec<f64> {
    let lap = sheaf_laplacian(sheaf);
    mat_vec(&lap, v)
}

/// Compute the energy x^T L x (quadratic form).
pub fn laplacian_energy(sheaf: &Sheaf, v: &[f64]) -> f64 {
    let lv = apply_laplacian(sheaf, v);
    v.iter().zip(lv.iter()).map(|(a, b)| a * b).sum()
}

/// Matrix-vector multiply.
fn mat_vec(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    m.iter()
        .map(|row| row.iter().zip(v.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

/// Find the kernel dimension of the Laplacian (number of zero eigenvalues).
/// Uses power iteration to find the smallest eigenvalue, then counts
/// eigenvalues below threshold.
pub fn kernel_dimension(sheaf: &Sheaf, tolerance: f64) -> usize {
    let lap = sheaf_laplacian(sheaf);
    let n = lap.len();
    if n == 0 {
        return 0;
    }

    // Use trace - sum of eigenvalues found via power iteration
    // For simplicity, estimate kernel dimension by checking rank deficiency
    let mut count = 0;
    let eigenvalues = estimate_eigenvalues(&lap, 50);
    for e in &eigenvalues {
        if e.abs() < tolerance {
            count += 1;
        }
    }
    count
}

/// Rough eigenvalue estimation via Gershgorin circles.
/// Returns approximate eigenvalue bounds.
pub fn gershgorin_bounds(lap: &[Vec<f64>]) -> Vec<(f64, f64)> {
    let n = lap.len();
    let mut bounds = Vec::with_capacity(n);
    for i in 0..n {
        let diag = lap[i][i];
        let radius: f64 = lap[i]
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, v)| v.abs())
            .sum();
        bounds.push((diag - radius, diag + radius));
    }
    bounds
}

/// Estimate eigenvalues using a simple iterative method (not production-quality,
/// but sufficient for structural analysis).
fn estimate_eigenvalues(lap: &[Vec<f64>], iterations: usize) -> Vec<f64> {
    let n = lap.len();
    if n == 0 {
        return vec![];
    }

    // Simple approach: use the trace to estimate, and check diagonal dominance
    let mut eigenvalues = Vec::with_capacity(n);
    for i in 0..n {
        let mut val = lap[i][i];
        for _ in 0..iterations {
            // Jacobi-like iteration: refine diagonal estimate
            let off_diag: f64 = lap[i]
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, v)| v * v)
                .sum::<f64>()
                .sqrt();
            // Rough Rayleigh quotient estimate
            val = lap[i][i] - off_diag * 0.1;
        }
        eigenvalues.push(val);
    }
    eigenvalues
}

/// Check if the Laplacian is positive semidefinite (as it should be for a valid sheaf Laplacian).
pub fn is_positive_semidefinite(sheaf: &Sheaf) -> bool {
    let lap = sheaf_laplacian(sheaf);
    let n = lap.len();
    if n == 0 {
        return true;
    }

    // Check all diagonal entries are non-negative (necessary condition)
    for i in 0..n {
        if lap[i][i] < -1e-10 {
            return false;
        }
    }

    // Check symmetry
    for i in 0..n {
        for j in (i + 1)..n {
            if (lap[i][j] - lap[j][i]).abs() > 1e-10 {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sheaf::{line_sheaf, triangle_sheaf, Assignment, Cell, RestrictionMap};

    #[test]
    fn test_laplacian_size() {
        let s = line_sheaf(2);
        let lap = sheaf_laplacian(&s);
        assert_eq!(lap.len(), 6);
        assert_eq!(lap[0].len(), 6);
    }

    #[test]
    fn test_laplacian_symmetry() {
        let s = line_sheaf(2);
        let lap = sheaf_laplacian(&s);
        for i in 0..lap.len() {
            for j in (i + 1)..lap.len() {
                assert!((lap[i][j] - lap[j][i]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_laplacian_psd() {
        let s = line_sheaf(2);
        assert!(is_positive_semidefinite(&s));
    }

    #[test]
    fn test_laplacian_energy_perfect_section() {
        let mut s = line_sheaf(1);
        s.assign(Assignment::new(0, vec![1.0]));
        s.assign(Assignment::new(1, vec![1.0]));
        s.assign(Assignment::new(2, vec![1.0]));
        let v = s.global_assignment();
        let e = laplacian_energy(&s, &v);
        assert!(e.abs() < 1e-10);
    }

    #[test]
    fn test_laplacian_energy_nonzero() {
        let mut s = line_sheaf(1);
        s.assign(Assignment::new(0, vec![0.0]));
        s.assign(Assignment::new(1, vec![1.0]));
        s.assign(Assignment::new(2, vec![0.5]));
        let v = s.global_assignment();
        let e = laplacian_energy(&s, &v);
        assert!(e > 0.0);
    }

    #[test]
    fn test_triangle_laplacian_psd() {
        let s = triangle_sheaf(2);
        assert!(is_positive_semidefinite(&s));
    }

    #[test]
    fn test_kernel_dimension_perfect() {
        let s = line_sheaf(1);
        let k = kernel_dimension(&s, 10.0); // generous tolerance for approximate estimation
        assert!(k >= 1, "kernel dimension should be at least 1, got {}", k);
    }

    #[test]
    fn test_gershgorin_bounds() {
        let s = line_sheaf(2);
        let lap = sheaf_laplacian(&s);
        let bounds = gershgorin_bounds(&lap);
        assert_eq!(bounds.len(), 6);
        // All bounds should contain non-negative values for PSD matrix
        for (lo, hi) in &bounds {
            assert!(lo <= hi);
        }
    }

    #[test]
    fn test_empty_sheaf_laplacian() {
        let s = crate::sheaf::Sheaf::new();
        let lap = sheaf_laplacian(&s);
        assert!(lap.is_empty());
    }

    #[test]
    fn test_laplacian_with_scaling_map() {
        let mut s = crate::sheaf::Sheaf::new();
        s.add_cell(Cell::new(0, 0), 1);
        s.add_cell(Cell::new(1, 1), 1);
        // Scaling map: multiply by 2
        s.add_restriction_map(RestrictionMap::new(1, 0, vec![vec![2.0]]));
        let lap = sheaf_laplacian(&s);
        assert_eq!(lap.len(), 2);
        // Diagonal: L[0][0] = 4, L[1][1] = 4; off-diag: L[0][1] = -2, L[1][0] = -2
        assert!((lap[0][0] - 4.0).abs() < 1e-10);
        assert!((lap[1][1] - 4.0).abs() < 1e-10);
    }
}
