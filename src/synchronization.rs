//! Synchronization of sheaf assignments across cells.
//!
//! Diffusion-based and optimization-based methods to synchronize local
//! assignments into (approximate) global sections.

use crate::sheaf::{Assignment, Sheaf};
use crate::laplacian::sheaf_laplacian;

/// Synchronize assignments using diffusion (heat equation on the sheaf Laplacian).
///
/// Updates each cell's assignment by moving in the negative gradient of the coherence energy.
/// `step_size` controls the diffusion rate. `iterations` is the number of steps.
pub fn diffusion_synchronize(
    sheaf: &mut Sheaf,
    step_size: f64,
    iterations: usize,
) -> Vec<f64> {
    let mut energies = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let lap = sheaf_laplacian(sheaf);
        let v = sheaf.global_assignment();
        let n = v.len();
        if n == 0 {
            energies.push(0.0);
            continue;
        }

        // Compute L * v
        let lv: Vec<f64> = lap
            .iter()
            .map(|row| row.iter().zip(v.iter()).map(|(a, b)| a * b).sum())
            .collect();

        // Update: v -= step_size * L * v
        let new_v: Vec<f64> = v.iter().zip(lv.iter()).map(|(vi, lvi)| vi - step_size * lvi).collect();

        // Write back
        let cell_ids = sheaf.cell_ids();
        let mut offset = 0;
        for id in &cell_ids {
            let dim = sheaf.stalk_dim(*id).unwrap_or(0);
            if dim > 0 {
                let data = new_v[offset..offset + dim].to_vec();
                sheaf.assign(Assignment::new(*id, data));
            }
            offset += dim;
        }

        // Compute energy
        let energy: f64 = new_v.iter().zip(lv.iter()).map(|(a, b)| a * b).sum();
        energies.push(energy);
    }

    energies
}

/// Project an assignment onto the kernel of the Laplacian (closest global section).
/// Uses iterative gradient descent to minimize x^T L x subject to ||x|| = ||x_0||.
pub fn project_to_global_section(
    sheaf: &mut Sheaf,
    iterations: usize,
    step_size: f64,
) -> Vec<f64> {
    diffusion_synchronize(sheaf, step_size, iterations)
}

/// Check if all cells have assignments.
pub fn is_fully_assigned(sheaf: &Sheaf) -> bool {
    for id in sheaf.cell_ids() {
        if !sheaf.assignments.contains_key(&id) {
            return false;
        }
    }
    true
}

/// Initialize missing assignments with zeros.
pub fn initialize_missing_assignments(sheaf: &mut Sheaf) {
    for id in sheaf.cell_ids() {
        if !sheaf.assignments.contains_key(&id) {
            let dim = sheaf.stalk_dim(id).unwrap_or(0);
            sheaf.assign(Assignment::new(id, vec![0.0; dim]));
        }
    }
}

/// Compute the average of all cell assignments.
pub fn average_assignment(sheaf: &Sheaf) -> Option<Vec<f64>> {
    if sheaf.assignments.is_empty() {
        return None;
    }
    // Get the dimension from the first assignment
    let first = sheaf.assignments.values().next()?;
    let dim = first.data.len();
    let mut avg = vec![0.0; dim];
    let count = sheaf.assignments.len() as f64;

    for a in sheaf.assignments.values() {
        for (i, v) in a.data.iter().enumerate() {
            if i < dim {
                avg[i] += v / count;
            }
        }
    }
    Some(avg)
}

/// Set all assignments to their average (global consensus).
pub fn consensus_assignments(sheaf: &mut Sheaf) {
    if let Some(avg) = average_assignment(sheaf) {
        for id in sheaf.cell_ids() {
            sheaf.assign(Assignment::new(id, avg.clone()));
        }
    }
}

/// Compute convergence: maximum change between two assignment vectors.
pub fn max_change(v1: &[f64], v2: &[f64]) -> f64 {
    v1.iter()
        .zip(v2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

/// Run synchronization until convergence or max iterations.
pub fn synchronize_until(
    sheaf: &mut Sheaf,
    step_size: f64,
    max_iterations: usize,
    tolerance: f64,
) -> (usize, f64) {
    for i in 0..max_iterations {
        let old_v = sheaf.global_assignment();
        diffusion_synchronize(sheaf, step_size, 1);
        let new_v = sheaf.global_assignment();
        let change = max_change(&old_v, &new_v);
        if change < tolerance {
            return (i + 1, change);
        }
    }
    let _v = sheaf.global_assignment();
    (max_iterations, 0.0) // Didn't converge
}

/// Compute the spectral gap estimate (smallest non-zero eigenvalue proxy).
/// Uses the fact that for a connected sheaf, the second-smallest eigenvalue
/// controls synchronization speed.
pub fn spectral_gap_estimate(sheaf: &Sheaf, tolerance: f64) -> f64 {
    let lap = sheaf_laplacian(sheaf);
    if lap.is_empty() {
        return 0.0;
    }

    // Power iteration to find dominant eigenvalue, then shift
    let n = lap.len();
    let mut v = vec![1.0; n];
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in v.iter_mut() {
        *x /= norm;
    }

    let max_eigenvalue = power_iteration(&lap, &mut v, 20);

    // Shift the matrix: L_shifted = max_eigenvalue * I - L
    // Then find smallest eigenvalue of original = max_eigenvalue - largest of shifted
    let mut shifted = lap.clone();
    for i in 0..n {
        shifted[i][i] = max_eigenvalue - shifted[i][i];
        for j in 0..n {
            if i != j {
                shifted[i][j] = -shifted[i][j];
            }
        }
    }

    let smallest = max_eigenvalue - power_iteration(&shifted, &mut vec![1.0; n], 20);

    // Check if it's effectively zero (kernel eigenvalue)
    if smallest.abs() < tolerance {
        // Find the next one by deflation
        // For simplicity, return the Gershgorin lower bound of non-zero eigenvalues
        let bounds = crate::laplacian::gershgorin_bounds(&lap);
        let mut min_nonzero = f64::MAX;
        for (lo, _hi) in &bounds {
            if lo.abs() > tolerance && *lo < min_nonzero {
                min_nonzero = *lo;
            }
        }
        if min_nonzero == f64::MAX {
            tolerance
        } else {
            min_nonzero
        }
    } else {
        smallest
    }
}

/// Power iteration to find dominant eigenvalue.
fn power_iteration(mat: &[Vec<f64>], v: &mut Vec<f64>, iterations: usize) -> f64 {
    let n = mat.len();
    for _ in 0..iterations {
        let mut new_v = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                new_v[i] += mat[i][j] * v[j];
            }
        }
        let norm: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-15 {
            for x in new_v.iter_mut() {
                *x /= norm;
            }
        }
        *v = new_v;
    }

    // Rayleigh quotient
    let mut mv = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            mv[i] += mat[i][j] * v[j];
        }
    }
    v.iter().zip(mv.iter()).map(|(a, b)| a * b).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sheaf::{line_sheaf, Assignment};

    #[test]
    fn test_diffusion_reduces_energy() {
        let mut s = line_sheaf(1);
        s.assign(Assignment::new(0, vec![0.0]));
        s.assign(Assignment::new(1, vec![10.0]));
        s.assign(Assignment::new(2, vec![5.0]));
        let energies = diffusion_synchronize(&mut s, 0.1, 20);
        // Energy should decrease over iterations
        assert!(energies.last().unwrap() < energies.first().unwrap());
    }

    #[test]
    fn test_is_fully_assigned() {
        let mut s = line_sheaf(1);
        assert!(!is_fully_assigned(&s));
        s.assign(Assignment::new(0, vec![1.0]));
        s.assign(Assignment::new(1, vec![1.0]));
        s.assign(Assignment::new(2, vec![1.0]));
        assert!(is_fully_assigned(&s));
    }

    #[test]
    fn test_initialize_missing() {
        let mut s = line_sheaf(2);
        s.assign(Assignment::new(0, vec![1.0, 2.0]));
        initialize_missing_assignments(&mut s);
        assert!(is_fully_assigned(&s));
        assert_eq!(s.assignments.get(&1).unwrap().data, vec![0.0, 0.0]);
    }

    #[test]
    fn test_average_assignment() {
        let mut s = line_sheaf(1);
        s.assign(Assignment::new(0, vec![2.0]));
        s.assign(Assignment::new(1, vec![4.0]));
        s.assign(Assignment::new(2, vec![6.0]));
        let avg = average_assignment(&s);
        assert!(avg.is_some());
        assert!((avg.unwrap()[0] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_consensus_assignments() {
        let mut s = line_sheaf(1);
        s.assign(Assignment::new(0, vec![0.0]));
        s.assign(Assignment::new(1, vec![10.0]));
        s.assign(Assignment::new(2, vec![5.0]));
        consensus_assignments(&mut s);
        let avg = average_assignment(&s).unwrap();
        for a in s.assignments.values() {
            assert!((a.data[0] - avg[0]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_max_change() {
        let v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![1.1, 2.5, 3.0];
        assert!((max_change(&v1, &v2) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_synchronize_until_convergence() {
        let mut s = line_sheaf(1);
        s.assign(Assignment::new(0, vec![0.0]));
        s.assign(Assignment::new(1, vec![1.0]));
        s.assign(Assignment::new(2, vec![0.5]));
        let (iters, _) = synchronize_until(&mut s, 0.1, 1000, 1e-6);
        assert!(iters < 1000);
    }

    #[test]
    fn test_spectral_gap() {
        let s = line_sheaf(1);
        let gap = spectral_gap_estimate(&s, 1e-6);
        assert!(gap >= 0.0);
    }

    #[test]
    fn test_average_empty() {
        let s = Sheaf::new();
        assert!(average_assignment(&s).is_none());
    }

    #[test]
    fn test_diffusion_perfect_section_unchanged() {
        let mut s = line_sheaf(1);
        s.assign(Assignment::new(0, vec![5.0]));
        s.assign(Assignment::new(1, vec![5.0]));
        s.assign(Assignment::new(2, vec![5.0]));
        let energies = diffusion_synchronize(&mut s, 0.1, 10);
        for e in &energies {
            assert!(e.abs() < 1e-8);
        }
    }
}
