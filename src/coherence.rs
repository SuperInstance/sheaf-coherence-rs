//! Coherence measure — how well local sections agree on overlaps.

use crate::sheaf::{Assignment, RestrictionMap, Sheaf};

/// Compute the disagreement for a single restriction map.
/// Measures ||F_map(x_source) - x_target||^2.
pub fn map_disagreement(map: &RestrictionMap, source: &Assignment, target: &Assignment) -> f64 {
    let projected = map.apply(&source.data);
    projected
        .iter()
        .zip(target.data.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum()
}

/// Compute total coherence energy of a sheaf assignment.
/// Lower = more coherent. Zero = perfect global section.
pub fn coherence_energy(sheaf: &Sheaf) -> f64 {
    let mut energy = 0.0;
    for map in &sheaf.restriction_maps {
        if let (Some(src), Some(tgt)) = (
            sheaf.assignments.get(&map.source),
            sheaf.assignments.get(&map.target),
        ) {
            energy += map_disagreement(map, src, tgt);
        }
    }
    energy
}

/// Normalize coherence energy by the number of restriction maps with assignments.
pub fn normalized_coherence_energy(sheaf: &Sheaf) -> f64 {
    let mut energy = 0.0;
    let mut count = 0;
    for map in &sheaf.restriction_maps {
        if let (Some(src), Some(tgt)) = (
            sheaf.assignments.get(&map.source),
            sheaf.assignments.get(&map.target),
        ) {
            energy += map_disagreement(map, src, tgt);
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        energy / count as f64
    }
}

/// Coherence score between 0.0 (incoherent) and 1.0 (perfectly coherent).
/// Uses exp(-energy) mapping.
pub fn coherence_score(sheaf: &Sheaf) -> f64 {
    let energy = normalized_coherence_energy(sheaf);
    (-energy).exp()
}

/// Check if the assignment is a global section (perfect coherence within tolerance).
pub fn is_global_section(sheaf: &Sheaf, tolerance: f64) -> bool {
    coherence_energy(sheaf) <= tolerance
}

/// Per-cell contribution to incoherence.
/// Returns a map from cell_id to the sum of disagreements on incident maps.
pub fn cell_incoherence(sheaf: &Sheaf) -> std::collections::HashMap<usize, f64> {
    let mut contributions = std::collections::HashMap::new();
    for map in &sheaf.restriction_maps {
        if let (Some(src), Some(tgt)) = (
            sheaf.assignments.get(&map.source),
            sheaf.assignments.get(&map.target),
        ) {
            let d = map_disagreement(map, src, tgt);
            *contributions.entry(map.source).or_insert(0.0) += d;
            *contributions.entry(map.target).or_insert(0.0) += d;
        }
    }
    contributions
}

/// Find the most incoherent cell.
pub fn most_incoherent_cell(sheaf: &Sheaf) -> Option<usize> {
    let contributions = cell_incoherence(sheaf);
    contributions
        .into_iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(id, _)| id)
}

/// Pairwise disagreement between two cells connected by a map.
pub fn pairwise_disagreement(
    sheaf: &Sheaf,
    source_id: usize,
    target_id: usize,
) -> Option<f64> {
    for map in &sheaf.restriction_maps {
        if map.source == source_id && map.target == target_id {
            if let (Some(src), Some(tgt)) = (
                sheaf.assignments.get(&source_id),
                sheaf.assignments.get(&target_id),
            ) {
                return Some(map_disagreement(map, src, tgt));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sheaf::line_sheaf;

    #[test]
    fn test_map_disagreement_zero() {
        let map = RestrictionMap::new(0, 1, vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        let src = Assignment::new(0, vec![1.0, 2.0]);
        let tgt = Assignment::new(1, vec![1.0, 2.0]);
        assert_eq!(map_disagreement(&map, &src, &tgt), 0.0);
    }

    #[test]
    fn test_map_disagreement_nonzero() {
        let map = RestrictionMap::new(0, 1, vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        let src = Assignment::new(0, vec![1.0, 2.0]);
        let tgt = Assignment::new(1, vec![3.0, 4.0]);
        // (1-3)^2 + (2-4)^2 = 4 + 4 = 8
        assert_eq!(map_disagreement(&map, &src, &tgt), 8.0);
    }

    #[test]
    fn test_coherence_energy_perfect() {
        let mut s = line_sheaf(2);
        s.assign(Assignment::new(0, vec![1.0, 2.0]));
        s.assign(Assignment::new(1, vec![1.0, 2.0]));
        s.assign(Assignment::new(2, vec![1.0, 2.0]));
        assert_eq!(coherence_energy(&s), 0.0);
    }

    #[test]
    fn test_coherence_energy_nonzero() {
        let mut s = line_sheaf(2);
        s.assign(Assignment::new(0, vec![1.0, 0.0]));
        s.assign(Assignment::new(1, vec![0.0, 1.0]));
        s.assign(Assignment::new(2, vec![1.0, 1.0]));
        let e = coherence_energy(&s);
        assert!(e > 0.0);
    }

    #[test]
    fn test_coherence_score_perfect() {
        let mut s = line_sheaf(2);
        s.assign(Assignment::new(0, vec![5.0, 3.0]));
        s.assign(Assignment::new(1, vec![5.0, 3.0]));
        s.assign(Assignment::new(2, vec![5.0, 3.0]));
        let score = coherence_score(&s);
        assert!((score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_is_global_section() {
        let mut s = line_sheaf(2);
        s.assign(Assignment::new(0, vec![1.0, 2.0]));
        s.assign(Assignment::new(1, vec![1.0, 2.0]));
        s.assign(Assignment::new(2, vec![1.0, 2.0]));
        assert!(is_global_section(&s, 1e-10));
    }

    #[test]
    fn test_is_not_global_section() {
        let mut s = line_sheaf(2);
        s.assign(Assignment::new(0, vec![1.0, 0.0]));
        s.assign(Assignment::new(1, vec![0.0, 1.0]));
        s.assign(Assignment::new(2, vec![0.0, 0.0]));
        assert!(!is_global_section(&s, 0.1));
    }

    #[test]
    fn test_cell_incoherence() {
        let mut s = line_sheaf(1);
        s.assign(Assignment::new(0, vec![0.0]));
        s.assign(Assignment::new(1, vec![1.0]));
        s.assign(Assignment::new(2, vec![0.5]));
        let ci = cell_incoherence(&s);
        assert!(ci.contains_key(&0));
        assert!(ci.contains_key(&1));
        assert!(ci.contains_key(&2));
    }

    #[test]
    fn test_most_incoherent_cell() {
        let mut s = line_sheaf(1);
        s.assign(Assignment::new(0, vec![0.0]));
        s.assign(Assignment::new(1, vec![100.0]));
        s.assign(Assignment::new(2, vec![0.5]));
        let cell = most_incoherent_cell(&s);
        assert!(cell.is_some());
        // Cell 2 (edge) is incident to both maps, so it gets the most incoherence
        let id = cell.unwrap();
        assert!(id == 1 || id == 2); // either vertex 1 or edge 2
    }

    #[test]
    fn test_pairwise_disagreement() {
        let mut s = line_sheaf(2);
        s.assign(Assignment::new(0, vec![1.0, 0.0]));
        s.assign(Assignment::new(1, vec![0.0, 1.0]));
        s.assign(Assignment::new(2, vec![1.0, 1.0]));
        // Map from edge(2) -> vertex(0): projected [1,1], target [1,0] → disagreement = 1.0
        let d = pairwise_disagreement(&s, 2, 0);
        assert!(d.is_some());
        assert!((d.unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalized_energy_empty() {
        let s = Sheaf::new();
        assert_eq!(normalized_coherence_energy(&s), 0.0);
    }
}
