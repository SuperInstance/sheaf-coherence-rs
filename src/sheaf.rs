//! Cellular sheaf data structure with assignments and restriction maps.

use std::collections::HashMap;

/// A cell in the cellular complex, identified by an index and dimension.
#[derive(Debug, Clone, PartialEq)]
pub struct Cell {
    pub id: usize,
    pub dimension: usize,
}

impl Cell {
    pub fn new(id: usize, dimension: usize) -> Self {
        Self { id, dimension }
    }
}

/// A restriction map between two cells, represented as a matrix (row-major).
#[derive(Debug, Clone)]
pub struct RestrictionMap {
    pub source: usize,
    pub target: usize,
    pub matrix: Vec<Vec<f64>>,
}

impl RestrictionMap {
    pub fn new(source: usize, target: usize, matrix: Vec<Vec<f64>>) -> Self {
        Self { source, target, matrix }
    }

    /// Apply the restriction map to a vector.
    pub fn apply(&self, v: &[f64]) -> Vec<f64> {
        self.matrix
            .iter()
            .map(|row| {
                row.iter().zip(v.iter()).map(|(a, b)| a * b).sum()
            })
            .collect()
    }

    /// Transpose of the restriction map.
    pub fn transpose(&self) -> RestrictionMap {
        if self.matrix.is_empty() {
            return RestrictionMap::new(self.target, self.source, vec![]);
        }
        let rows = self.matrix.len();
        let cols = self.matrix[0].len();
        let mut t = vec![vec![0.0; rows]; cols];
        for i in 0..rows {
            for j in 0..cols {
                t[j][i] = self.matrix[i][j];
            }
        }
        RestrictionMap::new(self.target, self.source, t)
    }

    /// Number of output dimensions (rows).
    pub fn output_dim(&self) -> usize {
        self.matrix.len()
    }

    /// Number of input dimensions (columns).
    pub fn input_dim(&self) -> usize {
        self.matrix.first().map_or(0, |r| r.len())
    }
}

/// Assignment of data to a cell.
#[derive(Debug, Clone)]
pub struct Assignment {
    pub cell_id: usize,
    pub data: Vec<f64>,
}

impl Assignment {
    pub fn new(cell_id: usize, data: Vec<f64>) -> Self {
        Self { cell_id, data }
    }

    /// Dimension of the assigned data.
    pub fn dimension(&self) -> usize {
        self.data.len()
    }
}

/// A cellular sheaf: assigns vector spaces to cells and linear maps to incidences.
#[derive(Debug, Clone)]
pub struct Sheaf {
    pub cells: HashMap<usize, Cell>,
    pub stalk_dimensions: HashMap<usize, usize>,
    pub restriction_maps: Vec<RestrictionMap>,
    pub assignments: HashMap<usize, Assignment>,
}

impl Sheaf {
    pub fn new() -> Self {
        Self {
            cells: HashMap::new(),
            stalk_dimensions: HashMap::new(),
            restriction_maps: Vec::new(),
            assignments: HashMap::new(),
        }
    }

    /// Add a cell to the sheaf.
    pub fn add_cell(&mut self, cell: Cell, stalk_dimension: usize) {
        self.stalk_dimensions.insert(cell.id, stalk_dimension);
        self.cells.insert(cell.id, cell);
    }

    /// Add a restriction map between two cells.
    pub fn add_restriction_map(&mut self, map: RestrictionMap) {
        self.restriction_maps.push(map);
    }

    /// Assign data to a cell.
    pub fn assign(&mut self, assignment: Assignment) {
        self.assignments.insert(assignment.cell_id, assignment);
    }

    /// Get the stalk dimension for a cell.
    pub fn stalk_dim(&self, cell_id: usize) -> Option<usize> {
        self.stalk_dimensions.get(&cell_id).copied()
    }

    /// Get all cell IDs.
    pub fn cell_ids(&self) -> Vec<usize> {
        let mut ids: Vec<usize> = self.cells.keys().copied().collect();
        ids.sort();
        ids
    }

    /// Total dimension of the global section space.
    pub fn global_dimension(&self) -> usize {
        self.stalk_dimensions.values().sum()
    }

    /// Get restriction maps involving a specific cell as source.
    pub fn maps_from(&self, cell_id: usize) -> Vec<&RestrictionMap> {
        self.restriction_maps
            .iter()
            .filter(|m| m.source == cell_id)
            .collect()
    }

    /// Get restriction maps involving a specific cell as target.
    pub fn maps_to(&self, cell_id: usize) -> Vec<&RestrictionMap> {
        self.restriction_maps
            .iter()
            .filter(|m| m.target == cell_id)
            .collect()
    }

    /// Build the global assignment vector (concatenation of all cell assignments in order).
    pub fn global_assignment(&self) -> Vec<f64> {
        let mut v = Vec::new();
        for id in self.cell_ids() {
            if let Some(a) = self.assignments.get(&id) {
                v.extend_from_slice(&a.data);
            } else if let Some(dim) = self.stalk_dimensions.get(&id) {
                v.extend(std::iter::repeat_n(0.0, *dim));
            }
        }
        v
    }

    /// Count restriction maps.
    pub fn num_maps(&self) -> usize {
        self.restriction_maps.len()
    }

    /// Count cells.
    pub fn num_cells(&self) -> usize {
        self.cells.len()
    }
}

impl Default for Sheaf {
    fn default() -> Self {
        Self::new()
    }
}

/// Build a simple line sheaf: 3 cells (vertices + edge) with identity restriction maps.
pub fn line_sheaf(dim: usize) -> Sheaf {
    let mut s = Sheaf::new();
    // Two vertices and one edge
    s.add_cell(Cell::new(0, 0), dim);
    s.add_cell(Cell::new(1, 0), dim);
    s.add_cell(Cell::new(2, 1), dim);

    let id_matrix: Vec<Vec<f64>> = (0..dim)
        .map(|i| {
            (0..dim)
                .map(|j| if i == j { 1.0 } else { 0.0 })
                .collect()
        })
        .collect();

    s.add_restriction_map(RestrictionMap::new(2, 0, id_matrix.clone()));
    s.add_restriction_map(RestrictionMap::new(2, 1, id_matrix));
    s
}

/// Build a triangle sheaf: 3 vertices + 3 edges + 1 face.
pub fn triangle_sheaf(dim: usize) -> Sheaf {
    let mut s = Sheaf::new();
    // Vertices 0,1,2 (dim 0); edges 3,4,5 (dim 1); face 6 (dim 2)
    for i in 0..3 {
        s.add_cell(Cell::new(i, 0), dim);
    }
    for i in 3..6 {
        s.add_cell(Cell::new(i, 1), dim);
    }
    s.add_cell(Cell::new(6, 2), dim);

    let id_matrix: Vec<Vec<f64>> = (0..dim)
        .map(|i| {
            (0..dim)
                .map(|j| if i == j { 1.0 } else { 0.0 })
                .collect()
        })
        .collect();

    // Edge 3 → vertices 0, 1
    // Edge 4 → vertices 1, 2
    // Edge 5 → vertices 0, 2
    s.add_restriction_map(RestrictionMap::new(3, 0, id_matrix.clone()));
    s.add_restriction_map(RestrictionMap::new(3, 1, id_matrix.clone()));
    s.add_restriction_map(RestrictionMap::new(4, 1, id_matrix.clone()));
    s.add_restriction_map(RestrictionMap::new(4, 2, id_matrix.clone()));
    s.add_restriction_map(RestrictionMap::new(5, 0, id_matrix.clone()));
    s.add_restriction_map(RestrictionMap::new(5, 2, id_matrix.clone()));
    // Face 6 → edges 3,4,5
    s.add_restriction_map(RestrictionMap::new(6, 3, id_matrix.clone()));
    s.add_restriction_map(RestrictionMap::new(6, 4, id_matrix.clone()));
    s.add_restriction_map(RestrictionMap::new(6, 5, id_matrix));
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cell_creation() {
        let c = Cell::new(3, 2);
        assert_eq!(c.id, 3);
        assert_eq!(c.dimension, 2);
    }

    #[test]
    fn test_restriction_map_apply() {
        let m = RestrictionMap::new(0, 1, vec![vec![1.0, 0.0], vec![0.0, 2.0]]);
        let result = m.apply(&[3.0, 4.0]);
        assert_eq!(result, vec![3.0, 8.0]);
    }

    #[test]
    fn test_restriction_map_transpose() {
        let m = RestrictionMap::new(0, 1, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let t = m.transpose();
        assert_eq!(t.matrix, vec![vec![1.0, 3.0], vec![2.0, 4.0]]);
        assert_eq!(t.source, 1);
        assert_eq!(t.target, 0);
    }

    #[test]
    fn test_restriction_map_dims() {
        let m = RestrictionMap::new(0, 1, vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]);
        assert_eq!(m.output_dim(), 2);
        assert_eq!(m.input_dim(), 3);
    }

    #[test]
    fn test_sheaf_add_cells() {
        let mut s = Sheaf::new();
        s.add_cell(Cell::new(0, 0), 3);
        s.add_cell(Cell::new(1, 0), 2);
        assert_eq!(s.num_cells(), 2);
        assert_eq!(s.stalk_dim(0), Some(3));
        assert_eq!(s.stalk_dim(1), Some(2));
    }

    #[test]
    fn test_sheaf_global_dimension() {
        let mut s = Sheaf::new();
        s.add_cell(Cell::new(0, 0), 3);
        s.add_cell(Cell::new(1, 0), 2);
        assert_eq!(s.global_dimension(), 5);
    }

    #[test]
    fn test_sheaf_assignment() {
        let mut s = Sheaf::new();
        s.add_cell(Cell::new(0, 0), 2);
        s.assign(Assignment::new(0, vec![1.0, 2.0]));
        let ga = s.global_assignment();
        assert_eq!(ga, vec![1.0, 2.0]);
    }

    #[test]
    fn test_sheaf_maps_from() {
        let mut s = Sheaf::new();
        s.add_cell(Cell::new(0, 0), 2);
        s.add_cell(Cell::new(1, 1), 2);
        s.add_restriction_map(RestrictionMap::new(1, 0, vec![vec![1.0, 0.0], vec![0.0, 1.0]]));
        assert_eq!(s.maps_from(1).len(), 1);
        assert_eq!(s.maps_from(0).len(), 0);
    }

    #[test]
    fn test_line_sheaf() {
        let s = line_sheaf(2);
        assert_eq!(s.num_cells(), 3);
        assert_eq!(s.num_maps(), 2);
        assert_eq!(s.global_dimension(), 6);
    }

    #[test]
    fn test_triangle_sheaf() {
        let s = triangle_sheaf(2);
        assert_eq!(s.num_cells(), 7);
        assert_eq!(s.num_maps(), 9);
    }

    #[test]
    fn test_empty_map_dims() {
        let m = RestrictionMap::new(0, 1, vec![]);
        assert_eq!(m.output_dim(), 0);
        assert_eq!(m.input_dim(), 0);
        assert_eq!(m.apply(&[]), vec![]);
    }
}
