# sheaf-coherence-rs

Cellular sheaf coherence — measure how well local data assembles into global structure, synchronize via diffusion, and diagnose disagreements.

Individual agents see fragments of reality. A sheaf tells you whether those
fragments stitch together into a consistent worldview. The sheaf Laplacian
generalizes the graph Laplacian: instead of measuring signal variation across
nodes, it measures how much local data *disagrees* through the restriction maps.

## Why Care?

Three agents observe the same event. Agent A sees `[1.0, 2.0]`, Agent B sees
`[1.1, 2.0]`, Agent C (the edge between them) says they should agree. But they
don't quite. How bad is it?

```rust
// A: [1.0, 2.0]  ←restrict←  C: [1.0, 2.0]  →restrict→  B: [1.1, 2.0]
//                                        disagreement at B: 0.01
```

The sheaf coherence score quantifies this: 1.0 means perfect agreement, 0.0
means total incoherence. The per-cell incoherence breakdown tells you *which*
agent is causing problems. Diffusion synchronization pushes everyone toward
the nearest global section.

## Quick Start

```toml
# Cargo.toml
[dependencies]
sheaf-coherence-rs = "0.1.0"
```

```rust
use sheaf_coherence_rs::sheaf::{Cell, RestrictionMap, Assignment, Sheaf, line_sheaf};
use sheaf_coherence_rs::coherence::{coherence_score, coherence_energy, is_global_section};
use sheaf_coherence_rs::synchronization::diffusion_synchronize;

// Build a simple sheaf: two vertices connected by an edge
let mut sheaf = line_sheaf(2);
// Cell 0 (vertex), Cell 1 (vertex), Cell 2 (edge)
// Identity restriction maps: edge → each vertex

// Assign slightly disagreeing data
sheaf.assign(Assignment::new(0, vec![1.0, 2.0]));
sheaf.assign(Assignment::new(1, vec![1.1, 2.0]));
sheaf.assign(Assignment::new(2, vec![1.0, 2.0]));

println!("Coherence score: {:.4}", coherence_score(&sheaf));
// => ~0.99 (nearly perfect)

println!("Total energy: {:.6}", coherence_energy(&sheaf));
// => ~0.01

// Synchronize: push toward global section via heat equation
let energies = diffusion_synchronize(&mut sheaf, 0.1, 50);
println!("Energy after sync: {:.6}", energies.last().unwrap());
// => Much smaller (agents converged)

println!("Is global section: {}", is_global_section(&sheaf, 1e-4));
// => true (within tolerance)
```

## Core Concepts Through Code

### Building a Sheaf from Scratch

A cellular sheaf assigns vector spaces (stalks) to cells and linear maps
(restrictions) between incident cells:

```rust
use sheaf_coherence_rs::sheaf::{Cell, RestrictionMap, Assignment, Sheaf};

let mut sheaf = Sheaf::new();

// Two vertices with 2D stalks and one edge with 3D stalk
sheaf.add_cell(Cell::new(0, 0), 2);  // vertex, stalk dimension 2
sheaf.add_cell(Cell::new(1, 0), 2);  // vertex, stalk dimension 2
sheaf.add_cell(Cell::new(2, 1), 3);  // edge,   stalk dimension 3

// Restriction maps: project the 3D edge stalk down to each 2D vertex stalk
sheaf.add_restriction_map(RestrictionMap::new(
    2, 0,  // from edge(2) to vertex(0)
    vec![
        vec![1.0, 0.0],  // x-coordinate
        vec![0.0, 1.0],  // y-coordinate
        vec![0.0, 0.0],  // z dropped
    ],
));
sheaf.add_restriction_map(RestrictionMap::new(
    2, 1,  // from edge(2) to vertex(1)
    vec![
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![0.0, 0.0],
    ],
));

// Assign data to each cell
sheaf.assign(Assignment::new(0, vec![1.0, 2.0]));   // vertex 0 sees (1, 2)
sheaf.assign(Assignment::new(1, vec![1.0, 2.0]));   // vertex 1 sees (1, 2)
sheaf.assign(Assignment::new(2, vec![1.0, 2.0, 5.0])); // edge sees (1, 2, 5)

println!("Global dimension: {}", sheaf.global_dimension());
// => 7 (= 2 + 2 + 3)
println!("Number of cells: {}", sheaf.num_cells());
// => 3
println!("Number of maps: {}", sheaf.num_maps());
// => 2
```

### Measuring Coherence

The coherence energy sums the disagreement over all restriction maps:

```rust
use sheaf_coherence_rs::sheaf::{Cell, RestrictionMap, Assignment, Sheaf, line_sheaf};
use sheaf_coherence_rs::coherence::{
    coherence_energy, normalized_coherence_energy, coherence_score,
    is_global_section, cell_incoherence, most_incoherent_cell,
    pairwise_disagreement,
};

let mut sheaf = line_sheaf(2);

// Perfect agreement: all cells have the same data
sheaf.assign(Assignment::new(0, vec![5.0, 3.0]));
sheaf.assign(Assignment::new(1, vec![5.0, 3.0]));
sheaf.assign(Assignment::new(2, vec![5.0, 3.0]));

println!("Energy: {}", coherence_energy(&sheaf));
// => 0.0 (perfect)
println!("Score:  {:.4}", coherence_score(&sheaf));
// => 1.0000
println!("Is global section: {}", is_global_section(&sheaf, 1e-10));
// => true

// Now introduce disagreement
sheaf.assign(Assignment::new(1, vec![0.0, 0.0])); // vertex 1 disagrees!
println!("Score: {:.4}", coherence_score(&sheaf));
// => < 1.0

// Find the worst offender
if let Some(worst) = most_incoherent_cell(&sheaf) {
    println!("Most incoherent cell: {}", worst);
}

// Per-cell breakdown
let incoherence = cell_incoherence(&sheaf);
for (cell_id, energy) in &incoherence {
    println!("  Cell {}: {:.4}", cell_id, energy);
}

// Pairwise disagreement between specific cells
if let Some(d) = pairwise_disagreement(&sheaf, 2, 0) {
    println!("Edge → Vertex 0 disagreement: {:.4}", d);
}
```

### The Sheaf Laplacian

The sheaf Laplacian `L` generalizes the graph Laplacian. For a global
assignment vector `x`, the energy `x^T L x` equals the total disagreement:

```rust
use sheaf_coherence_rs::sheaf::line_sheaf;
use sheaf_coherence_rs::laplacian::{
    sheaf_laplacian, apply_laplacian, laplacian_energy,
    is_positive_semidefinite, gershgorin_bounds, kernel_dimension,
};

let sheaf = line_sheaf(2);

// Build the Laplacian matrix
let lap = sheaf_laplacian(&sheaf);
println!("Laplacian size: {}×{}", lap.len(), lap[0].len());
// => 6×6 (3 cells × 2D stalks)

// Check properties
println!("Is PSD: {}", is_positive_semidefinite(&sheaf));
// => true (required for a valid sheaf Laplacian)

// Eigenvalue bounds via Gershgorin circles
let bounds = gershgorin_bounds(&lap);
for (i, (lo, hi)) in bounds.iter().enumerate() {
    println!("  λ[{}] ∈ [{:.4}, {:.4}]", i, lo, hi);
}

// Apply Laplacian to a vector
let v = sheaf.global_assignment();
let lv = apply_laplacian(&sheaf, &v);
println!("L·v = {:?}", lv);

// Energy: v^T L v
let energy = laplacian_energy(&sheaf, &v);
println!("Energy: {:.6}", energy);

// Kernel dimension = dimension of global sections
let ker_dim = kernel_dimension(&sheaf, 10.0);
println!("Kernel dimension: {}", ker_dim);
// => ≥ 1 (constant sections are always in the kernel)
```

### Synchronization via Diffusion

Push disagreeing assignments toward the nearest global section:

```rust
use sheaf_coherence_rs::sheaf::{Assignment, line_sheaf};
use sheaf_coherence_rs::synchronization::{
    diffusion_synchronize, synchronize_until, is_fully_assigned,
    initialize_missing_assignments, consensus_assignments, average_assignment,
};

let mut sheaf = line_sheaf(1);

// Agents disagree strongly
sheaf.assign(Assignment::new(0, vec![0.0]));
sheaf.assign(Assignment::new(1, vec![10.0]));
sheaf.assign(Assignment::new(2, vec![5.0]));

// Run diffusion: heat equation on the sheaf Laplacian
let energies = diffusion_synchronize(&mut sheaf, 0.1, 50);
println!("Energy went from {:.4} to {:.4}",
    energies[0], energies[energies.len()-1]);
// => Energy decreases monotonically

// Or run until convergence
let mut sheaf2 = line_sheaf(1);
sheaf2.assign(Assignment::new(0, vec![0.0]));
sheaf2.assign(Assignment::new(1, vec![1.0]));
sheaf2.assign(Assignment::new(2, vec![0.5]));
let (iters, residual) = synchronize_until(&mut sheaf2, 0.1, 1000, 1e-6);
println!("Converged in {} iterations, residual {:.8}", iters, residual);

// Utility: initialize missing cells with zeros
let mut sheaf3 = line_sheaf(2);
sheaf3.assign(Assignment::new(0, vec![1.0, 2.0]));
assert!(!is_fully_assigned(&sheaf3));
initialize_missing_assignments(&mut sheaf3);
assert!(is_fully_assigned(&sheaf3));

// Consensus: set everyone to the average
consensus_assignments(&mut sheaf3);
let avg = average_assignment(&sheaf3);
println!("Consensus value: {:?}", avg);
```

### Triangle Sheaf: Higher-Dimensional Cell Complex

```rust
use sheaf_coherence_rs::sheaf::{triangle_sheaf, Assignment};
use sheaf_coherence_rs::coherence::{coherence_score, coherence_energy};
use sheaf_coherence_rs::laplacian::{sheaf_laplacian, is_positive_semidefinite};

// 3 vertices + 3 edges + 1 face (7 cells total, 9 restriction maps)
let mut sheaf = triangle_sheaf(2);

println!("Cells: {}", sheaf.num_cells());  // => 7
println!("Maps:  {}", sheaf.num_maps());    // => 9

// Assign consistent data to all cells
let data = vec![3.0, 7.0];
for id in sheaf.cell_ids() {
    sheaf.assign(Assignment::new(id, data.clone()));
}
println!("Perfect coherence: {:.4}", coherence_score(&sheaf));
// => 1.0000

// The Laplacian is still PSD
assert!(is_positive_semidefinite(&sheaf));

let lap = sheaf_laplacian(&sheaf);
println!("Laplacian: {}×{}", lap.len(), lap[0].len());
// => 14×14 (7 cells × 2D stalks)
```

### Restriction Maps: Projections and Transformations

```rust
use sheaf_coherence_rs::sheaf::{Cell, RestrictionMap, Sheaf};

// A map that projects 3D → 2D (drops the z-coordinate)
let projection = RestrictionMap::new(
    0, 1,
    vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
    ],
);
assert_eq!(projection.input_dim(), 3);
assert_eq!(projection.output_dim(), 2);

let result = projection.apply(&[3.0, 4.0, 5.0]);
assert_eq!(result, vec![3.0, 4.0]); // z=5.0 dropped

// Transpose: 2D → 3D (pads with zero)
let adjoint = projection.transpose();
let adj_result = adjoint.apply(&[3.0, 4.0]);
assert_eq!(adj_result, vec![3.0, 4.0, 0.0]);
```

## API Reference

### `sheaf` Module — Core Data Structures

```rust
pub struct Cell {
    pub id: usize,
    pub dimension: usize,
}
impl Cell {
    pub fn new(id: usize, dimension: usize) -> Self
}

pub struct RestrictionMap {
    pub source: usize,
    pub target: usize,
    pub matrix: Vec<Vec<f64>>,
}
impl RestrictionMap {
    pub fn new(source: usize, target: usize, matrix: Vec<Vec<f64>>) -> Self
    pub fn apply(&self, v: &[f64]) -> Vec<f64>
    pub fn transpose(&self) -> RestrictionMap
    pub fn output_dim(&self) -> usize
    pub fn input_dim(&self) -> usize
}

pub struct Assignment {
    pub cell_id: usize,
    pub data: Vec<f64>,
}
impl Assignment {
    pub fn new(cell_id: usize, data: Vec<f64>) -> Self
    pub fn dimension(&self) -> usize
}

pub struct Sheaf { /* fields */ }
impl Sheaf {
    pub fn new() -> Self
    pub fn add_cell(&mut self, cell: Cell, stalk_dimension: usize)
    pub fn add_restriction_map(&mut self, map: RestrictionMap)
    pub fn assign(&mut self, assignment: Assignment)
    pub fn stalk_dim(&self, cell_id: usize) -> Option<usize>
    pub fn cell_ids(&self) -> Vec<usize>
    pub fn global_dimension(&self) -> usize
    pub fn global_assignment(&self) -> Vec<f64>
    pub fn maps_from(&self, cell_id: usize) -> Vec<&RestrictionMap>
    pub fn maps_to(&self, cell_id: usize) -> Vec<&RestrictionMap>
    pub fn num_maps(&self) -> usize
    pub fn num_cells(&self) -> usize
}

// Constructors
pub fn line_sheaf(dim: usize) -> Sheaf       // 2 vertices + 1 edge
pub fn triangle_sheaf(dim: usize) -> Sheaf    // 3 vertices + 3 edges + 1 face
```

### `coherence` Module — Measurement

```rust
pub fn map_disagreement(map: &RestrictionMap, source: &Assignment, target: &Assignment) -> f64
pub fn coherence_energy(sheaf: &Sheaf) -> f64
pub fn normalized_coherence_energy(sheaf: &Sheaf) -> f64
pub fn coherence_score(sheaf: &Sheaf) -> f64        // 0.0 to 1.0
pub fn is_global_section(sheaf: &Sheaf, tolerance: f64) -> bool
pub fn cell_incoherence(sheaf: &Sheaf) -> HashMap<usize, f64>
pub fn most_incoherent_cell(sheaf: &Sheaf) -> Option<usize>
pub fn pairwise_disagreement(sheaf: &Sheaf, source_id: usize, target_id: usize) -> Option<f64>
```

### `laplacian` Module — Hodge Theory

```rust
pub fn sheaf_laplacian(sheaf: &Sheaf) -> Vec<Vec<f64>>
pub fn apply_laplacian(sheaf: &Sheaf, v: &[f64]) -> Vec<f64>
pub fn laplacian_energy(sheaf: &Sheaf, v: &[f64]) -> f64
pub fn kernel_dimension(sheaf: &Sheaf, tolerance: f64) -> usize
pub fn gershgorin_bounds(lap: &[Vec<f64>]) -> Vec<(f64, f64)>
pub fn is_positive_semidefinite(sheaf: &Sheaf) -> bool
```

### `synchronization` Module — Diffusion

```rust
pub fn diffusion_synchronize(sheaf: &mut Sheaf, step_size: f64, iterations: usize) -> Vec<f64>
pub fn project_to_global_section(sheaf: &mut Sheaf, iterations: usize, step_size: f64) -> Vec<f64>
pub fn synchronize_until(sheaf: &mut Sheaf, step_size: f64, max_iterations: usize, tolerance: f64) -> (usize, f64)
pub fn is_fully_assigned(sheaf: &Sheaf) -> bool
pub fn initialize_missing_assignments(sheaf: &mut Sheaf)
pub fn average_assignment(sheaf: &Sheaf) -> Option<Vec<f64>>
pub fn consensus_assignments(sheaf: &mut Sheaf)
pub fn max_change(v1: &[f64], v2: &[f64]) -> f64
pub fn spectral_gap_estimate(sheaf: &Sheaf, tolerance: f64) -> f64
```

## Advanced Examples

### Multi-Agent Perception Sheaf

Build a sheaf where each agent observes a shared reality through different
sensors, then synchronize their views:

```rust
use sheaf_coherence_rs::sheaf::{Cell, RestrictionMap, Assignment, Sheaf};
use sheaf_coherence_rs::coherence::{coherence_score, most_incoherent_cell, cell_incoherence};
use sheaf_coherence_rs::synchronization::{diffusion_synchronize, initialize_missing_assignments};
use sheaf_coherence_rs::laplacian::sheaf_laplacian;

// 4 agents observing a 3D scene, each with a 2D camera (different projections)
let mut sheaf = Sheaf::new();

// Shared reality: 3D state
sheaf.add_cell(Cell::new(0, 2), 3); // "world" cell (dim 2 in complex, 3D stalk)

// Each agent has a 2D observation
for agent_id in 1..=4 {
    sheaf.add_cell(Cell::new(agent_id, 0), 2);
}

// Agent cameras: different rotation+projection of the 3D world
sheaf.add_restriction_map(RestrictionMap::new(0, 1, vec![
    vec![1.0, 0.0, 0.0],  // Agent 1 sees x, y
    vec![0.0, 1.0, 0.0],
]));
sheaf.add_restriction_map(RestrictionMap::new(0, 2, vec![
    vec![0.0, 1.0, 0.0],  // Agent 2 sees y, z
    vec![0.0, 0.0, 1.0],
]));
sheaf.add_restriction_map(RestrictionMap::new(0, 3, vec![
    vec![1.0, 0.0, 0.0],  // Agent 3 sees x, z
    vec![0.0, 0.0, 1.0],
]));
sheaf.add_restriction_map(RestrictionMap::new(0, 4, vec![
    vec![0.707, 0.707, 0.0], // Agent 4 sees (x+y)/√2, z
    vec![0.0,   0.0,   1.0],
]));

// Noisy observations from each agent
sheaf.assign(Assignment::new(0, vec![1.0, 2.0, 3.0])); // world
sheaf.assign(Assignment::new(1, vec![1.1, 2.0]));       // slight noise in x
sheaf.assign(Assignment::new(2, vec![2.0, 3.1]));       // slight noise in z
sheaf.assign(Assignment::new(3, vec![1.0, 3.0]));       // perfect
sheaf.assign(Assignment::new(4, vec![2.15, 3.0]));      // slight noise

println!("Initial coherence: {:.4}", coherence_score(&sheaf));

// Find which agent is causing the most incoherence
if let Some(worst) = most_incoherent_cell(&sheaf) {
    println!("Most incoherent cell: {}", worst);
}

// Synchronize to find the best consensus
let energies = diffusion_synchronize(&mut sheaf, 0.05, 100);
println!("Final coherence: {:.4}", coherence_score(&sheaf));
println!("Energy reduction: {:.4} → {:.4}", energies[0], *energies.last().unwrap());
```

### Diagnostic Pipeline: Find and Fix Incoherence

```rust
use sheaf_coherence_rs::sheaf::{Cell, RestrictionMap, Assignment, Sheaf, line_sheaf};
use sheaf_coherence_rs::coherence::{
    coherence_energy, coherence_score, is_global_section,
    cell_incoherence, most_incoherent_cell,
};
use sheaf_coherence_rs::synchronization::synchronize_until;

fn diagnose_and_fix(sheaf: &mut sheaf_coherence_rs::sheaf::Sheaf) {
    println!("=== Sheaf Coherence Diagnostic ===");
    println!("Cells: {}, Maps: {}", sheaf.num_cells(), sheaf.num_maps());

    let score = coherence_score(sheaf);
    let energy = coherence_energy(sheaf);

    println!("Coherence score: {:.4}", score);
    println!("Total energy:    {:.6}", energy);

    if is_global_section(sheaf, 1e-6) {
        println!("✓ Already a global section — no fix needed");
        return;
    }

    // Which cells are causing problems?
    let per_cell = cell_incoherence(sheaf);
    let mut sorted: Vec<_> = per_cell.iter().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    println!("\nIncoherence by cell:");
    for (cell_id, energy) in &sorted {
        println!("  Cell {}: {:.6}", cell_id, energy);
    }

    if let Some(worst) = most_incoherent_cell(sheaf) {
        println!("\nWorst offender: cell {}", worst);
    }

    // Synchronize
    println!("\nRunning diffusion synchronization...");
    let (iters, residual) = synchronize_until(sheaf, 0.05, 5000, 1e-6);
    println!("Converged in {} iterations", iters);
    println!("Final coherence: {:.6}", coherence_score(sheaf));
}

let mut sheaf = line_sheaf(2);
sheaf.assign(Assignment::new(0, vec![1.0, 0.0]));
sheaf.assign(Assignment::new(1, vec![0.0, 1.0]));
sheaf.assign(Assignment::new(2, vec![1.0, 1.0]));
diagnose_and_fix(&mut sheaf);
```

### Integration with entropy-conservation-rs

Use sheaf coherence to verify that entropy assignments are globally consistent:

```rust
use sheaf_coherence_rs::sheaf::{Cell, RestrictionMap, Assignment, Sheaf};
use sheaf_coherence_rs::coherence::{coherence_score, is_global_section};

// Model entropy budgets across agents as a sheaf
fn entropy_consistency_check(
    agent_entropies: &[(usize, Vec<f64>)],
    shared_constraints: &[(usize, usize, Vec<Vec<f64>>)],
) -> bool {
    let mut sheaf = Sheaf::new();

    // Each agent has a stalk for its entropy vector
    for (id, entropy) in agent_entropies {
        sheaf.add_cell(Cell::new(*id, 0), entropy.len());
        sheaf.assign(Assignment::new(*id, entropy.clone()));
    }

    // Constraint maps enforce consistency between shared resources
    for (src, tgt, matrix) in shared_constraints {
        sheaf.add_restriction_map(RestrictionMap::new(*src, *tgt, matrix.clone()));
    }

    // If the coherence is perfect, all agents agree on shared resources
    is_global_section(&sheaf, 1e-6)
}

let agents = vec![
    (0, vec![1.0, 2.0]),
    (1, vec![1.0, 2.0]),
];
let constraints = vec![
    (0, 1, vec![vec![1.0, 0.0], vec![0.0, 1.0]]), // identity: must match exactly
];
println!("Entropy consistent: {}", entropy_consistency_check(&agents, &constraints));
// => true
```

## Conservation Law Connections

The sheaf Laplacian is a conservation-law enforcer. For a sheaf built from
conservation constraints (e.g., "total entropy at connected cells must sum to
the same value"), the kernel of the Laplacian is exactly the space of
conservation-law-satisfying assignments.

- **Kernel dimension** = number of independent global sections = degrees of freedom
  in the conserved system
- **Spectral gap** = speed of synchronization = how fast information about
  conservation violations propagates through the network
- **Coherence energy** = total violation magnitude = how far the system is from
  satisfying all conservation laws simultaneously

### Relation to Other SuperInstance Crates

- **`entropy-conservation-rs`** — Entropy conservation is a special sheaf where
  restriction maps enforce sum invariants
- **`renormalization-group-rs`** — Coarse-graining defines restriction maps
  between scales; sheaf coherence measures whether the coarse-graining is
  consistent
- **`constraint-dynamics-rs`** — The sheaf Laplacian is the constraint force
  that drives the system toward conservation-law satisfaction

## Performance

- Sheaf construction: O(cells + maps)
- Coherence energy: O(maps × stalk_dim²)
- Sheaf Laplacian: O(maps × stalk_dim²) construction, O(global_dim²) storage
- Diffusion: O(iterations × global_dim²) per step
- Suitable for sheaves with up to ~100 cells and stalk dimensions ≤ 20

## License

MIT
