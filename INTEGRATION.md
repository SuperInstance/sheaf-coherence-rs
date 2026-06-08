# INTEGRATION.md — sheaf-coherence-rs × spectral-fleet-rs × conservation-law-rs

**Cellular sheaf coherence** measures how well local agent data glues into a
global structure. It connects to spectral methods for Laplacian analysis and
to Lagrangian mechanics for conservation-based consistency checks.

## Synergy Map

```
spectral-fleet-rs            sheaf-coherence-rs             conservation-law-rs
┌──────────────────┐        ┌──────────────────────┐       ┌─────────────────────┐
│ l2_norm           │        │ Sheaf                │       │ AgentState          │
│ normalize         │◄──────►│ RestrictionMap       │◄─────►│ total_energy        │
│ PowerIteration    │        │ Assignment           │       │ verify_noether      │
│ SpectralClustering│        │ sheaf_laplacian      │       │ ChargeMonitor       │
└──────────────────┘        │ diffusion_synchronize│       └─────────────────────┘
                            │ coherence_score      │
                            │ cell_incoherence     │
                            └──────────────────────┘
```

## Key Insight

In a fleet, each agent holds local state. Restrictions maps define how data
on one agent relates to data on another. The sheaf Laplacian measures
inconsistencies. Spectral-fleet finds the low-frequency modes (consensus
subspaces) of that Laplacian. Conservation-law verifies that the globally
synchronized state preserves energy.

## Example 1: Spectral Analysis of the Sheaf Laplacian

Build a sheaf from agent assignments, compute its Laplacian, and find the
spectral gap using power iteration.

```rust
use sheaf_coherence::sheaf::{Sheaf, Cell, RestrictionMap, Assignment};
use sheaf_coherence::laplacian::sheaf_laplacian;
use spectral_fleet::{l2_norm, normalize, dot};

fn spectral_gap_of_fleet_sheaf() -> f64 {
    // Build a 3-agent line sheaf with 2D stalks
    let mut sheaf = Sheaf::new();
    sheaf.add_cell(Cell::new(0, 0), 2);
    sheaf.add_cell(Cell::new(1, 0), 2);
    sheaf.add_cell(Cell::new(2, 0), 2);

    // Identity restriction maps between neighbors
    sheaf.add_restriction_map(RestrictionMap::new(0, 1, vec![
        vec![1.0, 0.0],
        vec![0.0, 1.0],
    ]));
    sheaf.add_restriction_map(RestrictionMap::new(1, 2, vec![
        vec![1.0, 0.0],
        vec![0.0, 1.0],
    ]));

    // Assign local data
    sheaf.assign(Assignment::new(0, vec![1.0, 2.0]));
    sheaf.assign(Assignment::new(1, vec![1.1, 1.9]));
    sheaf.assign(Assignment::new(2, vec![0.9, 2.1]));

    // Compute Laplacian
    let lap = sheaf_laplacian(&sheaf);
    let n = lap.len();

    // Power iteration to find dominant eigenvalue
    let mut vec = vec![1.0; n];
    normalize(&mut vec);
    for _ in 0..1000 {
        let mut next = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                next[i] += lap[i][j] * vec[j];
            }
        }
        let ev = dot(&next, &vec);
        normalize(&mut next);
        vec = next;
    }

    let dominant = dot(&vec, &lap.iter().enumerate().map(|(i, row)| {
        row.iter().zip(vec.iter()).map(|(&a, &b)| a * b).sum::<f64>()
    }).collect::<Vec<f64>>());

    println!("Sheaf Laplacian dominant eigenvalue: {:.4}", dominant);
    dominant
}
```

## Example 2: Synchronize Assignments Then Verify Energy Conservation

Diffuse agent assignments to consensus, then use conservation-law to check
that the synchronized state is physically valid.

```rust
use sheaf_coherence::sheaf::{Sheaf, Cell, RestrictionMap, Assignment};
use sheaf_coherence::synchronization::synchronize_until;
use conservation_law::lagrangian::{AgentState, MechanicalLagrangian, total_energy};

fn synchronize_and_verify() {
    let mut sheaf = Sheaf::new();
    for i in 0..3 {
        sheaf.add_cell(Cell::new(i, 0), 1);
    }
    sheaf.add_restriction_map(RestrictionMap::new(0, 1, vec![vec![1.0]]));
    sheaf.add_restriction_map(RestrictionMap::new(1, 2, vec![vec![1.0]]));
    sheaf.assign(Assignment::new(0, vec![10.0]));
    sheaf.assign(Assignment::new(1, vec![12.0]));
    sheaf.assign(Assignment::new(2, vec![8.0]));

    let (iters, final_change) = synchronize_until(&mut sheaf, 0.1, 1000, 1e-6);
    println!("Converged after {} iterations, final change = {:.6}", iters, final_change);

    // Verify energy conservation of synchronized state
    let global = sheaf.global_assignment();
    let lagrangian = MechanicalLagrangian {
        mass: 1.0,
        potential_fn: |q: &[f64; 1]| 0.5 * q[0] * q[0],
    };
    let state = AgentState::new([global[0]], [0.0]);
    let e = total_energy(&lagrangian, &state);
    println!("Synchronized state energy: {:.4}", e);
}
```

## Example 3: Coherence as a Fleet Health Metric

Use `coherence_score` to measure fleet consistency and flag incoherent agents.

```rust
use sheaf_coherence::coherence::{coherence_score, most_incoherent_cell};
use sheaf_coherence::sheaf::{Sheaf, Cell, RestrictionMap, Assignment};

fn fleet_health_score(sheaf: &Sheaf) -> f64 {
    let score = coherence_score(sheaf);
    println!("Fleet coherence score: {:.4} (1.0 = perfect)", score);

    if let Some(bad_cell) = most_incoherent_cell(sheaf) {
        println!("Most incoherent agent: {}", bad_cell);
    }
    score
}
```

## Cargo.toml Wiring

```toml
[dependencies]
sheaf-coherence = { git = "https://github.com/SuperInstance/sheaf-coherence-rs" }
spectral-fleet = { git = "https://github.com/SuperInstance/spectral-fleet-rs" }
conservation-law = { git = "https://github.com/SuperInstance/conservation-law-rs" }
```
