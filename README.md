# sheaf-coherence-rs

**Cellular sheaf coherence — measuring how well local data assembles into global structure.**

This crate implements the mathematics of cellular sheaves in Rust: define stalks on cells, attach restriction maps between them, assign local data, and then measure how coherent those assignments are using sheaf Laplacians (Hodge theory) and diffusion-based synchronization. With 42 tests covering edge cases from empty sheaves to multi-cell synchronization, it's a rigorous foundation for any system where local observations must agree globally.

## Why This Matters

Sheaf theory is the mathematical language of *local-to-global* consistency. In an AGI system, individual agents observe fragments of reality — sheaf coherence tells you whether those fragments stitch together into a coherent worldview or contain contradictions. The sheaf Laplacian generalizes the graph Laplacian: instead of measuring how much a signal varies across nodes, it measures how much local data *disagrees* through the restriction maps. This is the diagnostic tool for detecting hallucination, inconsistency, and alignment failure in multi-agent perception.

## Quick Start

```toml
# Cargo.toml
[dependencies]
sheaf-coherence-rs = "0.1.0"
```

```rust
use sheaf_coherence_rs::sheaf::{Cell, RestrictionMap, Assignment, Sheaf};
use sheaf_coherence_rs::coherence::{coherence_score, coherence_energy};
use sheaf_coherence_rs::synchronization::diffusion_synchronize;

// Build a sheaf on two cells with a projection between them
let mut sheaf = Sheaf::new();
sheaf.add_cell(Cell::new(0, 0));
sheaf.add_cell(Cell::new(1, 0));
sheaf.add_stalk_dim(0, 3); // Cell 0 has a 3D stalk
sheaf.add_stalk_dim(1, 2); // Cell 1 has a 2D stalk

// Restriction map: project 3D → 2D (drop z)
sheaf.add_restriction_map(RestrictionMap::new(
    0, 1,
    vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 0.0]],
));

// Assign noisy local data
sheaf.assign(Assignment::new(0, vec![1.0, 2.0, 5.0]));
sheaf.assign(Assignment::new(1, vec![1.1, 2.2]));

println!("Coherence score: {:.4}", coherence_score(&sheaf)); // ~0.96

// Synchronize via diffusion to find the nearest global section
let energies = diffusion_synchronize(&mut sheaf, 0.1, 50);
println!("Final energy: {:.6}", energies.last().unwrap());
println!("Post-sync score: {:.4}", coherence_score(&sheaf)); // ~1.0
```

## Architecture

| Module | Purpose |
|---|---|
| `sheaf` | Core data structures: `Cell`, `RestrictionMap`, `Assignment`, `Sheaf` |
| `coherence` | Energy functions, scores, per-cell incoherence diagnostics |
| `laplacian` | Sheaf Laplacian matrix construction, Hodge decomposition, kernel projection |
| `synchronization` | Diffusion-based and optimization-based methods to find global sections |

## API Tour

### Core Types (`sheaf`)

- **`Cell { id, dimension }`** — A cell in the cellular complex
- **`RestrictionMap { source, target, matrix }`** — Linear map between stalks
  - `.apply(&data)` — Project data through the map
  - `.transpose()` — Get the adjoint map
  - `.input_dim()` / `.output_dim()` — Dimension queries
- **`Assignment { cell_id, data }`** — Local data attached to a cell
- **`Sheaf`** — The central data structure
  - `.add_cell()`, `.add_stalk_dim()`, `.add_restriction_map()`
  - `.assign(Assignment)` — Attach data to a cell
  - `.global_assignment()` — Concatenate all assignments into one vector
  - `.global_dimension()` — Total dimension of all stalks
  - `.stalk_dim(id)` — Dimension of a specific stalk
  - `.cell_ids()` — All registered cell IDs

### Coherence Measurement (`coherence`)

- **`coherence_energy(&sheaf) → f64`** — Total disagreement energy (lower = better)
- **`normalized_coherence_energy(&sheaf) → f64`** — Per-map average
- **`coherence_score(&sheaf) → f64`** — 0.0 (incoherent) to 1.0 (perfect global section)
- **`is_global_section(&sheaf, tolerance)`** — Boolean check for perfect coherence
- **`cell_incoherence(&sheaf) → HashMap<usize, f64>`** — Per-cell blame attribution
- **`most_incoherent_cell(&sheaf) → Option<usize>`** — Worst offender

### Sheaf Laplacian (`laplacian`)

- **`sheaf_laplacian(&sheaf) → Vec<Vec<f64>>`** — Full Laplacian matrix
- **`laplacian_eigenvalues(&sheaf, iterations) → Vec<f64>`** — Via power iteration
- **`kernel_dimension(&sheaf, threshold) → usize`** — Dimension of global sections
- **`project_to_kernel(&sheaf) → Vec<f64>`** — Nearest global section

### Synchronization (`synchronization`)

- **`diffusion_synchronize(&mut sheaf, step_size, iterations) → Vec<f64>`** — Heat flow
- **`project_to_global_section(&mut sheaf, iterations, step_size) → Vec<f64>`**
- **`initialize_missing_assignments(&mut sheaf)`** — Zero-fill gaps
- **`is_fully_assigned(&sheaf) → bool`**

## Performance

- Dense matrix operations — suitable for sheaves with up to ~100 cells and stalk dimensions ≤ 20
- Sheaf Laplacian construction is O(n × d²) where n = number of restriction maps, d = max stalk dim
- Power iteration eigenvalues: O(k × N²) per eigenvalue, where N = global dimension
- Diffusion synchronization: O(iterations × N²) — converges rapidly for well-conditioned sheaves

## Ecosystem

Part of the **SuperInstance** family of crates for mathematical agent infrastructure:

- [`persistent-sheaf-rs`](https://github.com/SuperInstance/persistent-sheaf-rs) — Persistent homology for sheaves
- [`witness-topology-rs`](https://github.com/SuperInstance/witness-topology-rs) — Topological approximation from point clouds
- [`optimal-transport-rs`](https://github.com/SuperInstance/optimal-transport-rs) — Wasserstein distances for comparing distributions
- [`renormalization-group-rs`](https://github.com/SuperInstance/renormalization-group-rs) — Multi-scale analysis
- [`constraint-dynamics-rs`](https://github.com/SuperInstance/constraint-dynamics-rs) — Constraint satisfaction and propagation

## Ideas for Improvement

- **Sparse matrix support** — Use `sprs` or `nalgebra_sparse` for large sheaves
- **Parallel Laplacian construction** — Rayon over restriction maps
- **GPU acceleration** — CUDA/wGPU for large-scale synchronization
- **Streaming coherence** — Incremental updates as new assignments arrive
- **Categorical composition** — Functors between sheaves, pullback/pushforward
- **Persistent sheaf coherence** — Track coherence across filtration values

## License

MIT
