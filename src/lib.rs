//! Cellular sheaf coherence — measuring how well local data assembles into global structure.
//!
//! This crate provides tools for working with cellular sheaves, measuring coherence
//! of local assignments, computing sheaf Laplacians (Hodge theory), and synchronizing
//! assignments across cells.

#![allow(clippy::needless_range_loop)]

pub mod coherence;
pub mod laplacian;
pub mod sheaf;
pub mod synchronization;
