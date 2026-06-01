# lau-tropical-agent

A Rust crate that applies **tropical geometry** (max-plus algebra) to agent decision-making. In the tropical limit — think ℏ → 0 — neural-network activations collapse into piecewise-linear operations, softmax becomes hardmax, and optimal decisions reduce to tropical rational functions. This crate implements the full algebraic stack: semiring elements, matrices, polynomials, spectral theory, attention, shortest paths, and decision agents.

---

## What This Does

| Module | What It Gives You |
|---|---|
| `semiring` | The tropical semiring (ℝ ∪ {−∞}, max, +) with verified axioms |
| `matrix` | Tropical matrix multiply, power, Kleene star, trace, matrix-vector product |
| `polynomial` | Tropical polynomials — evaluation, corner locus (roots), Newton polytope |
| `rational_map` | Tropical rational functions f/g — piecewise-linear maps |
| `attention` | Tropical softmax/hardmax attention mechanism |
| `decision` | Max-plus decision agent with value iteration |
| `eigen` | Tropical eigenvalues (maximum cycle mean) and eigenvectors |
| `shortest_path` | All-pairs and single-source shortest paths via tropical matrix algebra |
| `spectral` | Tropical spectral radius, singular values, condition number |

All types derive `Serialize`/`Deserialize`. Every module includes inline tests verifying the algebraic laws (commutativity, associativity, distributivity, idempotency).

---

## Key Idea

In standard algebra, "add" is + and "multiply" is ×. In the **tropical semiring** (max-plus), addition ⊕ = max and multiplication ⊗ = +. This tiny swap has enormous consequences:

- **Matrix multiplication** becomes shortest-path computation: `C[i][j] = max_k (A[i][k] + B[k][j])`.
- **Eigenvalues** become maximum cycle means in a weighted digraph.
- **Softmax** degenerates to hardmax (the tropical limit of exponentials).
- **Polynomials** become piecewise-linear convex functions whose "roots" are the corners.

This crate builds agent decision theory on top of that algebra — every optimal action is a tropical argmax, every value function a tropical matrix power.

---

## Install

```toml
[dependencies]
lau-tropical-agent = "0.1"
```

Requires Rust **2021 edition**. Depends on `serde`, `serde_json`, and `nalgebra`.

---

## Quick Start

### Tropical arithmetic

```rust
use lau_tropical_agent::Tropical;

let a = Tropical(3.0);
let b = Tropical(5.0);

assert_eq!(a + b, Tropical(5.0));  // max(3, 5) = 5
assert_eq!(a * b, Tropical(8.0));  // 3 + 5 = 8
assert_eq!(Tropical::NEG_INF + a, a); // -∞ is the additive identity
```

### Tropical matrix multiplication

```rust
use lau_tropical_agent::TropicalMatrix;

let a = TropicalMatrix::from_f64(&[
    vec![1.0, 2.0],
    vec![3.0, 4.0],
]);
let b = TropicalMatrix::from_f64(&[
    vec![5.0, 6.0],
    vec![7.0, 8.0],
]);
let c = a.mul(&b);
// C[0][0] = max(1+5, 2+7) = 9
assert_eq!(c.get(0, 0), Tropical(9.0));
```

### Tropical decision agent

```rust
use lau_tropical_agent::{TropicalDecision, TropicalMatrix};

let rewards = TropicalMatrix::from_f64(&[
    vec![1.0, 5.0, 3.0],  // state 0: action 1 is best
    vec![4.0, 2.0, 6.0],  // state 1: action 2 is best
]);
let agent = TropicalDecision::new(rewards, 0.9);

assert_eq!(agent.best_action(0), 1);
assert_eq!(agent.best_action(1), 2);
```

### Shortest paths via Kleene star

```rust
use lau_tropical_agent::{TropicalMatrix, shortest_path::tropical_shortest_paths};

let adj = TropicalMatrix::from_f64(&[
    vec![f64::NEG_INFINITY, 1.0, 5.0],
    vec![1.0, f64::NEG_INFINITY, 2.0],
    vec![5.0, 2.0, f64::NEG_INFINITY],
]);
let dists = tropical_shortest_paths(&adj);
// dists[i][j] = shortest (max-plus) distance from i to j
```

---

## API Reference

### `Tropical` — semiring element

| Method / Impl | Description |
|---|---|
| `Tropical(val)` | Wrap an f64 |
| `Tropical::NEG_INF` | Additive identity (−∞) |
| `Tropical::ONE` | Multiplicative identity (0) |
| `a + b` | Tropical addition: max(a, b) |
| `a * b` | Tropical multiplication: a + b |
| `.pow(n)` | Tropical power: n·a |
| `.div(b)` | Tropical division: a − b |
| `.is_zero()` | Is this −∞? |
| `.is_finite()` | Is this not −∞? |

### `TropicalMatrix` — max-plus linear algebra

| Method | Description |
|---|---|
| `::from_f64(&[row])` | Build from ordinary f64 data |
| `::zeros(r, c)` | All −∞ |
| `::identity(n)` | 0 on diagonal, −∞ elsewhere |
| `.mul(&other)` | Tropical matrix multiply |
| `.add(&other)` | Element-wise max |
| `.pow(n)` | Repeated tropical multiply |
| `.kleene_star()` | I ⊕ A ⊕ A² ⊕ … ⊕ Aⁿ⁻¹ (all-pairs shortest paths) |
| `.apply_vec(&v)` | Tropical matrix-vector product |
| `.trace()` | Max of diagonal |
| `.to_nalgebra()` | Convert to `nalgebra::DMatrix<f64>` |

### `TropicalPoly` — tropical polynomial

| Method | Description |
|---|---|
| `::new(vec![(coeff, degree)])` | Build from terms |
| `::monomial(coeff, degree)` | Single term |
| `.eval(x)` | maxᵢ (cᵢ + dᵢ·x) |
| `.corner_locus()` | Points where ≥ 2 terms tie (tropical roots) |
| `.newton_polytope()` | (min_degree, max_degree) |
| `.add(&other)` | Tropical polynomial addition |
| `.mul(&other)` | Tropical polynomial multiplication (convolution) |
| `.degree()` | Highest degree with finite coefficient |

### `TropicalRationalMap` — f(x)/g(x)

| Method | Description |
|---|---|
| `::new(numerator, denominator)` | Construct from two polynomials |
| `.eval(x)` | f(x) − g(x) in tropical sense |

### `TropicalAttention` — max-plus attention

| Method | Description |
|---|---|
| `::hard()` | Temperature = 0 (hardmax) |
| `::new(temperature)` | Soft tropical approximation |
| `.weights(&scores)` | Attention weights (one-hot at hard limit) |
| `.attend(&keys, &queries)` | Tropical dot-product attention |

### `TropicalDecision` — max-plus agent

| Method | Description |
|---|---|
| `::new(rewards, discount)` | Reward matrix + discount factor |
| `.best_action(state)` | Tropical argmax over actions |
| `.optimal_value(state)` | Max reward for a state |
| `.value_iteration(&transition, iters)` | Run V' = maxₐ(R + γ⊗V) |

### `eigen` module — spectral theory

| Function | Description |
|---|---|
| `tropical_eigen(&matrix)` | Maximum cycle mean (tropical eigenvalue) |
| `tropical_eigenvector(&matrix)` | Kleene-star eigenvector |

### `shortest_path` module

| Function | Description |
|---|---|
| `tropical_shortest_paths(&adj)` | All-pairs via Kleene star |
| `single_source_shortest_paths(&adj, src)` | Single-source via matrix-vector relaxation |
| `shortest_path_distance(&adj, from, to)` | Distance between two nodes |

### `spectral` module

| Function | Description |
|---|---|
| `spectral_radius(&matrix)` | Same as `tropical_eigen` |
| `tropical_singular_values(&matrix)` | Via tropical Grammian |
| `tropical_condition_number(&matrix)` | Max − min singular value (tropical ratio) |

---

## How It Works

### The Tropical Semiring
Replace + with max and × with +. The "zero" element is −∞ (since max(a, −∞) = a) and the "one" element is 0 (since a + 0 = a). All the usual semiring axioms hold — commutativity, associativity, distributivity, idempotency of addition (max(a, a) = a).

### Tropical Matrix Algebra
Matrix multiplication C = A ⊗ B uses tropical inner products:
```
C[i][j] = max_k (A[i][k] + B[k][j])
```
This is exactly the Floyd–Warshall / shortest-paths computation. The **Kleene star** A* = I ⊕ A ⊕ A² ⊕ … ⊕ Aⁿ⁻¹ gives all-pairs shortest-path distances.

### Tropical Eigenvalues
The tropical eigenvalue λ of a matrix is the **maximum cycle mean**: over all directed cycles in the graph, the one with the highest average edge weight. Computed by examining diagonal entries of successive powers Aᵏ.

### Tropical Polynomials
A tropical polynomial p(x) = maxᵢ (cᵢ + dᵢ·x) is a piecewise-linear convex function. Its **corner locus** — points where two terms tie for the maximum — plays the role of roots. The Newton polytope (convex hull of exponent vectors) governs the shape.

### Tropical Attention
Standard softmax attention uses softmax(QKᵀ/√d)·V. In the tropical limit (temperature → 0), softmax collapses to **hardmax**: all weight goes to the maximum-scoring key. The crate supports both hard and soft (temperature-parametrised) modes.

### Tropical Decision Theory
An agent with reward matrix R selects actions by tropical argmax. Value iteration becomes pure max-plus matrix algebra: V' = maxₐ(R(s,a) + γ·Σₛ' T(s,a,s')·V(s')), which in the tropical setting simplifies to tropical matrix-vector products.

---

## The Math

**Tropical semiring operations:**
$$a \oplus b = \max(a, b), \qquad a \otimes b = a + b$$

**Tropical matrix multiplication:**
$$(A \otimes B)_{ij} = \bigoplus_k A_{ik} \otimes B_{kj} = \max_k (A_{ik} + B_{kj})$$

**Tropical eigenvalue (max cycle mean):**
$$\lambda = \max_{\text{cycles } C} \frac{\sum_{(i,j) \in C} A_{ij}}{|C|}$$

**Tropical polynomial:**
$$p(x) = \bigoplus_i (c_i \otimes x^{\otimes d_i}) = \max_i (c_i + d_i \cdot x)$$

**Corner locus (tropical roots):**
$$\text{Corners} = \{ x : c_i + d_i \cdot x = c_j + d_j \cdot x \text{ for some } i \neq j \}$$

**Kleene star (all-pairs shortest paths):**
$$A^* = I \oplus A \oplus A^{\otimes 2} \oplus \cdots \oplus A^{\otimes (n-1)}$$

---

## Tests

The crate contains **~60 inline and integration tests** covering:
- Semiring axiom verification (commutativity, associativity, distributivity, identity, absorption, idempotency)
- Matrix multiply, power, Kleene star correctness
- Eigenvalue and eigenvector computation
- Polynomial evaluation, corner locus, Newton polytope
- Attention weight distribution (hard and soft)
- Decision agent action selection and value iteration
- Shortest-path correctness
- Serde round-trips for all serialisable types

Run with:
```bash
cargo test
```

---

## License

MIT
