import torch


def canonicalize_for_current_runtime(adj):
    """
    Convert an intended N x N adjacency matrix into the matrix layout
    currently expected by the released GDesigner runtime.

    Intended semantics:
        adj[i][j] == 1  <=>  edge i -> j

    Current runtime behavior:
        - fixed_spatial_masks is flattened to N^2 values
        - potential_spatial_edges contains only N(N-1) non-self edges
        - zip(...) consumes the FIRST N(N-1) flattened mask values
        - those values are interpreted in this edge order:

            0->1, 0->2, ...
            1->0, 1->2, ...
            ...

    Therefore we:
        1. extract intended non-self edges in canonical runtime-edge order
        2. place those bits into the first N(N-1) flat positions
        3. zero all remaining flat positions

    This does NOT change graph content conceptually.
    It only aligns the generator's edge coordinate system with the
    released runtime's mask coordinate system.
    """

    if isinstance(adj, torch.Tensor):
        A = adj.detach().cpu().int().tolist()
    else:
        A = [
            [int(x) for x in row]
            for row in adj
        ]

    n = len(A)

    if any(len(row) != n for row in A):
        raise ValueError("Adjacency matrix must be square.")

    # Canonical intended non-self edge order.
    mask = []

    for i in range(n):
        for j in range(n):
            if i != j:
                mask.append(
                    int(A[i][j])
                )

    # Current released runtime consumes first N(N-1)
    # positions of an N^2 flattened matrix.
    runtime_flat = [0] * (n * n)

    for k, bit in enumerate(mask):
        runtime_flat[k] = bit

    runtime_matrix = [
        runtime_flat[i*n:(i+1)*n]
        for i in range(n)
    ]

    return runtime_matrix


def intended_nonself_adjacency(adj):
    """
    Return intended adjacency with diagonal forced to zero.
    Useful for semantic-fidelity checks.
    """

    if isinstance(adj, torch.Tensor):
        A = adj.detach().cpu().int().tolist()
    else:
        A = [
            [int(x) for x in row]
            for row in adj
        ]

    n = len(A)

    out = [
        row[:] for row in A
    ]

    for i in range(n):
        out[i][i] = 0

    return out


if __name__ == "__main__":
    # Minimal sanity check.

    A = [
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ]

    print("Intended adjacency:")

    for row in A:
        print(row)

    print("\nRuntime-aligned input:")

    B = canonicalize_for_current_runtime(
        A
    )

    for row in B:
        print(row)
