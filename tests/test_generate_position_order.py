"""
Sanity tests for position order generation.
Run: python tests/test_generate_position_order.py
"""
import torch
import sys
sys.path.insert(0, ".")


def test_inverse_permutation():
    """Verify that the inverse permutation logic correctly recovers raster order."""
    print("Test 1: Inverse permutation correctness")
    bsz = 4
    block_size = 16  # small for testing

    # Create a known sequence in raster order
    raster_tokens = torch.arange(block_size).unsqueeze(0).expand(bsz, -1)  # [bsz, block_size]

    # Create random token orders
    token_order = torch.stack([torch.randperm(block_size) for _ in range(bsz)])

    # "Generate" tokens in token_order sequence (simulate what generate() produces)
    # At step i, the model generates the token for spatial position token_order[i]
    # So generated_seq[b, i] = raster_tokens[b, token_order[b, i]]
    generated_seq = torch.gather(raster_tokens, 1, token_order)

    # Now apply the inverse permutation (same logic as in generate_position_order.py)
    inv_order = torch.zeros_like(token_order)
    inv_order.scatter_(
        1, token_order,
        torch.arange(block_size).unsqueeze(0).expand_as(token_order)
    )
    recovered = torch.gather(generated_seq, 1, inv_order)

    assert torch.equal(recovered, raster_tokens), \
        f"Inverse permutation failed!\nExpected: {raster_tokens}\nGot: {recovered}"
    print("  PASSED: inverse permutation correctly recovers raster order\n")


def test_identity_order_is_noop():
    """Verify that raster (identity) token_order produces no reordering."""
    print("Test 2: Identity order is a no-op")
    bsz = 4
    block_size = 16

    token_order = torch.arange(block_size).unsqueeze(0).expand(bsz, -1)
    generated_seq = torch.randint(0, 1000, (bsz, block_size))

    inv_order = torch.zeros_like(token_order)
    inv_order.scatter_(
        1, token_order,
        torch.arange(block_size).unsqueeze(0).expand_as(token_order)
    )
    recovered = torch.gather(generated_seq, 1, inv_order)

    assert torch.equal(recovered, generated_seq), \
        "Identity order should not change the sequence!"
    print("  PASSED: identity order leaves sequence unchanged\n")


def test_seed_determinism():
    """Verify that same order-seed produces same token_order."""
    print("Test 3: Seed determinism")
    block_size = 256
    bsz = 8

    def make_order(seed):
        rng = torch.Generator(device='cpu')
        rng.manual_seed(seed)
        return torch.stack([
            torch.randperm(block_size, generator=rng)
            for _ in range(bsz)
        ])

    order1 = make_order(42)
    order2 = make_order(42)
    order3 = make_order(99)

    assert torch.equal(order1, order2), "Same seed should produce same order!"
    assert not torch.equal(order1, order3), "Different seeds should produce different orders!"
    print("  PASSED: seed determinism works correctly\n")


def test_freqs_cis_indexing():
    """
    Verify that order-aware freqs_cis indexing during inference
    matches the training-time reordering.
    """
    print("Test 4: freqs_cis indexing consistency")
    from autoregressive.models.gpt_position_order import precompute_freqs_cis_2d

    grid_size = 4  # 4x4 = 16 tokens
    block_size = grid_size ** 2
    head_dim = 64
    cls_token_num = 1

    freqs_cis = precompute_freqs_cis_2d(grid_size, head_dim, 10000, cls_token_num)
    # freqs_cis shape: [cls_token_num + block_size, head_dim//2, 2]

    bsz = 2
    token_order = torch.stack([torch.randperm(block_size) for _ in range(bsz)])

    # Training-time: reorder all freqs at once
    # token_freqs_cis = freqs_cis[cls_token_num:][token_order]  -> [bsz, block_size, hd//2, 2]
    train_freqs = freqs_cis[cls_token_num:][token_order]

    # Inference-time: index one step at a time
    for step in range(block_size):
        spatial_pos = token_order[:, step]  # [bsz]
        freqs_indices = cls_token_num + spatial_pos
        infer_freqs = freqs_cis[freqs_indices]  # [bsz, hd//2, 2]

        expected = train_freqs[:, step]  # [bsz, hd//2, 2]
        assert torch.equal(infer_freqs, expected), \
            f"Mismatch at step {step}!\n  infer: {infer_freqs[0,:3]}\n  train: {expected[0,:3]}"

    print("  PASSED: inference freqs_cis indexing matches training\n")


def test_future_peek_consistency():
    """
    Verify that the future peek freqs during inference match what
    the training-time shift would produce.
    """
    print("Test 5: Future peek consistency")
    from autoregressive.models.gpt_position_order import precompute_freqs_cis_2d

    grid_size = 4
    block_size = grid_size ** 2
    head_dim = 64
    cls_token_num = 1

    freqs_cis = precompute_freqs_cis_2d(grid_size, head_dim, 10000, cls_token_num)

    bsz = 2
    token_order = torch.stack([torch.randperm(block_size) for _ in range(bsz)])

    # Training-time reordered freqs (image tokens only)
    train_freqs = freqs_cis[cls_token_num:][token_order]  # [bsz, block_size, hd//2, 2]

    # In training, the "future" freq at position i is train_freqs[:, i+1]
    # (with zero/ones for the last position)
    for step in range(block_size - 1):
        # Inference computation for future peek
        next_spatial_pos = token_order[:, step + 1]
        next_freqs_indices = cls_token_num + next_spatial_pos
        infer_future = freqs_cis[next_freqs_indices]  # [bsz, hd//2, 2]

        # Training: shifted freq at this step (within image tokens only)
        train_future = train_freqs[:, step + 1]

        assert torch.equal(infer_future, train_future), \
            f"Future peek mismatch at step {step}!"

    print("  PASSED: future peek freqs match training shift\n")


if __name__ == "__main__":
    test_inverse_permutation()
    test_identity_order_is_noop()
    test_seed_determinism()
    test_freqs_cis_indexing()
    test_future_peek_consistency()
    print("All tests passed!")
