"""Trained MLX26 input order, selected by key from the ML26 descriptors.

The shared evaluator retains its Rung3.5 and VV10 definitions. Its independent
HFX feature is excluded here and added by the post-SCF driver's hybrid path.
"""


MLX26_FEATURE_KEYS = (
    ("lak_x", "lak_c")
    + tuple(f"lyp_t{i}" for i in range(1, 7))
    # I09-I66: 40 MN15 cc terms followed by nine a and nine b terms.
    + tuple(
        f"mn15_cc{i}{j}{k}"
        for i, count in enumerate((6, 5, 4, 3))
        for j, width in enumerate(range(count, 2, -1))
        for k in range(width)
    )
    + tuple(f"mn15_a{i}" for i in range(9))
    + tuple(f"mn15_b{i}" for i in range(9))
    # I67-I74: local-hybrid then B05 for each beta in (0.1, 0.2, 1.0, 3.0).
    + tuple(
        f"r35_2_beta{tag}_{term}"
        for tag in ("0p1", "0p2", "1p0", "3p0")
        for term in ("local_hybrid", "b05")
    )
    + ("cs1_same_spin_1", "cs1_same_spin_2", "cs1_opposite_spin", "vv10")
)


__all__ = ["MLX26_FEATURE_KEYS"]
