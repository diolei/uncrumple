"""mulberry32, ported exactly (verified value-for-value against the
reference implementation). The dataset draws must come from the same
stream the frozen seeds imply: same seed, same floats, same
coefficient pairs."""

MASK = 0xFFFFFFFF


def _imul(a: int, b: int) -> int:
    """Low 32 bits of the product (matches JS Math.imul bit pattern)."""
    return (a * b) & MASK


def _to_int32(x: int) -> int:
    x &= MASK
    return x - 0x100000000 if x >= 0x80000000 else x


def mulberry32(seed: int):
    a = seed & MASK

    def rand() -> float:
        nonlocal a
        a = (a + 0x6D2B79F5) & MASK
        t = a
        t = _imul(t ^ (t >> 15), t | 1)
        # JS: t ^= t + Math.imul(...). The + is float64, the ^ applies
        # ToInt32 to both sides — i.e. XOR of the low-32-bit patterns.
        s = (t + _to_int32(_imul(t ^ (t >> 7), t | 61))) & MASK
        t = (t ^ s) & MASK
        return ((t ^ (t >> 14)) & MASK) / 4294967296

    return rand
