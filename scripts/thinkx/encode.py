"""Canonical cross-phase encoder + BOS row writer (shared by all miners)."""

BOS = 8188


class FallbackEncoder:
    """ASCII -> raw byte (0-127), non-ASCII -> 256+b per UTF-8 byte.

    Matches code_python.txt, which Phase 1 trained on. Never prefer BPE.
    """

    def encode(self, text):
        ids = []
        for ch in text:
            if ord(ch) < 256:
                ids.append(ord(ch))
            else:
                for b in ch.encode("utf-8"):
                    ids.append(256 + b)
        return ids


enc = FallbackEncoder()


def write_rows(rows, out):
    """Write [(text)] as BOS-prefixed id rows; return kept count."""
    import hashlib
    import os
    seen, kept = set(), 0
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".tmp", "w") as f:
        for text in rows:
            h = hashlib.sha256(text.encode()).hexdigest()[:16]
            if h in seen:
                continue
            seen.add(h)
            ids = enc.encode(text)
            f.write(f"{BOS} " + " ".join(map(str, ids)) + "\n")
            kept += 1
    os.rename(out + ".tmp", out)
    return kept
