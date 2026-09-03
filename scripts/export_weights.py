#!/usr/bin/env python3
"""export_weights.py — torch-free PyTorch checkpoint exporter (stdlib only).

Reads a torch.save() .pt (zip + pickle) WITHOUT torch installed, using a
restricted Unpickler that stubs torch classes and materializes raw storage
bytes. Writes one flat float32 .npy per model weight + manifest.txt.

  /usr/bin/python3 scripts/export_weights.py <checkpoint.pt> <out_dir> [--audit]

--audit: print structure (keys, shapes, dtypes, config) without writing.
Requires: nothing but Python 3.8+ stdlib.

Output .npy files are 1-D float32 C-order flats — exactly the layout the
Fortran engine (src/lib) consumes via stdlib_io_npy::load_npy.
bfloat16 storages (wte) are converted to float32 on export.
"""
import array
import io
import pickle
import struct
import sys
import zipfile

# --------------------------------------------------------------------------
# torch class stubs: record construction args, build nothing torch-flavored.


class StorageType:
    """Marker for torch.*Storage classes carried in persistent ids."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"<StorageType {self.name}>"


# torch storage-type name -> (numpy-ish dtype tag, itemsize)
STORAGE_DTYPES = {
    "FloatStorage": ("f4", 4),
    "DoubleStorage": ("f8", 8),
    "HalfStorage": ("f2", 2),
    "BFloat16Storage": ("bf16", 2),
    "ByteStorage": ("u1", 1),
    "CharStorage": ("i1", 1),
    "ShortStorage": ("i2", 2),
    "IntStorage": ("i4", 4),
    "LongStorage": ("i8", 8),
    "BoolStorage": ("bool", 1),
    "UntypedStorage": ("raw", 1),
}


class TensorRecord:
    """A rebuilt tensor: dtype tag + shape + flat C-order raw bytes."""

    def __init__(self, dtype_tag, size, stride, storage_key):
        self.dtype_tag = dtype_tag
        self.size = tuple(size)
        self.stride = tuple(stride)
        self.storage_key = storage_key
        self.offset = 0

    def numel(self):
        n = 1
        for s in self.size:
            n *= s
        return n


class ParameterRecord(TensorRecord):
    pass


class TorchUnpickler(pickle.Unpickler):
    """Restricted unpickler: real builtins/collections, stubbed torch."""

    def __init__(self, *args, **kwargs):
        self.zip = kwargs.pop("zipfile_obj")
        self.prefix = kwargs.pop("prefix")
        self.storages = {}  # key -> (dtype_tag, raw bytes)
        super().__init__(*args, **kwargs)

    def find_class(self, module, name):
        if module in ("builtins", "__builtin__"):
            return getattr(__builtins__ if isinstance(__builtins__, dict) else __builtins__, name) \
                if False else __import__("builtins").__dict__[name]
        if module == "collections" and name == "OrderedDict":
            import collections
            return collections.OrderedDict
        if module == "torch":
            if name in STORAGE_DTYPES or name == "UntypedStorage":
                return StorageType(name)
            if name == "device":
                return lambda *a: a[0] if a else "cpu"
            if name == "dtype":
                return lambda *a: a[0] if a else None
            raise pickle.UnpicklingError(f"stub torch.{name} on demand")
        if module == "torch._utils":
            fn = getattr(self, name, None)
            if fn is None:
                raise pickle.UnpicklingError(f"no stub for torch._utils.{name}")
            return fn
        raise pickle.UnpicklingError(f"blocked import {module}.{name}")

    # -- torch._utils rebuild functions (record, don't build) ----------------
    def _rebuild_tensor_v2(self, storage, storage_offset, size, stride,
                           requires_grad, backward_hooks=None, metadata=None):
        rec = TensorRecord(storage[0], size, stride, storage[1])
        rec.offset = storage_offset
        return rec

    def _rebuild_parameter(self, data, requires_grad, backward_hooks=None):
        if isinstance(data, TensorRecord):
            return ParameterRecord(data.dtype_tag, data.size,
                                   data.stride, data.storage_key)
        return data

    def _rebuild_parameter_with_state(self, *a):
        return self._rebuild_parameter(*a[:2])

    def _rebuild_meta_tensor_no_storage(self, *a):
        return TensorRecord("meta", (), (), None)

    # -- storages arrive through persistent ids ------------------------------
    def persistent_load(self, pid):
        if not (isinstance(pid, tuple) and pid and pid[0] == "storage"):
            raise pickle.UnpicklingError(f"unsupported persistent id {pid!r}")
        _, storage_type, key, location, numel = pid
        dtype_tag = STORAGE_DTYPES.get(
            storage_type.name if isinstance(storage_type, StorageType)
            else str(storage_type), ("raw", 1))[0]
        raw = self.zip.read(f"{self.prefix}/data/{key}")
        self.storages[key] = (dtype_tag, raw)
        return (dtype_tag, key)


# --------------------------------------------------------------------------
# minimal .npy v1.0 writer (C-order), no numpy.


def write_npy_1d_f32(path, flat_floats):
    """flat_floats: iterable of Python floats -> 1-D '<f4' .npy file."""
    n = len(flat_floats)
    header = "{'descr': '<f4', 'fortran_order': False, 'shape': (%d,)}" % n
    header += " " * (64 - (10 + len(header) + 1) % 64) + "\n"
    try:
        with open(path, "wb") as f:
            f.write(b"\x93NUMPY\x01\x00")
            f.write(struct.pack("<H", len(header)))
            f.write(header.encode("latin1"))
            buf = array.array("f", flat_floats)
            if sys.byteorder != "little":
                buf.byteswap()
            buf.tofile(f)
    except OSError as e:
        raise SystemExit(f"cannot write {path}: {e}")


def bf16_to_f32_list(raw):
    """bytes of little-endian bf16 -> list of Python floats."""
    n = len(raw) // 2
    h = array.array("H", raw[: 2 * n])
    out = array.array("I", [0]) * 0
    # widen in C: build uint32 via bytes? do shift in array ops is python-level;
    # use struct iteration in chunks for speed.
    res = [0.0] * n
    for i in range(0, n, 65536):
        blk = h[i: i + 65536]
        wide = array.array("I", (v << 16 for v in blk))
        res[i: i + len(blk)] = struct.unpack("<%df" % len(blk), wide.tobytes())
    return res


def storage_to_f32_list(dtype_tag, raw, rec):
    """Materialize a (possibly strided) tensor record as flat float list."""
    if dtype_tag == "bf16":
        vals = bf16_to_f32_list(raw)
    elif dtype_tag == "f4":
        vals = list(array.array("f", raw))
        if sys.byteorder != "little":
            raise ValueError("big-endian host unsupported")
    elif dtype_tag == "f2":
        vals = list(array.array("e", raw)) if hasattr(array, "e") else None
        if vals is None:
            raise ValueError("float16 storage needs python>=3.11 array('e')")
    else:
        raise ValueError(f"unsupported storage dtype {dtype_tag}")
    # apply storage offset (in elements) and stride -> C-order flat
    size, stride, off = rec.size, rec.stride, rec.offset
    if not size:
        return vals[off: off + 1]
    # default C-contiguous strides?
    exp, n = 1, 1
    ok = True
    for s in reversed(size):
        n *= s
    # check contiguity: stride[i] == prod(size[i+1:])
    acc = 1
    for s, st in zip(reversed(size), reversed(stride)):
        if st != acc:
            ok = False
            break
        acc *= s
    if ok:
        return vals[off: off + n]
    # general strided gather (slow path, rare for state_dict weights)
    idx = [0] * len(size)
    out = []
    for _ in range(n):
        flat = off + sum(a * b for a, b in zip(idx, stride))
        out.append(vals[flat])
        for d in range(len(size) - 1, -1, -1):
            idx[d] += 1
            if idx[d] < size[d]:
                break
            idx[d] = 0
    return out


# --------------------------------------------------------------------------


def load_checkpoint(path):
    z = zipfile.ZipFile(path)
    names = z.namelist()
    prefix = next(n for n in names if n.endswith("/data.pkl")).rsplit("/", 1)[0]
    pkl = z.read(prefix + "/data.pkl")
    up = TorchUnpickler(io.BytesIO(pkl), zipfile_obj=z, prefix=prefix)
    obj = up.load()
    return obj, up.storages


def main():
    audit = "--audit" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ckpt, outdir = args[0], args[1] if len(args) > 1 else "weights"
    obj, storages = load_checkpoint(ckpt)

    print("top-level keys:", list(obj.keys()))
    cfg = obj.get("config", {})
    print("config:", {k: cfg[k] for k in
                      ("n_embd", "n_head", "n_kv_head", "n_layer",
                       "vocab_size", "sequence_len") if k in cfg})
    print("storages:", len(storages),
          {t for t, _ in storages.values()})

    msd = obj["model_state_dict"]
    print("model tensors:", len(msd))
    for k, v in msd.items():
        if isinstance(v, TensorRecord):
            print(f"  {k} shape={v.size} dtype={v.dtype_tag}")
        else:
            print(f"  {k} <{type(v).__name__}>")

    if audit:
        return
    import os
    try:
        os.makedirs(outdir, exist_ok=True)
    except OSError as e:
        raise SystemExit(f"cannot create {outdir}: {e}")
    manifest = []
    for k, v in msd.items():
        if not isinstance(v, TensorRecord) or v.dtype_tag == "meta":
            continue
        dtype_tag, raw = storages[v.storage_key]
        # UntypedStorage: dtype decided by tensor record context (assume f4)
        if dtype_tag == "raw":
            dtype_tag = "f4"
        flat = storage_to_f32_list(dtype_tag, raw, v)
        name = k.replace(".", "_") + ".npy"
        write_npy_1d_f32(os.path.join(outdir, name), flat)
        manifest.append(f"{name} {v.size} {len(flat)}")
        print(f"wrote {name} n={len(flat)}")
    try:
        with open(os.path.join(outdir, "manifest.txt"), "w") as f:
            f.write("\n".join(manifest) + "\n")
    except OSError as e:
        raise SystemExit(f"cannot write manifest: {e}")
    print("manifest.txt written")


if __name__ == "__main__":
    main()
