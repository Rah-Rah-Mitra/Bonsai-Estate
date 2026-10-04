"""Minimal glTF 2.0 binary (.glb) writer and reader (numpy + struct + json), replacing trimesh.

trimesh is not available in Blender's bundled Python, and the engine export only needs static triangle meshes:
POSITION (float32, with min/max), optional NORMAL and TEXCOORD_0, uint32 indices, one primitive per material,
factor-only PBR materials (no textures), named nodes with optional matrix/translation and a single scene.
Every buffer view starts on a 4-byte boundary, the JSON chunk is padded with spaces and the BIN chunk with zeros.

Triangle soups are welded per primitive (identical position + normal + UV) and cleaned of degenerate and
duplicate triangles. Blender's importer runs ``mesh.validate()``, which would silently drop such faces, so
cleaning them here keeps the triangle counts identical after a round trip.

A mesh may be referenced by any number of nodes (instancing: doors and furniture of one type, trees); triangle and
vertex counts in ``stats`` and ``summary`` are per node, i.e. what an engine draws, and ``meshes`` counts the distinct
ones.

The reader (``read_glb``, ``summary``, ``validate``) exists for round-trip tests and the export checks.
glTF is Y up: ``to_yup`` converts IFC (x, y, z) Z-up coordinates to (x, z, -y).
"""
from __future__ import annotations

import json
import re
import struct
from pathlib import Path

import numpy as np

MAGIC = 0x46546C67            # b"glTF"
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942
FLOAT, UINT = 5126, 5125
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963
GENERATOR = "Sample Town N5 estate exporter (estate/export/glb.py)"

LEAF_RE = re.compile(r"_leaf(_[A-Z0-9]+)?$")     # DOOR_..._leaf, _leaf_L / _leaf_R: door leaves under their door


def node_counts(names) -> dict:
    """DOOR_ (doors, leaves excluded), door leaf, WIN_, LIFT_, TREE_ and FURN_ node counts of a list of node names."""
    names = list(names)
    leaves = sum(1 for n in names if n.startswith("DOOR_") and LEAF_RE.search(n))
    return dict(door_nodes=sum(1 for n in names if n.startswith("DOOR_")) - leaves, door_leaves=leaves,
                window_nodes=sum(1 for n in names if n.startswith("WIN_")),
                lift_nodes=sum(1 for n in names if n.startswith("LIFT_")),
                tree_nodes=sum(1 for n in names if n.startswith("TREE_")),
                furniture_nodes=sum(1 for n in names if n.startswith("FURN_")))


# IFC Z-up -> glTF Y-up: (x, y, z) -> (x, z, -y); a proper rotation (det +1)
YUP = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
ZUP = YUP.T


def to_yup(v):
    v = np.asarray(v, dtype=float)
    return np.stack([v[..., 0], v[..., 2], -v[..., 1]], axis=-1)


def from_yup(v):
    v = np.asarray(v, dtype=float)
    return np.stack([v[..., 0], -v[..., 2], v[..., 1]], axis=-1)


def matrix_to_yup(M):
    """A Z-up 4x4 transform expressed in the Y-up frame (C M C^-1)."""
    return YUP @ np.asarray(M, dtype=float) @ ZUP


# ----------------------------------------------------------------------------- mesh preparation
def face_normals(tris):
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    ln = np.linalg.norm(n, axis=1)
    return n, ln


def box_uvs(tris_zup):
    """Box-projected UVs per corner (dominant normal axis), 1 UV unit per metre, in the Z-up frame.

    Vertical faces: u along the face, v = -z (glTF v runs down, so textures stand upright); horizontal faces:
    (x, -y), i.e. the glTF (x, z) plane."""
    tris_zup = np.asarray(tris_zup, dtype=float)
    n, _ = face_normals(tris_zup)
    ax = np.argmax(np.abs(n), axis=1)
    v = tris_zup.reshape(-1, 3)
    a = np.repeat(ax, 3)[:, None]
    uv = np.where(a == 0, np.column_stack([v[:, 1], -v[:, 2]]),
                  np.where(a == 1, np.column_stack([v[:, 0], -v[:, 2]]), np.column_stack([v[:, 0], -v[:, 1]])))
    return uv.reshape(-1, 3, 2)


def soup(tris, uvs=None, normals=True, min_area=1e-8):
    """Triangle soup (n, 3, 3) in the output frame -> welded primitive arrays.

    Returns dict(positions, normals, uvs, indices, kept) where ``kept`` indexes the input triangles that survive
    (degenerate and exact duplicate triangles are removed). Normals are flat face normals."""
    tris = np.asarray(tris, dtype=float).reshape(-1, 3, 3)
    n, ln = face_normals(tris)
    keep = np.nonzero(0.5 * ln > min_area)[0]
    tris, n, ln = tris[keep], n[keep], ln[keep]
    cols = [tris.reshape(-1, 3).astype(np.float32)]
    if normals:
        nq = np.round(n / ln[:, None], 5)
        nq /= np.linalg.norm(nq, axis=1)[:, None]
        cols.append(np.repeat(nq, 3, axis=0).astype(np.float32))
    if uvs is not None:
        cols.append(np.asarray(uvs, dtype=float).reshape(-1, 3, 2)[keep].reshape(-1, 2).astype(np.float32))
    rows = np.ascontiguousarray(np.hstack(cols)) + np.float32(0.0)     # + 0 folds -0.0 into 0.0 so they weld
    if len(rows):
        key = rows.view(np.dtype((np.void, rows.dtype.itemsize * rows.shape[1]))).ravel()
        _, first, inv = np.unique(key, return_index=True, return_inverse=True)
        order = np.argsort(first, kind="stable")          # keep first-seen vertex order (deterministic, cache friendly)
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        verts = rows[first[order]]
        idx = rank[inv.ravel()].reshape(-1, 3)
    else:
        verts, idx = rows, np.zeros((0, 3), np.int64)
    ok = (idx[:, 0] != idx[:, 1]) & (idx[:, 1] != idx[:, 2]) & (idx[:, 0] != idx[:, 2])
    srt = np.sort(idx, axis=1)
    if len(srt):
        _, uniq = np.unique(srt, axis=0, return_index=True)
        dup = np.ones(len(idx), bool)
        dup[uniq] = False
        ok &= ~dup
    idx, keep = idx[ok], keep[ok]
    used = np.unique(idx)
    if len(used) != len(verts):                            # drop vertices orphaned by removed triangles
        remap = np.full(len(verts), -1, np.int64)
        remap[used] = np.arange(len(used))
        verts, idx = verts[used], remap[idx]
    out = dict(positions=verts[:, 0:3], normals=None, uvs=None, indices=idx.astype(np.uint32).ravel(), kept=keep)
    c = 3
    if normals:
        out["normals"], c = verts[:, c:c + 3], c + 3
    if uvs is not None:
        out["uvs"] = verts[:, c:c + 2]
    return out


# ----------------------------------------------------------------------------- writer
class GlbWriter:
    """Accumulate materials, meshes and nodes; ``write`` packs one BIN buffer and the JSON."""

    def __init__(self, generator=GENERATOR, extras=None):
        self.asset = {"version": "2.0", "generator": generator}
        self.extras = extras
        self.materials, self.meshes, self.nodes, self.accessors, self.views = [], [], [], [], []
        self._mat_by_key = {}
        self._bin = bytearray()
        self.roots = []

    # ---- materials
    def material(self, name, rgba, roughness=0.8, metallic=0.0, double_sided=True, alpha_mode=None):
        rgba = [float(np.clip(c, 0.0, 1.0)) for c in rgba]
        rgba += [1.0] * (4 - len(rgba))
        mode = alpha_mode or ("BLEND" if rgba[3] < 0.99 else "OPAQUE")
        key = (name, tuple(round(c, 4) for c in rgba), mode)
        if key in self._mat_by_key:
            return self._mat_by_key[key]
        m = {"name": name, "pbrMetallicRoughness": {"baseColorFactor": [round(c, 4) for c in rgba],
                                                    "metallicFactor": float(metallic), "roughnessFactor": float(roughness)},
             "alphaMode": mode, "doubleSided": bool(double_sided)}
        self.materials.append(m)
        self._mat_by_key[key] = len(self.materials) - 1
        return self._mat_by_key[key]

    # ---- binary
    def _view(self, arr, target):
        arr = np.ascontiguousarray(arr)
        pad = (-len(self._bin)) % 4
        self._bin += b"\x00" * pad
        off = len(self._bin)
        self._bin += arr.tobytes()
        self.views.append({"buffer": 0, "byteOffset": off, "byteLength": arr.nbytes, "target": target})
        return len(self.views) - 1

    def _accessor(self, arr, kind, target, minmax=False):
        ctype = UINT if arr.dtype == np.uint32 else FLOAT
        acc = {"bufferView": self._view(arr, target), "componentType": ctype, "count": int(len(arr)), "type": kind}
        if minmax:
            acc["min"] = [float(x) for x in np.atleast_1d(arr.min(axis=0))]
            acc["max"] = [float(x) for x in np.atleast_1d(arr.max(axis=0))]
            if ctype == UINT:
                acc["min"], acc["max"] = [int(x) for x in acc["min"]], [int(x) for x in acc["max"]]
        self.accessors.append(acc)
        return len(self.accessors) - 1

    def mesh(self, name, primitives):
        """primitives: [(prim dict from ``soup``, material index)]; empty primitives are skipped.

        Returns the mesh index, or None when nothing is left."""
        prims = []
        for p, mat in primitives:
            if p is None or len(p["indices"]) == 0:
                continue
            attrs = {"POSITION": self._accessor(np.asarray(p["positions"], np.float32), "VEC3", ARRAY_BUFFER, True)}
            if p.get("normals") is not None:
                attrs["NORMAL"] = self._accessor(np.asarray(p["normals"], np.float32), "VEC3", ARRAY_BUFFER)
            if p.get("uvs") is not None:
                attrs["TEXCOORD_0"] = self._accessor(np.asarray(p["uvs"], np.float32), "VEC2", ARRAY_BUFFER)
            idx = self._accessor(np.asarray(p["indices"], np.uint32), "SCALAR", ELEMENT_ARRAY_BUFFER, True)
            prim = {"attributes": attrs, "indices": idx, "mode": 4}
            if mat is not None:
                prim["material"] = int(mat)
            prims.append(prim)
        if not prims:
            return None
        self.meshes.append({"name": name, "primitives": prims})
        return len(self.meshes) - 1

    # ---- nodes
    def node(self, name, mesh=None, matrix=None, translation=None, parent=None, extras=None):
        """Add a node; ``matrix`` is a row-major 4x4 in the output frame (written column-major).

        A node without parent is a scene root."""
        nd = {"name": name}
        if mesh is not None:
            nd["mesh"] = int(mesh)
        if matrix is not None and not np.allclose(matrix, np.eye(4), atol=1e-12):
            nd["matrix"] = [float(x) for x in np.asarray(matrix, float).T.ravel()]
        elif translation is not None and np.any(np.abs(translation) > 0):
            nd["translation"] = [float(x) for x in translation]
        if extras:
            nd["extras"] = extras
        self.nodes.append(nd)
        i = len(self.nodes) - 1
        if parent is None:
            self.roots.append(i)
        else:
            self.nodes[parent].setdefault("children", []).append(i)
        return i

    # ---- output
    def gltf(self, scene_name="Scene"):
        g = {"asset": dict(self.asset), "scene": 0, "scenes": [{"name": scene_name, "nodes": list(self.roots)}],
             "nodes": self.nodes}
        if self.extras:
            g["asset"]["extras"] = self.extras
        if self.meshes:
            g["meshes"] = self.meshes
        if self.materials:
            g["materials"] = self.materials
        if self.accessors:
            g["accessors"] = self.accessors
            g["bufferViews"] = self.views
            g["buffers"] = [{"byteLength": len(self._bin) + (-len(self._bin)) % 4}]
        return g

    def to_bytes(self, scene_name="Scene") -> bytes:
        js = json.dumps(self.gltf(scene_name), separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        js += b" " * ((-len(js)) % 4)
        binary = bytes(self._bin) + b"\x00" * ((-len(self._bin)) % 4)
        total = 12 + 8 + len(js) + (8 + len(binary) if binary else 0)
        out = [struct.pack("<III", MAGIC, 2, total), struct.pack("<II", len(js), CHUNK_JSON), js]
        if binary:
            out += [struct.pack("<II", len(binary), CHUNK_BIN), binary]
        return b"".join(out)

    def write(self, path, scene_name="Scene") -> dict:
        data = self.to_bytes(scene_name)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
        return self.stats(len(data))

    def stats(self, nbytes=None) -> dict:
        tris = {i: sum(self.accessors[p["indices"]]["count"] // 3 for p in m["primitives"])
                for i, m in enumerate(self.meshes)}
        mesh_nodes = [n for n in self.nodes if "mesh" in n]
        return dict(nodes=len(self.nodes), mesh_nodes=len(mesh_nodes), meshes=len(self.meshes),
                    materials=len(self.materials), triangles=int(sum(tris[n["mesh"]] for n in mesh_nodes)),
                    vertices=int(sum(self.accessors[p["attributes"]["POSITION"]]["count"]
                                     for n in mesh_nodes for p in self.meshes[n["mesh"]]["primitives"])),
                    bytes=nbytes)


# ----------------------------------------------------------------------------- reader
COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
DTYPES = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}


class GlbError(ValueError):
    pass


def read_glb(path):
    """Parse a .glb: returns (gltf dict, bin bytes). Raises GlbError on a malformed container."""
    data = Path(path).read_bytes()
    if len(data) < 20:
        raise GlbError("file too short")
    magic, version, length = struct.unpack_from("<III", data, 0)
    if magic != MAGIC or version != 2:
        raise GlbError(f"bad header magic={magic:#x} version={version}")
    if length != len(data):
        raise GlbError(f"header length {length} != file size {len(data)}")
    off, chunks = 12, []
    while off < len(data):
        clen, ctype = struct.unpack_from("<II", data, off)
        if clen % 4 or off % 4:
            raise GlbError(f"chunk at {off} not 4-byte aligned (length {clen})")
        chunks.append((ctype, data[off + 8: off + 8 + clen]))
        off += 8 + clen
    if not chunks or chunks[0][0] != CHUNK_JSON:
        raise GlbError("first chunk is not JSON")
    g = json.loads(chunks[0][1].decode("utf-8"))
    binary = next((c for t, c in chunks[1:] if t == CHUNK_BIN), b"")
    return g, binary


def accessor(g, binary, i):
    a = g["accessors"][i]
    v = g["bufferViews"][a["bufferView"]]
    n = COMPONENTS[a["type"]]
    dt = np.dtype(DTYPES[a["componentType"]])
    start = v.get("byteOffset", 0) + a.get("byteOffset", 0)
    arr = np.frombuffer(binary, dtype=dt, count=a["count"] * n, offset=start)
    return arr.reshape(a["count"], n) if n > 1 else arr


def node_matrix(nd):
    if "matrix" in nd:
        return np.array(nd["matrix"], float).reshape(4, 4).T
    M = np.eye(4)
    if "scale" in nd:
        M = np.diag(list(nd["scale"]) + [1.0]) @ M
    if "rotation" in nd:
        x, y, z, w = nd["rotation"]
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        T = np.eye(4)
        T[:3, :3] = R
        M = T @ M
    if "translation" in nd:
        T = np.eye(4)
        T[:3, 3] = nd["translation"]
        M = T @ M
    return M


def world_matrices(g):
    out = {}

    def walk(i, P):
        W = P @ node_matrix(g["nodes"][i])
        out[i] = W
        for c in g["nodes"][i].get("children", []):
            walk(c, W)

    for r in g["scenes"][g.get("scene", 0)]["nodes"]:
        walk(r, np.eye(4))
    return out


def summary(path, zup=True) -> dict:
    """Counts and world bounds of a glb (bounds converted back to Z-up unless zup=False)."""
    g, binary = read_glb(path)
    W = world_matrices(g)
    tris, names, lo, hi = 0, [], np.full(3, np.inf), np.full(3, -np.inf)
    per_node, node_bounds = {}, {}
    for i, nd in enumerate(g.get("nodes", [])):
        names.append(nd.get("name", ""))
        if "mesh" not in nd or i not in W:
            continue
        t, nlo, nhi = 0, np.full(3, np.inf), np.full(3, -np.inf)
        for p in g["meshes"][nd["mesh"]]["primitives"]:
            t += g["accessors"][p.get("indices", p["attributes"]["POSITION"])]["count"] // 3
            pos = accessor(g, binary, p["attributes"]["POSITION"]).astype(float)
            w = pos @ W[i][:3, :3].T + W[i][:3, 3]
            if zup:
                w = from_yup(w)
            nlo, nhi = np.minimum(nlo, w.min(axis=0)), np.maximum(nhi, w.max(axis=0))
        lo, hi = np.minimum(lo, nlo), np.maximum(hi, nhi)
        per_node[nd.get("name", str(i))] = t
        node_bounds[nd.get("name", str(i))] = [nlo.tolist(), nhi.tolist()]
        tris += t
    return dict(nodes=len(g.get("nodes", [])), mesh_nodes=len(per_node), triangles=int(tris),
                meshes=len(g.get("meshes", [])), materials=len(g.get("materials", [])), names=names,
                per_node=per_node, node_bounds=node_bounds, **node_counts(names),
                bounds=[[float(x) for x in lo], [float(x) for x in hi]] if per_node else None)


def validate(path) -> list[str]:
    """Spec checks used by the tests: alignment, accessor ranges and min/max, index bounds, unit normals."""
    problems = []
    try:
        g, binary = read_glb(path)
    except GlbError as e:
        return [str(e)]
    if g.get("asset", {}).get("version") != "2.0":
        problems.append("asset.version != 2.0")
    blen = g.get("buffers", [{}])[0].get("byteLength", 0) if g.get("buffers") else 0
    if blen > len(binary) or len(binary) - blen > 3:
        problems.append(f"buffer byteLength {blen} vs BIN chunk {len(binary)}")
    for k, v in enumerate(g.get("bufferViews", [])):
        if v.get("byteOffset", 0) % 4:
            problems.append(f"bufferView {k} offset {v.get('byteOffset')} not 4-byte aligned")
        if v.get("byteOffset", 0) + v["byteLength"] > blen:
            problems.append(f"bufferView {k} exceeds the buffer")
    for k, a in enumerate(g.get("accessors", [])):
        arr = accessor(g, binary, k)
        v = g["bufferViews"][a["bufferView"]]
        need = a["count"] * COMPONENTS[a["type"]] * np.dtype(DTYPES[a["componentType"]]).itemsize
        if need > v["byteLength"]:
            problems.append(f"accessor {k} overruns its bufferView")
        if "min" in a:
            mn = arr.min(axis=0) if arr.ndim > 1 else np.array([arr.min()])
            mx = arr.max(axis=0) if arr.ndim > 1 else np.array([arr.max()])
            if not (np.array_equal(np.float32(a["min"]), mn.astype(np.float32).ravel())
                    and np.array_equal(np.float32(a["max"]), mx.astype(np.float32).ravel())):
                problems.append(f"accessor {k} min/max do not match its data")
    for mi, m in enumerate(g.get("meshes", [])):
        for pi, p in enumerate(m["primitives"]):
            pos = g["accessors"][p["attributes"]["POSITION"]]
            if "min" not in pos or "max" not in pos:
                problems.append(f"mesh {mi} prim {pi}: POSITION without min/max")
            if "indices" in p:
                idx = accessor(g, binary, p["indices"])
                if len(idx) % 3:
                    problems.append(f"mesh {mi} prim {pi}: index count not a multiple of 3")
                if len(idx) and idx.max() >= pos["count"]:
                    problems.append(f"mesh {mi} prim {pi}: index out of range")
            for attr, acc_i in p["attributes"].items():
                if g["accessors"][acc_i]["count"] != pos["count"]:
                    problems.append(f"mesh {mi} prim {pi}: {attr} count differs from POSITION")
            if "NORMAL" in p["attributes"]:
                nrm = accessor(g, binary, p["attributes"]["NORMAL"]).astype(float)
                if len(nrm) and np.abs(np.linalg.norm(nrm, axis=1) - 1.0).max() > 1e-4:
                    problems.append(f"mesh {mi} prim {pi}: NORMAL not unit length")
            if "material" in p and p["material"] >= len(g.get("materials", [])):
                problems.append(f"mesh {mi} prim {pi}: material index out of range")
    names = [n.get("name") for n in g.get("nodes", [])]
    if len(set(names)) != len(names):
        problems.append("node names are not unique")
    return problems
