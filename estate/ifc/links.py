"""LINKED_MODEL document references: how SITE.ifc federates the building IFCs for Bonsai.

Bonsai's ``bim.link_ifc`` records a link in the host IFC as one ``IfcDocumentInformation(Scope="LINKED_MODEL")``
per linked file plus an ``IfcDocumentReference`` whose ``Location`` is the linked file's path relative to the host
IFC's folder (forward slashes) and whose ``Identification`` holds a 4x4 matrix as 16 comma-separated numbers,
row-major (parsed with ``np.fromstring(s, sep=",").reshape(4, 4)``). When Bonsai loads the host these references
appear as ``scene.BIMProjectProperties.links``, ready for ``bim.load_link``. This module writes that encoding so
the host is authored by ifcopenshell alone (bonsai/bim/module/project/operator.py LinkIfc._execute,
tool/project.py load_linked_models_from_ifc and calculate_link_matrix).

Bonsai places a loaded link at ``inv(G_host) @ T @ G_link``, where ``T`` is the Identification matrix and ``G`` is a
file's map frame: rotation by its map conversion's x-axis angle, translation to the map coordinates of its local
origin. So ``T`` acts in MAP coordinates: (1) every federated file must carry the same IfcMapConversion, otherwise a
link lands offset by the difference of the origins (a building without georeferencing lands at (-E, -N, -H));
(2) a placement ``M`` given in the host's local engineering coordinates is stored as ``G @ M @ inv(G)``, which is
what ``add_linked_models`` does. Our building IFCs are already placed in estate coordinates, so ``M`` is normally
the identity and is stored as the identity.
"""
from __future__ import annotations

import math
from pathlib import PurePosixPath, PureWindowsPath

import numpy as np

import ifcopenshell.api.document as document
import ifcopenshell.util.geolocation as geolocation
import ifcopenshell.util.unit as uunit

SCOPE = "LINKED_MODEL"


def encode_matrix(matrix=None) -> str:
    """Bonsai's Identification string: 16 row-major numbers joined by commas (identity when None)."""
    m = np.eye(4) if matrix is None else np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    return ",".join(str(float(v)) for v in m.flatten().tolist())


def decode_matrix(text) -> np.ndarray:
    if not text or text == "X":
        return np.eye(4)
    return np.array([float(v) for v in str(text).split(",")], dtype=np.float64).reshape(4, 4)


def map_frame(f) -> np.ndarray:
    """Bonsai's model-origin matrix for a file: local origin -> map coordinates (SI), rotated to grid north."""
    e, n, h = geolocation.auto_xyz2enh(f, 0.0, 0.0, 0.0, should_return_in_map_units=False)
    s = uunit.calculate_unit_scale(f)
    a = math.radians(-geolocation.get_grid_north(f))
    G = np.eye(4)
    G[:2, :2] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
    G[:3, 3] = [e * s, n * s, h * s]
    return G


def to_identification(f, matrix=None) -> np.ndarray:
    """Local placement (host engineering coordinates) -> the map-frame matrix Bonsai expects."""
    if matrix is None:
        return np.eye(4)
    M = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    if np.allclose(M, np.eye(4)):
        return np.eye(4)
    G = map_frame(f)
    T = G @ M @ np.linalg.inv(G)
    T[np.abs(T) < 1e-9] = 0.0
    return T


def from_identification(f, T) -> np.ndarray:
    G = map_frame(f)
    return np.linalg.inv(G) @ np.asarray(T) @ G


def _location(path) -> str:
    p = str(path).replace("\\", "/")
    if PureWindowsPath(p).is_absolute() or PurePosixPath(p).is_absolute():
        raise ValueError(f"linked model path must be relative to the host IFC folder: {path!r}")
    return PurePosixPath(p).as_posix()


def add_linked_models(W, entries) -> list:
    """Add LINKED_MODEL references to the host file of IfcWriter W (or a bare ifcopenshell.file).

    ``entries``: dicts with ``path`` (relative to the host IFC's folder, forward slashes), ``name`` (label) and
    optional ``matrix`` (4x4 placement in the host's local coordinates, identity by default; converted to Bonsai's
    map-frame Identification). Georeference the host (IfcMapConversion) BEFORE calling this. Entries sharing a path
    share one IfcDocumentInformation, like Bonsai. Returns the created IfcDocumentReferences in entry order.
    """
    f = getattr(W, "m", W)
    id_attr = "DocumentId" if f.schema == "IFC2X3" else "Identification"
    existing = {}
    for info in f.by_type("IfcDocumentInformation"):
        if info.Scope == SCOPE:
            refs = info.DocumentReferences if f.schema == "IFC2X3" else info.HasDocumentReferences
            for r in refs or []:
                existing.setdefault(r.Location, info)
    out = []
    for e in entries:
        loc = _location(e["path"])
        label = e.get("name") or PurePosixPath(loc).stem
        info = existing.get(loc)
        if info is None:
            info = document.add_information(f)      # also associates it with the IfcProject
            setattr(info, id_attr, label)
            info.Name = PurePosixPath(loc).name      # Bonsai names the document after the file
            info.Scope = SCOPE
            if f.schema != "IFC2X3":
                info.Location = loc
            info.Description = e.get("description") or f"Linked model {label}"
            info.Purpose = "Federated building model"
            existing[loc] = info
        ref = document.add_reference(f, information=info)
        ident = encode_matrix(to_identification(f, e.get("matrix")))
        if f.schema == "IFC2X3":
            ref.ItemReference = ident
        else:
            ref.Identification = ident     # no Name: WR1 allows Name XOR ReferencedDocument (label lives on info)
        ref.Location = loc
        out.append(ref)
    return out


def linked_models(f) -> list[dict]:
    """Read back a host's LINKED_MODEL references: [dict(path, name, matrix (local), identification (raw), document)]."""
    out = []
    for info in f.by_type("IfcDocumentInformation"):
        if info.Scope != SCOPE:
            continue
        refs = info.DocumentReferences if f.schema == "IFC2X3" else info.HasDocumentReferences
        for r in refs or []:
            T = decode_matrix(r.ItemReference if f.schema == "IFC2X3" else r.Identification)
            out.append(dict(path=r.Location, name=getattr(r, "Name", None) or info.Name, identification=T,
                            matrix=from_identification(f, T), document=info.Name))
    return sorted(out, key=lambda d: (d["path"], d["name"] or ""))
