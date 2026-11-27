"""Tree species: each estate species has its own crown silhouette, within the web pack's per-mesh budget, as valid
closed shells that tessellate to exactly one Bark and one Foliage colour; the legacy block keeps its icosphere."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from estate.env import bootstrap  # noqa: E402

bootstrap()

import numpy as np  # noqa: E402

from estate.ifc import vegetation as V  # noqa: E402
from estate.site.mscp import GARDEN_SPECIES  # noqa: E402

ALL = V.ESTATE_SPECIES + GARDEN_SPECIES


def _edges(faces):
    out = {}
    for f in faces:
        for a, b in zip(f, f[1:] + f[:1]):
            out[(a, b)] = out.get((a, b), 0) + 1
    return out


def _winding(verts, faces, q):
    """Winding number of the closed shell about point q (1 inside, 0 outside): solid angles, Van Oosterom-Strackee."""
    total = 0.0
    for f in faces:
        for k in range(1, len(f) - 1):
            a, b, c = (verts[i] - np.asarray(q) for i in (f[0], f[k], f[k + 1]))
            la, lb, lc = np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c)
            total += 2 * np.arctan2(np.dot(a, np.cross(b, c)),
                                    la * lb * lc + np.dot(a, b) * lc + np.dot(a, c) * lb + np.dot(b, c) * la)
    return total / (4 * np.pi)


class Species(unittest.TestCase):
    def test_five_species_names_unchanged(self):
        self.assertEqual([s[0] for s in V.ESTATE_SPECIES],
                         ["Rain tree (young)", "Angsana", "Yellow flame", "Sea apple", "Tembusu"])
        self.assertEqual(V.SPECIES, V.ESTATE_SPECIES[:2])

    def test_triangle_budget(self):
        for name, h, rad in ALL:
            with self.subTest(name):
                self.assertLessEqual(V.triangle_count(name, h, rad), V.MAX_TREE_TRIANGLES)

    def test_closed_outward_planar_shells(self):
        for name, h, rad in ALL:
            parts, _ = V.crown_parts(name, h, rad)
            for i, (verts, faces) in enumerate(parts):
                with self.subTest(name, shell=i):
                    e = _edges([tuple(f) for f in faces])
                    # every directed edge once, its reverse once: closed, consistently oriented, manifold
                    self.assertTrue(all(n == 1 and e.get((b, a)) == 1 for (a, b), n in e.items()))
                    # Euler characteristic of a sphere
                    self.assertEqual(len(verts) - len(e) // 2 + len(faces), 2)
                    # outward: positive enclosed volume (divergence theorem over the fan of each face)
                    vol = 0.0
                    for f in faces:
                        p = verts[list(f)]
                        for k in range(1, len(f) - 1):
                            vol += np.dot(p[0], np.cross(p[k], p[k + 1])) / 6
                    self.assertGreater(vol, 0)
                    # every quad is planar (the wobble scales whole meridians)
                    for f in faces:
                        if len(f) == 4:
                            p = verts[list(f)]
                            n = np.cross(p[1] - p[0], p[2] - p[0])
                            self.assertLess(abs(np.dot(n / np.linalg.norm(n), p[3] - p[0])), 1e-9)

    def test_crown_holds_the_trunk_top_and_keeps_its_size(self):
        for name, h, rad in ALL:
            with self.subTest(name):
                parts, top = V.crown_parts(name, h, rad)
                v = np.concatenate([p[0] for p in parts])
                reach = np.hypot(v[:, 0], v[:, 1]).max()
                self.assertTrue(0.8 * rad <= reach <= 1.2 * rad, reach / rad)
                self.assertGreater(v[:, 2].min(), 0.5 * h)          # clear head room under every crown
                # the trunk ends inside a crown shell, so no crown floats
                self.assertTrue(any(_winding(v, f, (0.0, 0.0, top)) > 0.5 for v, f in parts), (name, top))

    def test_silhouettes_differ(self):
        """Depth and reach over canopy radius, and shell count: no two species share a crown shape."""
        sig = {}
        for name, h, rad in V.ESTATE_SPECIES:
            parts, _ = V.crown_parts(name, h, rad)
            v = np.concatenate([p[0] for p in parts])
            sig[name] = (len(parts), round((v[:, 2].max() - v[:, 2].min()) / rad, 1),
                         round(np.hypot(v[:, 0], v[:, 1]).max() / rad, 1))
        self.assertEqual(len(set(sig.values())), 5, sig)
        self.assertEqual(sig["Yellow flame"][0], 3)       # three tiers
        self.assertEqual(sig["Sea apple"][0], 4)          # four lobes
        self.assertGreater(sig["Tembusu"][1] / sig["Tembusu"][2], 2.5)        # columnar
        self.assertLess(sig["Rain tree (young)"][1] / sig["Rain tree (young)"][2], 1.0)   # flattened umbrella


class IfcTypes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from estate.ifc.writer import IfcWriter
        cls.W = IfcWriter(guid_key="test-vegetation", guid_base="SN5/TEST")
        cls.types = V.tree_types(cls.W, V.ESTATE_SPECIES)
        cls.W.flush()

    @classmethod
    def tearDownClass(cls):
        cls.W.close()

    def test_one_type_per_species_with_bark_and_foliage(self):
        import ifcopenshell.geom
        self.assertEqual([t.Name for t in self.types], [s[0] for s in V.ESTATE_SPECIES])
        s = ifcopenshell.geom.settings()
        for t, (name, h, rad) in zip(self.types, V.ESTATE_SPECIES):
            with self.subTest(name):
                self.assertEqual(t.PredefinedType, "VEGETATION")
                rep = t.RepresentationMaps[0].MappedRepresentation
                names = {it.StyledByItem[0].Styles[0].Name for it in rep.Items}
                self.assertEqual(names, {"Bark", "Foliage"})
                g = ifcopenshell.geom.create_shape(s, rep)
                faces = np.array(g.faces).reshape(-1, 3)
                self.assertEqual(len(faces), V.triangle_count(name, h, rad))
                from estate.export.meshcache import _colour
                from estate.rules import PALETTE
                self.assertEqual(sorted(tuple(round(c, 3) for c in _colour(mm)[:3]) for mm in g.materials),
                                 sorted(PALETTE[k][1] for k in ("bark", "foliage")))

    def test_legacy_style_is_the_toolkit_icosphere(self):
        from estate.ifc.writer import IfcWriter
        W = IfcWriter(guid_key="test-vegetation-legacy", parametric=False)
        try:
            (rain, _) = V.tree_types(W, style="legacy")
            rep = rain.RepresentationMaps[0].MappedRepresentation
            trunk, crown = rep.Items
            self.assertEqual(trunk.Depth, 3.0)
            self.assertEqual(len(crown.Outer.CfsFaces), 80)
        finally:
            W.close()


if __name__ == "__main__":
    unittest.main()
