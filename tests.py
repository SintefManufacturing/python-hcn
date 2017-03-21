import unittest
import math

import numpy as np
import math3d as m3d
from IPython import embed

import hcn

try:
    import vtk_visualizer as vv
except:
    pass


class TestsHTuple(unittest.TestCase):
    def test_tuple_double(self):
        val = 0.199
        d = hcn.HTuple(val)
        self.assertEqual(d.length(), 1)
        a = np.array([val])
        self.assertEqual(d.to_array(), a)
        self.assertEqual(d[0], val)
        self.assertEqual(d.type(), hcn.TupleType.Double)

    def test_tuple_int(self):
        val = 7
        d = hcn.HTuple(val)
        self.assertEqual(d.length(), 1)
        a = np.array([val])
        self.assertEqual(d.to_array(), a)
        self.assertEqual(d[0], val)
        self.assertEqual(d.type(), hcn.TupleType.Int)

    def test_tuple_bytes(self):
        val = b"toto is None"
        d = hcn.HTuple(val)
        self.assertEqual(d.length(), 1)
        a = [val]
        self.assertEqual(d.to_list(), a)
        self.assertEqual(d[0], val)
        self.assertEqual(d.type(), hcn.TupleType.String)

    def test_to_from_bytes(self):
        l = ["1", "2", "3", "b"]

    def test_append(self):
        val = 0.199
        d = hcn.HTuple(val)
        d.append(3.4)
        self.assertEqual(d.length(), 2)

    def test_array_double(self):
        a = np.array([1.1, 2.1, 3.1, 4.1, 5.1], dtype=np.double)
        t = hcn.HTuple.from_array(a)
        self.assertEqual(t.length(), 5)
        self.assertEqual(t[0], a[0])
        self.assertEqual(t[4], t[4])

    def test_array_int(self):
        a = np.array([1, 2, 3, 4, 5], dtype=np.int)
        t = hcn.HTuple.from_array(a)
        self.assertEqual(t.length(), 5)
        self.assertEqual(t[0], t[0])
        self.assertEqual(t[4], t[4])

    def test_mixed(self):
        l = [1, 2.5, b"jkl", "oøl"]
        tup = hcn.HTuple.from_list(l)
        self.assertEqual(tup.length(), 4)
        self.assertEqual(tup[0], 1)
        self.assertEqual(tup[2], b"jkl")
        with self.assertRaises(ValueError):
            print(tup[5])


class TestsHPose(unittest.TestCase):
    def test_empty(self):
        p = hcn.HPose()

    def test_transform(self):
        p = hcn.HPose()
        t = m3d.Transform()
        t.pos = m3d.Vector(2, 3, 1)
        t.orient.rotate_zb(math.pi / 2)
        p = hcn.HPose(*t.pose_vector)
        t2 = m3d.Transform(p.to_list()[:-1])
        self.assertEqual(t, t2)


class TestsModel3D(unittest.TestCase):

    def test_read_model(self):
        m = hcn.Model3D.from_file("simple.obj", "m")
        ar = m.to_array()
        np.testing.assert_array_equal(ar[0], np.array([1, 1, 1]))
        np.testing.assert_array_equal(ar[3], np.array([1, 1, 2]))

    def test_to_from_array_model(self):
        ar = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.double)
        m = hcn.Model3D.from_array(ar)
        new = m.to_array()
        np.testing.assert_array_equal(ar, new)

    def test_bounding_box(self):
        print("A")
        m = self._get_simple_model()
        print("B")
        m.get_bounding_box()
        #print("BOUND", m.get_bounding_box())
        #FIXME: check result

    def test_convex_hull(self):
        m = self._get_simple_model()
        ch = m.get_convex_hull()

    def test_select(self):
        m = self._get_simple_model()
        m = m.select_x(0, 1)
        ar = m.to_array()
        self.assertTrue(max(ar[:, 0]) <= 1)

    def test_create_surface(self):
        m = self._get_simple_model()
        #s = m.create_surface_model(0.1)

    def test_sample(self):
        m = self._get_simple_model()
        new = m.sampled("fast", 2)
        self.assertEqual(len(new.to_array()), 2)

    def test_smooth(self):
        m = self._get_simple_model()
        new = m.smoothed(knn=200)
        #self.assertEqual(len(new.to_array()), 2)

    def test_exception(self):
        m = self._get_simple_model()
        with self.assertRaises(RuntimeError):
            m.to_file("obj", "/t.obj")

    def _get_simple_model(self):
        ar = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0.1], [3, 0, -0.1]], dtype=np.double)
        return hcn.Model3D.from_array(ar)

    def test_box(self):
        p = hcn.HPose(1, 2, 3)
        b = hcn.Box(p, 1, 2, 0.5)
        n = b.sampled("fast", 0.01)
        #embed()

    def test_plane(self):
        pose = hcn.HPose(0, 0, 2)
        p = hcn.Plane(pose, [1, 0, 0], [0, 1, 0])
        n = p.sampled("fast", 0.1)

    def test_sphere(self):
        s = hcn.Sphere(1, 2, 3, 2)
        n = s.sampled("fast", 0.1)

    def test_normals(self):
        m = hcn.Model3D.from_file("simple.obj", "m")
        #print("F", m.normals_to_array())
        m.compute_normals(60, 2)
        #print("E", m.normals_to_array())
        # FIXME

    def test_localize(self):
        box = hcn.Box(hcn.HPose(0.1, 0.2, -0.05), 0.1, 0.2, 0.3)
        # make a scene and sample it
        trans = m3d.Transform((0.2, 0, 0, 0, 0, math.pi / 2))
        scene = box.transformed(hcn.HPose(trans))
        scene = scene.sampled("fast_compute_normals", 0.01)
        #scene = scene.select_z(0, 1)  # FIXME: does not work
        # sample our box to something different
        box = box.sampled("fast_compute_normals", 0.01)
        surf = box.create_surface_model(0.02)
        poses, score = surf.find_surface_model(scene, 0.001, 0.2, min_score=0, params={"num_matches":1})
        if poses:
            tr = box.transformed(poses[0])
        self.assertGreater(len(poses), 1)

    def test_segment(self):
        box = hcn.Box(hcn.HPose(0.1, 0.2, -0.05), 0.1, 0.2, 0.3)
        box = box.sampled("fast_compute_normals", 0.001)
        #box = hcn.Model3D.from_file("KA/punktskyer/ka1.ply", "mm", params={"xyz_map_width":2024})
        #box.prepare("segmentation")
        seg = box.segment2({"fitting":"true", "primitive_type":"plane", "fitting_algorithm":"least_squares"})
        #seg = box.segment({"primitive_type":"plane"})
        #seg = box.segment2({"fitting":"false", "output_xyz_mapping":"false"})
        #seg = box.segment()
        s = seg[0]
        embed()

    def test_fit(self):
        mod = hcn.Sphere(0.1, 0.2, 0.3, 0.1)
        mod = mod.sampled("fast", 0.01)
        results = mod.fit_primitive({"primitive_type":"all"})
        res = results[0]
        ptype = res.get_attribute("primitive_type").to_list()
        embed()
        self.assertEqual(ptype[0], b"sphere")

    def test_distance(self):
        box = hcn.Box(hcn.HPose(0.1, 0.2, -0.05), 0.1, 0.2, 0.3)
        box = box.sampled("fast", 0.1)
        sphere = hcn.Sphere(0.1, 0.2, -0.05, 0.2)
        sphere = sphere.sampled("fast", 0.1)
        box.distance(sphere)
        dists = box.get_attribute("&distance")
        self.assertEqual(box.to_array().shape[0], dists.length())



if __name__ == "__main__":
    unittest.main(verbosity=3)
