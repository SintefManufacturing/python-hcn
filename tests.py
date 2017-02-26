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
        m = self._get_simple_model()
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
        new_box = box.transformed(hcn.HPose(trans))
        box2 = hcn.Box(hcn.HPose(0, -0.2, -0.05), 0.3, 0.3, 0.3)
        sphere = hcn.Sphere(0, 0, -0.05, 0.2)
        print(1)
        new_box = new_box.sampled("fast_compute_normals", 0.02)
        box2 = box2.sampled("fast_compute_normals", 0.02)
        print(2)
        sphere = sphere.sampled("fast_compute_normals", 0.02)
        scene = sphere.union(new_box, sphere, box2)
        print(3)
        #scene = scene.select_z(0, 1)  # FIXME: does not work
        # sample our box to something different
        box = box.sampled("fast_compute_normals", 0.01)
        print(4)
        surf = box.create_surface_model(0.02)
        poses, score = surf.find_surface_model(scene, 0.04, 0.1, 0.01)
        print(5)
        if poses:
            tr = box.transformed(poses[0])
        self.assertGreater(len(poses), 1)
        #embed()


if __name__ == "__main__":
    unittest.main(verbosity=3)
