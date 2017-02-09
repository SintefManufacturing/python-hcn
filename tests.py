import unittest

import numpy as np

import hcn

from IPython import embed

class TestTuple(unittest.TestCase):

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

   

class TestsModel(unittest.TestCase):
    def test_read_model(self):
        m = hcn.Model.from_file("simple.obj", "m")
        ar = m.to_array()
        np.testing.assert_array_equal(ar[0], np.array([1, 1, 1]))
        np.testing.assert_array_equal(ar[3], np.array([1, 1, 2]))

    def test_to_from_array_model(self):
        ar = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.double)
        m = hcn.Model.from_array(ar)
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

    def test_sample(self):
        m = self._get_simple_model()
        print("F", m.to_array())
        new = m.sample(2)
        self.assertEqual(len(new.to_array()), 2)

    def test_exception(self):
        m = self._get_simple_model()
        with self.assertRaises(RuntimeError):
            m.to_file("obj", "/t.obj")

    def _get_simple_model(self):
        ar = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.double)
        return hcn.Model.from_array(ar)








if __name__ == "__main__":
    unittest.main()
