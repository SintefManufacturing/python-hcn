
import unittest

import numpy as np

import halpy

from IPython import embed


class HalpyTests(unittest.TestCase):

    def test_tuple_double(self):
        val = 0.199
        d = halpy.HTuple(val)
        self.assertEqual(d.length(), 1)
        a = np.array([val])
        self.assertEqual(d.to_array(), a)
        self.assertEqual(d[0], val)
        self.assertEqual(d.type(), halpy.TupleType.Double)

    def test_tuple_int(self):
        val = 7
        d = halpy.HTuple(val)
        self.assertEqual(d.length(), 1)
        a = np.array([val])
        self.assertEqual(d.to_array(), a)
        self.assertEqual(d[0], val)
        self.assertEqual(d.type(), halpy.TupleType.Int)

    def test_tuple_bytes(self):
        val = b"toto is None"
        d = halpy.HTuple(val)
        self.assertEqual(d.length(), 1)
        a = [val]
        self.assertEqual(d.to_list(), a)
        self.assertEqual(d[0], val)
        self.assertEqual(d.type(), halpy.TupleType.String)

    def test_append(self):
        val = 0.199
        d = halpy.HTuple(val)
        d.append(3.4)
        self.assertEqual(d.length(), 2)
    
    def test_read_model(self):
        res = halpy.read_object_model_3d("arm_base.stl", "mm", None, None)

    def test_array_double(self):
        a = np.array([1.1, 2.1, 3.1, 4.1, 5.1], dtype=np.double)
        t = halpy.HTuple.from_array(a)
        self.assertEqual(t.length(), 5)
        self.assertEqual(t[0], a[0])
        self.assertEqual(t[4], t[4])

    def test_array_int(self):
        a = np.array([1, 2, 3, 4, 5], dtype=np.int)
        t = halpy.HTuple.from_array(a)
        self.assertEqual(t.length(), 5)
        self.assertEqual(t[0], t[0])
        self.assertEqual(t[4], t[4])





if __name__ == "__main__":
    unittest.main()
"""
    e = halpy.HTuple()
    #d.append(9.2)
    s = halpy.HTuple("totot is back")
    print("LENGTH empty", e.length())
    #embed()
    print("val[0]", d[0])
    print("val array", d.to_array())
    #print("string array", d.to_string())
    print("String", s.length())
    print("String", s[0])
    #print("String", s.to_string())
    print("Try read model")
    try:
        res = halpy.read_object_model_3d("arm_base.stl", "mm", None, None)
    except ex:
        print("Exception", ex)
    print(res)
"""
