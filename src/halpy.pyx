import numpy as np

from libcpp cimport string
cimport numpy as cnp
cimport cpp_halpy as cpp
from cython.view cimport array as cvarray


from enum import Enum


class TupleType(Enum):
    Int = 1
    Double = 2
    String = 4


cdef class HTuple:

    cdef cpp.HTuple me

    def __cinit__(self, arg=None):
        if arg is None:
            self.me = cpp.HTuple()
        elif isinstance(arg, float):
            self.me = cpp.HTuple(<double>arg)
        elif isinstance(arg, int):
            self.me = cpp.HTuple(<int>arg)
        elif isinstance(arg, bytes):
            self.me = cpp.HTuple((<const char*>arg))
        elif isinstance(arg, str):
            tt = arg.encode()
            self.me = cpp.HTuple((<const char*>tt))
        else:
            raise RuntimeError("Argument not supported", arg)

    @staticmethod
    def from_array(ar):
        if ar.dtype == np.int:
            return HTuple.from_array_int(ar)
        elif ar.dtype == np.double:
            return HTuple.from_array_double(ar)
        else:
            raise RuntimeError("Argument not supported", ar)

    @staticmethod
    def from_array_double(cnp.ndarray[cnp.double_t, ndim=1, mode="c"] arg):
        cdef cpp.HTuple t = cpp.HTuple(<double*>&arg[0], <int> arg.shape[0])
        pyt = HTuple()
        pyt.me = t
        return pyt

    @staticmethod
    def from_array_int(cnp.ndarray[cnp.long_t, ndim=1, mode="c"] arg):
        cdef cpp.HTuple t = cpp.HTuple(<long*>&arg[0], <int> arg.shape[0])
        pyt = HTuple()
        pyt.me = t
        return pyt


    @staticmethod
    def from_double(double val):
        t = HTuple()
        t.me.assign(val)
        return t

    @staticmethod
    def from_string(str val):
        val = val.encode("utf-8")
        return HTuple.from_bytes(val)

    @staticmethod
    def from_bytes(bytes val):
        t = HTuple()
        cdef bytes py_bytes = val
        cdef const char* s = py_bytes
        t.me.assign(s)
        return t

    def type(self):
        return TupleType(self.me.Type())

    def to_string(self):
        cdef cpp.HString hs = self.me.ToString()
        cdef const char * c_string = hs.Text()
        cdef bytes py_string = c_string
        return py_string

    def to_array(self):
        #FIXME: should be possible to access C array and make numpy array from it instead of looping
        cdef int n = self.me.Length()
        dt = self.me.Type()
        if dt == 0:
            return None
        elif dt == 1:
            result = np.empty(n, dtype=np.int)
            for i in range(n):
                result[i] = self.me[i].L()
        elif dt == 2:
            result = np.empty(n, dtype=np.double)
            for i in range(n):
                result[i] = self.me[i].D()
        elif dt == 4:
            result = np.empty(n, dtype=np.object)
            for i in range(n):
                result[i] = self.me[i].C()
        elif dt == 8:
            # THIS is wrong fix, 8 mean mixed
            result = np.empty(n, dtype=np.double)
            for i in range(n):
                result[i] = self.me[i].D()
        else:
            raise RuntimeError("unknown data type", dt)
        return result

    def to_list(self):
        return [self[i] for i in range(self.length())]

    def append(self, val):
        if isinstance(val, float):
            self.me.Append(<cpp.HTuple> cpp.HTuple((<double>val)))
        elif isinstance(val, int):
            self.me.Append(<cpp.HTuple> cpp.HTuple((<int>val)))
        elif isinstance(val, bytes):
            self.me.Append(<cpp.HTuple> cpp.HTuple((<const char*>val)))
        else:
            raise RuntimeError("Unknown type")

    #def append(self, val):
        #cdef cpp.HTuple tpl = cpp.HTuple()
        #if isinstance(val, float):
            #tpl.assign(<double> val)
        #elif isinstance(val, int):
            #tpl.assign(<int> val)
        #self.me.Append(tpl)

    def __getitem__(self, int val):
        dt = self.me.Type()
        if val >= self.length():
            raise ValueError("Out of bound")
        if dt == 0:
            return None
        elif dt == 1:
            return self.me[val].L()
        elif dt == 2:
            return self.me[val].D()
        elif dt == 4:
            return self.me[val].C()

    def length(self):
        return self.me.Length()


cdef _ht2ar(cpp.HTuple tup):
    """
    cpp.HTuple to numpy array
    """
    t = HTuple()
    t.me = tup
    return t.to_array()


cdef cpp.HTuple _ar2ht(cnp.ndarray ar):
    """
    nupy array to cpp.HTuple
    """
    if not ar.flags['C_CONTIGUOUS']:
        ar = np.ascontiguousarray(ar)
    cdef HTuple t = HTuple.from_array(ar)
    return t.me


cdef class Model:

    cdef cpp.HObjectModel3D me

    def __cinit__(self):
        self.me = cpp.HObjectModel3D()

    @staticmethod
    def from_file(str path, str scale):
        model = Model()
        cdef bytes bscale = scale.encode() # no idea why I need thi intemediary step for HTuple and not for HString??
        cdef cpp.HTuple status;
        model.me = cpp.HObjectModel3D(cpp.HString(path.encode()), cpp.HTuple(bscale), cpp.HTuple(), cpp.HTuple(), &status)
        return model
        #print("STATUS", status.ToString())
    
    @staticmethod
    def from_array(ar):
        model = Model()
        model.me = cpp.HObjectModel3D(_ar2ht(ar[:, 0]), _ar2ht(ar[:, 1]), _ar2ht(ar[:, 2]))
        return model

    def get_bounding_box(self, oriented=True):
        cdef double x, y, z
        cdef cpp.HPose pose = self.me.SmallestBoundingBoxObjectModel3d("oriented", &x, &y, &z)
        #p = HTuple()
        #p.me = pose.ConvertToTuple()
        return x, y, z

    def to_array(self):
        cdef cpp.HTuple x = self.me.GetObjectModel3dParams(cpp.HTuple(b"point_coord_x"))
        cdef cpp.HTuple y = self.me.GetObjectModel3dParams(cpp.HTuple(b"point_coord_y"))
        cdef cpp.HTuple z = self.me.GetObjectModel3dParams(cpp.HTuple(b"point_coord_z"))
        nx = _ht2ar(x)
        nx.shape = -11, 1
        ny = _ht2ar(y)
        ny.shape = -1, 1
        nz = _ht2ar(z)
        nz.shape = -1, 1
        return np.hstack((nx, ny, nz))

    def normals_to_array(self):
        cdef cpp.HTuple x = self.me.GetObjectModel3dParams(cpp.HTuple(b"point_normal_x"))
        cdef cpp.HTuple y = self.me.GetObjectModel3dParams(cpp.HTuple(b"point_normal_y"))
        cdef cpp.HTuple z = self.me.GetObjectModel3dParams(cpp.HTuple(b"point_normal_z"))
        nx = _ht2ar(x)
        nx.shape = -11, 1
        ny = _ht2ar(y)
        ny.shape = -1, 1
        nz = _ht2ar(z)
        nz.shape = -1, 1
        return np.hstack((nx, ny, nz))

    def get_convex_hull(self):
        m = Model()
        m.me = self.me.ConvexHullObjectModel3d()
        return m

    def sample(self, double dist, str method="fast"):
        m = Model()
        m.me = self.me.SampleObjectModel3d(method.encode(), dist, cpp.HTuple(), cpp.HTuple())
        return m

    def Xto_file(self, str filetype, str path):
        self.me.WriteObjectModel3d(cpp.HString(filetype.encode()), cpp.HString(path.encode()), cpp.HTuple(), cpp.HTuple())



