import numpy as np

from libcpp cimport string
cimport numpy as cnp
cimport hcn.cpp_hcn as cpp
from cython.view cimport array as cvarray


from enum import Enum


class TupleType(Enum):
    Int = 1
    Double = 2
    String = 4
    Mixed = 8


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
    def from_list(arg):
        pyt = HTuple()
        pyt.me = _list2tuple(arg)
        return pyt

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

    def to_array_double(self):
        cdef int n = self.me.Length()
        #ToDarr() instead of Darr() might generate an extra copy, but also works for mixed type
        cdef cnp.double_t[:] view = <cnp.double_t[:n]> self.me.ToDArr()
        return np.asarray(view)

    def to_array_int(self):
        cdef int n = self.me.Length()
        #cdef cnp.long_t[:] view = <cnp.long_t[:n]> self.me.LArr()  # cython does not want this..
        cdef long[:] view = <long[:n]> self.me.ToLArr()
        return np.asarray(view)

    def to_array_string(self):
        cdef int n = self.me.Length()
        result = cnp.empty(n, dtype=np.object)
        for i in range(n):
            result[i] = self.me[i].C()

    def to_array(self):
        dt = self.me.Type() 
        if dt == 0:
            return None
        elif dt == 1:
            return self.to_array_int()
        elif dt == 2:
            return self.to_array_double()
        elif dt == 4:
            return self.to_array_string()
        elif dt == 8:
            raise RuntimeError("HTuple of type mixed cannot be converted to numpy array, if you know its type try to call to_array_double or to_array_int")
        else:
            raise RuntimeError("unknown data type", dt)

    def to_list(self):
        return [self[i] for i in range(self.length()) ]

    def append(self, val):
        if isinstance(val, float):
            self.me.Append(<cpp.HTuple> cpp.HTuple((<double>val)))
        elif isinstance(val, int):
            self.me.Append(<cpp.HTuple> cpp.HTuple((<int>val)))
        elif isinstance(val, bytes):
            self.me.Append(<cpp.HTuple> cpp.HTuple((<const char*>val)))
        else:
            raise RuntimeError("Unknown type")

    def __getitem__(self, int val):
        if val >= self.length():
            raise ValueError("Out of bound")
        dt = self.me[val].Type()
        if dt == 0:
            return None
        elif dt == 1:
            return self.me[val].L()
        elif dt == 2:
            return self.me[val].D()
        elif dt == 4:
            return self.me[val].S().Text()
        else:
            raise RuntimeError("Unknown type", dt)

    def length(self):
        return self.me.Length()


cdef _append_double(cpp.HTuple& tup, double val):
    tup.Append(cpp.HTuple(val))


cdef _append_int(cpp.HTuple& tup, long val):
    tup.Append(cpp.HTuple(val))


cdef _append_bytes(cpp.HTuple& tup, bytes val):
    tup.Append(cpp.HTuple(val))


cdef cpp.HTuple _list2tuple(arg):
    cdef cpp.HTuple tup
    for i in arg:
        if isinstance(i, float):
            _append_double(tup, i)
        elif isinstance(i, int):
            _append_int(tup, i)
        elif isinstance(i, str):
            _append_bytes(tup, i.encode())
        elif isinstance(i, bytes):
            _append_bytes(tup, i)
    return tup


cdef _ht2ar(cpp.HTuple tup):
    """
    cpp.HTuple to numpy array double
    """
    t = HTuple()
    t.me = tup
    return t.to_array_double()


cdef cpp.HTuple _ar2ht(cnp.ndarray ar):
    """
    nupy array to cpp.HTuple
    """
    if not ar.flags['C_CONTIGUOUS']:
        ar = np.ascontiguousarray(ar)
    cdef HTuple t = HTuple.from_array(ar)
    return t.me


cdef cpp.HPose transform_to_hpose(trans):
    cdef int rx, ry, rz 
    rx, ry, rz = trans.orient.to_euler("xyz")
    cdef cpp.HPose pose = cpp.HPose(trans.pos.x, trans.pos.y, trans.pos.z, rx, ry, rz, "Rp+T", "gba", "point")
    return pose


cdef class Surface:

    cdef cpp.HSurfaceModel me

    def __cinit__(self):
        self.me = cpp.HSurfaceModel()


cdef class Model3D:

    cdef cpp.HObjectModel3D me

    def __cinit__(self):
        self.me = cpp.HObjectModel3D()

    @staticmethod
    def from_file(str path, str scale):
        model = Model3D()
        cdef bytes bscale = scale.encode() # no idea why I need thi intemediary step for HTuple and not for HString??
        cdef cpp.HTuple status;
        model.me = cpp.HObjectModel3D(cpp.HString(path.encode()), cpp.HTuple(bscale), cpp.HTuple(), cpp.HTuple(), &status)
        return model
        #print("STATUS", status.ToString())
    
    @staticmethod
    def from_array(ar):
        model = Model3D()
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
        m = Model3D()
        m.me = self.me.ConvexHullObjectModel3d()
        return m

    def sampled(self, double dist, str method="fast"):
        """
        Return a sampled copy of point cloud
        """
        m = Model3D()
        m.me = self.me.SampleObjectModel3d(method.encode(), dist, cpp.HTuple(), cpp.HTuple())
        return m

    def smoothed(self, int knn=60, int order=2):
        m = Model3D()
        names = []
        vals = []
        names.append(b"mls_kNN")
        vals.append(knn)
        names.append(b"mls_order")
        vals.append(order)
        m.me = self.me.SmoothObjectModel3d(b"mls", _list2tuple(names), _list2tuple(vals))
        return m

    def create_surface_model(self, double dist, invert_normals="false"):
        s = Surface()
        names = []
        vals = []
        names.append(b"model_invert_normals")
        vals.append(invert_normals)
        s.me = self.me.CreateSurfaceModel(dist, _list2tuple(names), _list2tuple(vals))
        return s

    def compute_normals(self, int knn, int order):
        m = Model3D()
        names = []
        vals = []
        names.append(b"mls_kNN")
        vals.append(knn)
        names.append(b"mls_order")
        vals.append(order)
        m.me = self.me.SurfaceNormalsObjectModel3d(b"mls", _list2tuple(names), _list2tuple(vals))
        return m

    def to_file(self, str filetype, str path):
        self.me.WriteObjectModel3d(cpp.HString(filetype.encode()), cpp.HString(path.encode()), cpp.HTuple(), cpp.HTuple())               

    def select_points(self, str attr, double min_val, double max_val): 
        """
        attr is one of:
            point_coord_x
            point_coord_y
            point_coord_z
            point_normal_x
            point_normal_y
            point_normal_z
            etc

        """
        m = Model3D()
        m.me = self.me.SelectPointsObjectModel3d(attr.encode(), min_val, max_val)
        return m

    def select_x(self, min_val, max_val):
        return self.select_points("point_coord_x", min_val, max_val)

    def select_y(self, min_val, max_val):
        return self.select_points("point_coord_y", min_val, max_val)

    def select_z(self, min_val, max_val):
        return self.select_points("point_coord_z", min_val, max_val)



cdef class Plane(Model3D):
    def __init__(self, trans, xext, yext):
        Model3D.__init__(self)
        cdef cpp.HPose pose = transform_to_hpose(trans)
        self.me.GenPlaneObjectModel3d(pose, cpp.HTuple(), cpp.HTuple())


cdef class Sphere(Model3D):
    def __init__(self, double x, double y , double z, double radius):
        Model3D.__init__(self)
        self.me.GenSphereObjectModel3dCenter(x, y, z, radius)


cdef class Box(Model3D):
    def __init__(self, trans, double x, double y, double z):
        Model3D.__init__(self)
        cdef cpp.HPose pose = transform_to_hpose(trans)
        self.me.GenBoxObjectModel3d(pose, x, y, z)
        #self.sample(0.2)


