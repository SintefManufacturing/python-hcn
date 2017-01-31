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


def read_model(const char* path, const char * scale, GenParamName, GenParamValue):
    #t_path = HTuple(path)
    #t_scale = HTuple(scale_str)
    #cdef cpp.HTuple t_name = cpp.HTuple()
    #cdef cpp.HTuple t_value = cpp.HTuple()
    t_res = HTuple()
    t_status = HTuple()
    #cpp.ReadObjectModel3d(t_path.me, t_scale.me, t_name, t_value, &t_res.me, &t_status.me)
    cpp.ReadObjectModel3d(cpp.HTuple(path), cpp.HTuple(scale), cpp.HTuple(), cpp.HTuple(), &t_res.me, &t_status.me)
    print("STATUS", t_status.to_string())
    return t_res.to_array()

def sample_model(HTuple t_model, const char* method, double sample_dist):
    #if isinstance(t_model, np.ndarray):
        #t_model = HTuple.from_array(t_model)
    #t_method = HTuple(method)
    #t_dist = HTuple(sample_dist)
    #cdef cpp.HTuple t_name = cpp.HTuple()
    #cdef cpp.HTuple t_value = cpp.HTuple()
    t_res = HTuple()
    cpp.SampleObjectModel3d(<cpp.HTuple> t_model.me, cpp.HTuple(method), cpp.HTuple(sample_dist), cpp.HTuple(), cpp.HTuple(), &t_res.me)
    return t_res.to_array()



cdef class HObjectModel3D:

    cdef cpp.HObjectModel3D me

    def __cinit__(self):
        self.me = cpp.HObjectModel3D()
