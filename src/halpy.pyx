cimport numpy as cnp
import numpy as np
from libcpp cimport string
cimport cpp_halpy as cpp


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
            raise RuntimeError("Argument not supported")

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

    def append(self, double val):
        cdef cpp.HTuple tpl = cpp.HTuple(val)
        print("my length!", self.me.Length())
        #self.me.Append(tpl)
        self.me.add(tpl)
        print("my length!", self.me.Length())

    #def append(self, val):
        #cdef cpp.HTuple tpl = cpp.HTuple()
        #if isinstance(val, float):
            #tpl.assign(<double> val)
        #elif isinstance(val, int):
            #tpl.assign(<int> val)
        #self.me.Append(tpl)

    def __getitem__(self, int val):
        dt = self.me.Type()
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


def read_object_model_3d(str path, str scale_str, GenParamName, GenParamValue):
    t_path = HTuple(path)
    t_scale = HTuple(scale_str)
    cdef cpp.HTuple t_name = cpp.HTuple()
    cdef cpp.HTuple t_value = cpp.HTuple()
    t_res = HTuple()
    t_status = HTuple()
    print("ARGS", t_path[0], t_scale[0])
    cpp.ReadObjectModel3d(t_path.me, t_scale.me, t_name, t_value, &t_res.me, &t_status.me)
    return t_res.to_array()



