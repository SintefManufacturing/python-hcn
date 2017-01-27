from libcpp cimport string
cimport cpp_halpy as cpp


cdef class HTuple:

    cdef cpp.HTuple me

    def __cinit__(self, arg=None):
        self.me = cpp.HTuple()

    @staticmethod
    def from_double(double val):
        t = HTuple()
        t.me.assign(val)
        return t

    def to_string(self):
        hs = self.me.ToString()
        return None

    def __getitem__(self, int val):
        return self.me[val]

    def length(self):
        return self.me.Length()

#def read_object_model_3d(path, unit_str, a, b):
    #t_path = cpp.Htuple(path)
    #t_unit = cpp.HTuple(unit_str)
    #cpp.ReadObjectModel3d(



