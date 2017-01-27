cimport cpp_halpy as cpp


cdef class HTuple:

    cdef cpp.HTuple *me

    def __cinit__(self):
        self.me = new cpp.HTuple()

    def __dealloc__(self):
        del self.me

    def state(self):
        return self.me.State()

    def reset(self):
        return self.me.Reset()



