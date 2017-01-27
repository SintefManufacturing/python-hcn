from libc.stdint cimport uint32_t, uint16_t
from libcpp cimport bool


cdef extern from "HalconCpp.h" namespace "HalconCpp":
    cdef cppclass HTuple:
        HTuple()
        int State()
        int Reset()


