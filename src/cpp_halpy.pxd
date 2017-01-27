from libc.stdint cimport uint32_t, uint16_t
from libcpp cimport bool


cdef extern from "HalconCpp.h" namespace "HalconCpp":
    cdef cppclass HLong:
        HLong()

    cdef cppclass HString:
        HString()

    cdef cppclass HTuple:
        HTuple()
        HTuple(str)
        HTuple(int)
        HTuple(double)
        double Length()
        void assign "operator="(double)
        double operator[](int)
        HString ToString()
        double* DArr()


    void ReadObjectModel3d(HTuple, HTuple, HTuple, HTuple, HTuple, HTuplehv_Status)
