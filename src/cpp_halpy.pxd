from libc.stdint cimport uint32_t, uint16_t
from libcpp cimport string
from libcpp cimport bool


cdef extern from "HalconCpp.h" namespace "HalconCpp":

    cdef cppclass HString:
        HString() except +
        HString(const char*) except +
        const char* Text() const

    cdef cppclass HTupleElement:
        HTupleElement() except +
        HTupleElement(double) except +
        HTupleElement(int) except +
        int Type()
        int I()
        int L()
        double D()
        HString S()
        const char* C()

    cdef cppclass HTuple:
        HTuple() except +
        HTuple(const char*) except +
        HTuple(const HString&) except +
        HTuple(int) except +
        HTuple(double) except +
        int Type()
        int Length()
        void assign "operator="(int)
        void assign "operator="(double)
        void assign "operator="(char*)
        void add "operator+="(HTuple)
        HTupleElement operator[](int)
        HString ToString() const
        double* DArr()
        HTuple& Append(const HTuple&)
        void Clear()


    void ReadObjectModel3d(const HTuple& filename, const HTuple& scale, const HTuple& GenParamName, const HTuple& GenParamValue, HTuple* ObjectModel3D, HTuple* Status)


