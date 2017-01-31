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
        HTuple(long) except +
        HTuple(double) except +
        HTuple(double*, int) except +
        HTuple(long*, int) except +
        int Type()
        int Length()
        void assign "operator="(int)
        void assign "operator="(double)
        void assign "operator="(char*)
        void add "operator+="(double)
        void add "operator+="(int)
        void add "operator+="(HTuple)
        HTupleElement operator[](int)
        HString ToString() const
        double* DArr()
        HTuple& Append(const HTuple&)
        void Clear()


    void ReadObjectModel3d(const HTuple& FileName, const HTuple& Scale, const HTuple& GenParamName, const HTuple& GenParamValue, HTuple* ObjectModel3D, HTuple* Status)

    void SampleObjectModel3d(const HTuple& ObjectModel3D, const HTuple& Method, const HTuple& SampleDistance, const HTuple& GenParamName, const HTuple& GenParamValue, HTuple* SampledObjectModel3D)


# This is the object oriented interface
# I am not usre it is interressting from Python, since
# this object interface is auto generated and not very intuitive
# probably better to use functions and create our own object 
# oriented interface in Python

cdef extern from "HObjectModel3D.h" namespace "HalconCpp":
    cdef cppclass HObjectModel3D:
        HObjectModel3D()
