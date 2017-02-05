from libc.stdint cimport uint32_t, uint16_t
from libcpp cimport string
from libcpp cimport bool


cdef extern from "HalconCpp.h" namespace "HalconCpp":

    cdef cppclass HPose:
        HPose() except +
        HTuple ConvertToTuple()

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


cdef extern from "HObjectModel3D.h" namespace "HalconCpp":
    cdef cppclass HObjectModel3D:
        #constructors
        HObjectModel3D() except +
        HObjectModel3D(const HTuple& X, const HTuple& Y, const HTuple& Z) except +
        HObjectModel3D(const HString& FileName, const HTuple& Scale, const HTuple& GenParamName, const HTuple& GenParamValue, HTuple* Status) except +
        void GenPlaneObjectModel3d(const HPose& Pose, double XExtent, double YExtent) except +
        void GenSphereObjectModel3dCenter(double X, double Y, double Z, double Radius) except +

        #write
        void WriteObjectModel3d(const HString& FileType, const HString& FileName, const HTuple& GenParamName, const HTuple& GenParamValue) const

        #operations
        HObjectModel3D SelectPointsObjectModel3d(const char* Attrib, double MinValue, double MaxValue) const;
        HPose SmallestBoundingBoxObjectModel3d(const char*, double*, double*, double*)
        HObjectModel3D ConvexHullObjectModel3d() const;
        HTuple GetObjectModel3dParams(const HTuple& GenParamName) const;
        HObjectModel3D FitPrimitivesObjectModel3d(const HTuple& GenParamName, const HTuple& GenParamValue) const;
        HObjectModel3D SurfaceNormalsObjectModel3d(const char* Method, const HTuple& GenParamName, const HTuple& GenParamValue) const;
        HObjectModel3D SmoothObjectModel3d(const char* Method, const HTuple& GenParamName, const HTuple& GenParamValue) const;
        #HSurfaceModel CreateSurfaceModel(double RelSamplingDistance, const char* GenParamName, const char* GenParamValue) const;
        void DistanceObjectModel3d(const HObjectModel3D& ObjectModel3DTo, const HPose& Pose, double MaxDistance, const char* GenParamName, const char* GenParamValue) const;
        HObjectModel3D SampleObjectModel3d(const char* Method, double SampleDistance, const HTuple& GenParamName, const HTuple& GenParamValue) const;
        
        HObjectModel3D EdgesObjectModel3d(const HTuple& MinAmplitude, const HTuple& GenParamName, const HTuple& GenParamValue) const;
