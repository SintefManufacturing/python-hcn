from libc.stdint cimport uint32_t, uint16_t
from libcpp cimport string
from libcpp cimport bool


cdef extern from "hcn/cy_handler.h":
  cdef void raise_py_error()


cdef extern from "HalconCpp.h" namespace "HalconCpp":

    cdef union Hpar:
        double d
        char* s
        long l

    cdef struct Hcpar:
        Hpar par
        int type

    cdef cppclass HQuaternion:
        HQuaternion() except +raise_py_error
        HQuaternion(const HTuple&) except +raise_py_error
        HPose QuatToPose()
        void PoseToQuat(const HPose&) except +raise_py_error
        HTuple ConvertToTuple()

    cdef cppclass HPose:
        HPose() except +raise_py_error
        HPose(const HTuple&) except +raise_py_error
        HTuple ConvertToTuple() except +raise_py_error
        HPose(double TransX, double TransY, double TransZ, double RotX, double RotY, double RotZ, const char* OrderOfTransform, const char* OrderOfRotation, const char* ViewOfTransform) except +raise_py_error

    cdef cppclass HPoseArray:
        HPoseArray() except +raise_py_error
        long Length()
        HPose* Data()
        HTuple ConvertToTuple() except +raise_py_error

    cdef cppclass HSurfaceMatchingResult:
        HSurfaceMatchingResult() except +raise_py_error

    cdef cppclass HSurfaceMatchingResultArray:
        HSurfaceMatchingResultArray() except +raise_py_error

    cdef cppclass HString:
        HString() except +raise_py_error
        HString(const char*) except +raise_py_error
        const char* Text() const

    cdef cppclass HTupleElement:
        #HTupleElement() except +raise_py_error
        HTupleElement(double) except +raise_py_error
        HTupleElement(int) except +raise_py_error
        int Type()
        int L()
        double D()
        Hcpar P()
        HString S()
        const char* C()

    cdef cppclass HTuple:
        HTuple() except +raise_py_error
        HTuple(const char*) except +raise_py_error
        HTuple(const HString&) except +raise_py_error
        HTuple(long) except +raise_py_error
        HTuple(double) except +raise_py_error
        HTuple(double*, int) except +raise_py_error
        HTuple(long*, int) except +raise_py_error
        int Type()
        int Length()

        long*  LArr();
        double* DArr();
        char**  SArr();
        long*  ToLArr();
        double* ToDArr();
        char**  ToSArr();
        Hcpar*  ToPArr();

        void assign "operator="(int)
        void assign "operator="(double)
        void assign "operator="(char*)
        void add "operator+="(double)
        void add "operator+="(int)
        void add "operator+="(HTuple)
        void add "operator+="(HObjectModel3D)
        HTupleElement operator[](int)
        HString ToString()
        HTuple& Append(const HTuple&) except +raise_py_error
        void Clear()


cdef extern from "HSurfaceModel.h" namespace "HalconCpp":
    cdef cppclass HSurfaceModel:
        #constructors
        HSurfaceModel() except +raise_py_error
        HSurfaceModel(const char* FileName) 
        HPoseArray FindSurfaceModel(const HObjectModel3D&, double RelSamplingDistance, double KeyPointFraction, const HTuple& MinScore, const HString& ReturnResultHandle, const HTuple& GenParamName, const HTuple& GenParamValue, HTuple* Score, HSurfaceMatchingResultArray*)  except +raise_py_error
        HPose RefineSurfaceModelPose(const HObjectModel3D& ObjectModel3D, const HPose& InitialPose, double MinScore, const HString& ReturnResultHandle, const HTuple& GenParamName, const HTuple& GenParamValue, HTuple* Score, HSurfaceMatchingResult* SurfaceMatchingResultID) const;


cdef extern from "HObjectModel3D.h" namespace "HalconCpp":

    cdef cppclass HObjectModel3D:
        #constructors
        HObjectModel3D() except +raise_py_error
        HObjectModel3D(const HTuple& X, const HTuple& Y, const HTuple& Z) except +raise_py_error
        HObjectModel3D(const HString& FileName, const HTuple& Scale, const HTuple& GenParamName, const HTuple& GenParamValue, HTuple* Status) except +raise_py_error
        void GenPlaneObjectModel3d(const HPose& Pose, const HTuple& XExtent, const HTuple& YExtent) except +raise_py_error
        void GenSphereObjectModel3dCenter(double X, double Y, double Z, double Radius) except +raise_py_error
        void GenBoxObjectModel3d(const HPose& Pose, double LengthX, double LengthY, double LengthZ) except +raise_py_error

        #write
        void WriteObjectModel3d(const HString& FileType, const HString& FileName, const HTuple& GenParamName, const HTuple& GenParamValue) except +raise_py_error

        #operations
        HObjectModel3D SelectPointsObjectModel3d(const char* Attrib, double MinValue, double MaxValue) except +raise_py_error
        HPose SmallestBoundingBoxObjectModel3d(const char*, double*, double*, double*) except +raise_py_error
        HObjectModel3D ConvexHullObjectModel3d() except +raise_py_error
        HTuple GetObjectModel3dParams(const HTuple& GenParamName) except +raise_py_error
        HObjectModel3D FitPrimitivesObjectModel3d(const HTuple& GenParamName, const HTuple& GenParamValue) except +raise_py_error
        @staticmethod
        HObjectModel3DArray FitPrimitivesObjectModel3d(const HObjectModel3DArray& ObjectModel3D, const HTuple& GenParamName, const HTuple& GenParamValue);
        HObjectModel3D SurfaceNormalsObjectModel3d(const char* Method, const HTuple& GenParamName, const HTuple& GenParamValue) except +raise_py_error
        HObjectModel3D SmoothObjectModel3d(const char* Method, const HTuple& GenParamName, const HTuple& GenParamValue) except +raise_py_error;
        HSurfaceModel CreateSurfaceModel(double RelSamplingDistance, const HTuple& GenParamName, const HTuple& GenParamValue) except +raise_py_error
        void DistanceObjectModel3d(const HObjectModel3D& ObjectModel3DTo, const HPose& Pose, const HTuple& MaxDistance, const HTuple& GenParamName, const HTuple& GenParamValue) except +raise_py_error
        HObjectModel3D SampleObjectModel3d(const char* Method, double SampleDistance, const HTuple& GenParamName, const HTuple& GenParamValue) except +raise_py_error
        
        HObjectModel3D EdgesObjectModel3d(const HTuple& MinAmplitude, const HTuple& GenParamName, const HTuple& GenParamValue) const;
        HObjectModel3D RigidTransObjectModel3d(const HPose& Pose) except +raise_py_error
        @staticmethod
        HObjectModel3D UnionObjectModel3d(const HObjectModel3DArray& ObjectModels3D, const HString& Method) except +raise_py_error
        HObjectModel3D SegmentObjectModel3d(const HTuple& GenParamName, const HTuple& GenParamValue) except +raise_py_error
        void PrepareObjectModel3d(const char* Purpose, const char* OverwriteData, const HTuple& GenParamName, const HTuple& GenParamValue) except +raise_py_error
        @staticmethod
        HObjectModel3DArray SegmentObjectModel3d(const HObjectModel3DArray& ObjectModel3D, const HTuple& GenParamName, const HTuple& GenParamValue) except +raise_py_error
        HObjectModel3DArray ConnectionObjectModel3d(const char* Feature, double Value) const;


    cdef cppclass HObjectModel3DArray:
        HObjectModel3DArray() except +raise_py_error
        HObjectModel3DArray(HObjectModel3D* classes, long length) except +raise_py_error
        long Length()
        HObjectModel3D* Tools()
        HTuple ConvertToTuple() except +raise_py_error
        void SetFromTuple(const HTuple& concatenated) except +raise_py_error


