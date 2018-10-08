import numpy as np

from libcpp cimport string
cimport numpy as cnp
cimport hcn.cpp_hcn as cpp
from cython.view cimport array as cvarray
from cython import NULL

from enum import Enum

MATH3D = True
try:
    import math3d as m3d
except ImportError:
    MATH3D = False


class TupleType(Enum):
    Int = 1
    Double = 2
    String = 4
    Mixed = 8


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
    def from_list(arg):
        pyt = HTuple()
        pyt.me = _list2tuple(arg)
        return pyt

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

    def __repr__(self):
        return "HTuple({})".format(self.to_list())

    def type(self):
        return TupleType(self.me.Type())

    def to_string(self):
        cdef cpp.HString hs = self.me.ToString()
        cdef const char * c_string = hs.Text()
        cdef bytes py_string = c_string
        return py_string

    def to_array_double(self):
        cdef int n = self.me.Length()
        #ToDarr() instead of Darr() might generate an extra copy, but also works for mixed type
        cdef cnp.double_t[:] view = <cnp.double_t[:n]> self.me.ToDArr()
        return np.asarray(view)

    def to_array_int(self):
        cdef int n = self.me.Length()
        #cdef cnp.long_t[:] view = <cnp.long_t[:n]> self.me.LArr()  # cython does not want this..
        cdef long[:] view = <long[:n]> self.me.ToLArr()
        return np.asarray(view)

    def to_array_string(self):
        cdef int n = self.me.Length()
        result = cnp.empty(n, dtype=np.object)
        for i in range(n):
            result[i] = self.me[i].C()

    def to_array(self):
        dt = self.me.Type() 
        if dt == 0:
            return None
        elif dt == 1:
            return self.to_array_int()
        elif dt == 2:
            return self.to_array_double()
        elif dt == 4:
            return self.to_array_string()
        elif dt == 8:
            raise RuntimeError("HTuple of type mixed cannot be converted to numpy array, if you know its type try to call to_array_double or to_array_int")
        else:
            raise RuntimeError("unknown data type", dt)

    def to_list(self):
        return [self[i] for i in range(self.length()) ]

    def append(self, val):
        if isinstance(val, float):
            self.me.Append(<cpp.HTuple> cpp.HTuple((<double>val)))
        elif isinstance(val, int):
            self.me.Append(<cpp.HTuple> cpp.HTuple((<int>val)))
        elif isinstance(val, bytes):
            self.me.Append(<cpp.HTuple> cpp.HTuple((<const char*>val)))
        else:
            raise RuntimeError("Unknown type")

    def __getitem__(self, int val):
        if val >= self.length():
            raise ValueError("Out of bound")
        dt = self.me[val].Type()
        if dt == 0:
            return None
        elif dt == 1:
            return self.me[val].L()
        elif dt == 2:
            return self.me[val].D()
        elif dt == 4:
            return self.me[val].S().Text()
        else:
            raise RuntimeError("Unknown type", dt)

    def length(self):
        return self.me.Length()


cdef _append_double(cpp.HTuple& tup, double val):
    tup.Append(cpp.HTuple(val))


cdef _append_int(cpp.HTuple& tup, long val):
    tup.Append(cpp.HTuple(val))


cdef _append_bytes(cpp.HTuple& tup, bytes val):
    tup.Append(cpp.HTuple(val))


cdef cpp.HTuple _list2tuple(arg):
    cdef cpp.HTuple tup
    for i in arg:
        if isinstance(i, float):
            _append_double(tup, i)
        elif isinstance(i, int):
            _append_int(tup, i)
        elif isinstance(i, str):
            _append_bytes(tup, i.encode())
        elif isinstance(i, bytes):
            _append_bytes(tup, i)
        else:
            raise RuntimeError("arg type not supported")
    return tup


cdef _ht2ar(cpp.HTuple& tup):
    """
    cpp.HTuple to numpy array double
    """
    t = HTuple()
    t.me = tup
    return t.to_array_double()


cdef cpp.HTuple _ar2ht(cnp.ndarray ar):
    """
    nupy array to cpp.HTuple
    """
    if not ar.flags['C_CONTIGUOUS']:
        ar = np.ascontiguousarray(ar)
    cdef HTuple t = HTuple.from_array(ar)
    return t.me


cdef class HPose:
    """
    A HPose represent a position and an orientation in Halcon.
    Takes either
    * nothing
    * a pos vector
    * a pos vector + axis angle (6 tall)
    * a pos vector + euler rotation + encoding (see Halcon doc)
    """

    cdef cpp.HPose me

    def __cinit__(self, *args):
        if len(args) == 0:
            self.me = cpp.HPose()
        elif len(args) == 1:
            # suppose this is math3d transform
            self.me = _trans2hpose(args[0])
        elif len(args) == 3:
            # we suppose this is a vector
            self.me = cpp.HPose(<double> args[0], <double> args[1], <double> args[2], 0, 0, 0, "Rp+T", "gba", "point")
        elif len(args) == 6:
            # we suppose this is a vector with axis angle, from ur for example
            self.me = cpp.HPose(<double> args[0], <double> args[1], <double> args[2], <double> args[3], <double> args[4], <double> args[5], "Rp+T", "rodriguez", "point")
        elif len(args) == 7:
            # this is an Halcon encoded rotation see documentation
            self.me = cpp.HPose(_list2tuple(args))
        else:
            raise ValueError("HPose argument not understood", args)

    def to_list(self):
        tup = HTuple()
        tup.me = self.me.ConvertToTuple()
        return tup.to_list()

    def __repr__(self):
        return "HPose({})".format(self.to_list())

    def to_quaternion(self):
        """
        Return list containing:
          [:3] Pos vector
          [3:] Rotation expressed as quaternion
        """
        pos = self.to_list()[:3]
        ori = _pose2quat(self.me)
        return pos + ori

    def to_transform(self):
        """
        Return a math3d transform.
        """
        pos = m3d.Vector(self.to_list()[:3])
        ori = m3d.Orientation(m3d.UnitQuaternion(*_pose2quat(self.me)))
        return  m3d.Transform(ori, pos)



cdef cpp.HPose _trans2hpose(trans):
    cdef int rx, ry, rz 
    rx, ry, rz = trans.orient.to_euler("xyz")
    cdef cpp.HPose pose = cpp.HPose(trans.pos.x, trans.pos.y, trans.pos.z, rx, ry, rz, "Rp+T", "gba", "point")
    return pose


cdef _pose2quat(cpp.HPose& pose):
    cdef cpp.HQuaternion q 
    q.PoseToQuat(pose)
    tup = HTuple()
    tup.me = q.ConvertToTuple()
    return tup.to_list()


cdef _hposear2list(cpp.HPoseArray& ar):
    poses = []
    for i in range(ar.Length()):
        p = HPose()
        p.me = <cpp.HPose> ar.Data()[i]
        poses.append(p)
    return poses

cdef _model_array_to_model_list(cpp.HObjectModel3DArray& ar):
    res = []
    for i in range(ar.Length()):
        m = Model3D()
        m.me = ar.Tools()[i]
        res.append(m)
    return res 


cdef class Surface:

    cdef cpp.HSurfaceModel me

    def __cinit__(self):
        self.me = cpp.HSurfaceModel()

    @staticmethod
    def from_file(path):
        surf = Surface()
        surf.me = cpp.HSurfaceModel(path.encode())
        return surf

    def find_surface_model(self, Model3D model, double rel_sample_dist=0.05, double key_point_fraction=0.2, double min_score=0.5, params=None):
        """
        Find our surface in a scene. Read Halcon documentation for more
        params is a dict of parameters in Halcon style

        List of params values (Halcon 13): "3d_edge_min_amplitude_abs", "3d_edge_min_amplitude_rel", "3d_edges", "dense_pose_refinement", "max_overlap_dist_abs", "max_overlap_dist_rel", "num_matches", "pose_ref_dist_threshold_abs", "pose_ref_dist_threshold_rel", "pose_ref_num_steps", "pose_ref_scoring_dist_abs", "pose_ref_scoring_dist_rel", "pose_ref_sub_sampling", "pose_ref_use_scene_normals", "scene_normal_computation", "score_type", "sparse_pose_refinement", "viewpoint"
        """
        if params is None:
            params = {}
        cdef cpp.HString reHandle
        score = HTuple()
        cdef cpp.HSurfaceMatchingResultArray sres
        cdef cpp.HPoseArray pose_array = self.me.FindSurfaceModel(model.me, rel_sample_dist, key_point_fraction, cpp.HTuple(min_score), cpp.HString(b"false"), _list2tuple(params.keys()), _list2tuple(params.values()), &score.me, &sres)
        poses = _hposear2list(pose_array)
        return poses, score.to_list()

    def find_surface_model_image(self, Image img, Model3D model, double rel_sample_dist=0.05, double key_point_fraction=0.2, double min_score=0.5, params=None):
        """
        Find our surface in scene and 2d image. Read Halcon documentation for more
        params is a dict of parameters in Halcon style

        List of params values (Halcon 13): "3d_edge_min_amplitude_abs", "3d_edge_min_amplitude_rel", "3d_edges", "dense_pose_refinement", "max_overlap_dist_abs", "max_overlap_dist_rel", "num_matches", "pose_ref_dist_threshold_abs", "pose_ref_dist_threshold_rel", "pose_ref_num_steps", "pose_ref_scoring_dist_abs", "pose_ref_scoring_dist_rel", "pose_ref_sub_sampling", "pose_ref_use_scene_normals", "scene_normal_computation", "score_type", "sparse_pose_refinement", "viewpoint"
        """
        if params is None:
            params = {}
        cdef cpp.HString reHandle
        score = HTuple()
        cdef cpp.HSurfaceMatchingResultArray sres
        cdef cpp.HPoseArray pose_array = self.me.FindSurfaceModelImage(img.me, model.me, rel_sample_dist, key_point_fraction, cpp.HTuple(min_score), cpp.HString(b"false"), _list2tuple(params.keys()), _list2tuple(params.values()), &score.me, &sres)
        poses = _hposear2list(pose_array)
        return poses, score.to_list()

    def refine_pose(self, Model3D model, HPose pose, double min_score):
        new_pose = HPose()
        cdef cpp.HString handle
        cdef cpp.HSurfaceMatchingResult sres

        names = []
        vals = []
        score = HTuple()
        new_pose.me = self.me.RefineSurfaceModelPose(model.me, pose.me, min_score, cpp.HString("false"),  _list2tuple(names), _list2tuple(vals), &score.me, &sres)
        return new_pose, score.to_list()

    def set_cam_params(self, str path):
        cdef cpp.HCamPar cp
        cp.ReadCamPar(cpp.HString(path.encode()))
        self.me.SetSurfaceModelParam(cpp.HString(b"camera_parameter"), cpp.HTuple(cp))

    def to_file(self, path):
        self.me.WriteSurfaceModel(path.encode())               

cdef class Model3D:

    cdef cpp.HObjectModel3D me

    def __cinit__(self):
        self.me = cpp.HObjectModel3D()

    @property
    def diameter(self):
        return self._get_diameter()

    @staticmethod
    cdef from_cpp(cpp.HObjectModel3D obj):
        m = Model3D()
        m.me = obj
        return m

    @staticmethod
    def from_file(str path, str scale, params=None):
        if params is None:
            params = {}
        model = Model3D()
        cdef bytes bscale = scale.encode() # no idea why I need thi intemediary step for HTuple and not for HString??
        cdef cpp.HTuple status;
        model.me = cpp.HObjectModel3D(cpp.HString(path.encode()), cpp.HTuple(bscale), _list2tuple(params.keys()), _list2tuple(params.values()), &status)
        return model
    
    @staticmethod
    def from_array(ar):
        model = Model3D()
        model.me = cpp.HObjectModel3D(_ar2ht(ar[:, 0]), _ar2ht(ar[:, 1]), _ar2ht(ar[:, 2]))
        return model

    def get_bounding_box(self, oriented=True):
        cdef double x, y, z
        pose = HPose()
        pose.me = self.me.SmallestBoundingBoxObjectModel3d("oriented", &x, &y, &z)
        #p = HTuple()
        #p.me = pose.ConvertToTuple()
        return pose, (x, y, z)

    def _get_diameter(self):
        cdef cpp.HTuple diameter = self.me.GetObjectModel3dParams(_list2tuple([b"diameter_axis_aligned_bounding_box"]))
        ar = _ht2ar(diameter)
        return ar[0]

    def to_array(self):
        cdef cpp.HTuple points = self.me.GetObjectModel3dParams(_list2tuple([b"point_coord_x", b"point_coord_y", "point_coord_z"]))
        ar = _ht2ar(points)
        ar.shape = (3, points.Length()/3)
        return ar.transpose()

    def normals_to_array(self):
        cdef cpp.HTuple points = self.me.GetObjectModel3dParams(_list2tuple([b"point_normal_x", b"point_normal_y", "point_normal_z"]))
        ar = _ht2ar(points)
        ar.shape = (3, points.Length()/3)
        return ar.transpose()

    def points_and_normals_to_array(self, divider=1):
        pts = self.to_array()
        nls = self.normals_to_array(divider)
        c = np.hstack((pts, nls))
        return c

    def get_convex_hull(self):
        m = Model3D()
        m.me = self.me.ConvexHullObjectModel3d()
        return m

    def sampled(self, str method="fast", double dist=0.01, max_angle_diff=None, min_num_points=None ):
        """
        Return a sampled copy of point cloud
        args: methods and distance
        methods: accurate, accurate_use_normals, fast, fast_compute_normals
        """
        m = Model3D()
        names = []
        vals = []
        if max_angle_diff is not None:
            names.append(b"max_angle_diff")
            vals.append(max_angle_diff)
        if min_num_points is not None:
            names.append(b"min_num_points")
            vals.append(min_num_points)
        m.me = self.me.SampleObjectModel3d(method.encode(), dist, _list2tuple(names), _list2tuple(vals))
        return m

    def smoothed(self, int knn=60, int order=2):
        m = Model3D()
        names = []
        vals = []
        names.append(b"mls_kNN")
        vals.append(knn)
        names.append(b"mls_order")
        vals.append(order)
        m.me = self.me.SmoothObjectModel3d(b"mls", _list2tuple(names), _list2tuple(vals))
        return m

    def create_surface_model(self, double dist, invert_normals="false", train_edges="false"):
        s = Surface()
        names = []
        vals = []
        names.append(b"model_invert_normals")
        names.append(b"train_3d_edges")
        vals.append(invert_normals)
        vals.append(train_edges)
        s.me = self.me.CreateSurfaceModel(dist, _list2tuple(names), _list2tuple(vals))
        return s

    def compute_normals(self, int knn, int order, force_inwards=False):
        """
        Compute normals
        """
        m = Model3D()
        names = []
        vals = []
        names.append(b"mls_kNN")
        vals.append(knn)
        names.append(b"mls_order")
        vals.append(order)
        if force_inwards:
            names.append(b"mls_force_inwards")
            vals.append("true")

        m.me = self.me.SurfaceNormalsObjectModel3d(b"mls", _list2tuple(names), _list2tuple(vals))
        return m

    def to_file(self, str filetype, str path):
        self.me.WriteObjectModel3d(cpp.HString(filetype.encode()), cpp.HString(path.encode()), cpp.HTuple(), cpp.HTuple())               

    def select_points(self, str attr, double min_val, double max_val): 
        """
        attr is one of:
            point_coord_x
            point_coord_y
            point_coord_z
            point_normal_x
            point_normal_y
            point_normal_z
            etc

        """
        m = Model3D()
        m.me = self.me.SelectPointsObjectModel3d(attr.encode(), min_val, max_val)
        return m

    def select_x(self, min_val, max_val):
        return self.select_points("point_coord_x", min_val, max_val)

    def select_y(self, min_val, max_val):
        return self.select_points("point_coord_y", min_val, max_val)

    def select_z(self, min_val, max_val):
        return self.select_points("point_coord_z", min_val, max_val)

    def transformed(self, HPose pose):
        m = Model3D()
        m.me = self.me.RigidTransObjectModel3d(pose.me)
        return m
    
    @staticmethod
    #def union(Model3D[] models);  # syntax not supported?
    def union(Model3D m0, Model3D m1, Model3D m2=None):
        cdef int nb = 2
        if m2 is not None:
            nb += 1
        cdef cpp.HObjectModel3D car[10]
        car[0] = m0.me
        car[1] = m1.me
        if m2 is not None:
            car[2] = m2.me
        cdef cpp.HObjectModel3DArray ar = cpp.HObjectModel3DArray(car, nb)
        m = Model3D()
        m.me = cpp.HObjectModel3D.UnionObjectModel3d(ar, cpp.HString(b"points_surface"))
        return m

    def fit_primitive(self, params=None):
        """
        fit a primitive. See Halcon doc
        params is a dic of key val. for example:
        {"primitive_type": "cylinder", min_radius=0.1, max_radius:0.2, "fitting_algorithm":"least_squares"}
        primitive_type is of: cylinder, plane, sphere, all

        """
        if params is None:
            params = {}
        cdef cpp.HObjectModel3DArray ar = cpp.HObjectModel3DArray(&self.me, 1)
        cdef cpp.HObjectModel3DArray results = cpp.HObjectModel3D.FitPrimitivesObjectModel3d(ar, _list2tuple(params.keys()), _list2tuple(params.values()))
        res = []
        for i in range(results.Length()):
            m = Model3D()
            m.me = results.Tools()[i]
            res.append(m)
        return res 

    def fit_primitive2(self, params=None):
        """
        fit a primitive. See Halcon doc
        params is a dic of key val. for example:
        {"primitive_type": "cylinder", min_radius=0.1, max_radius:0.2, "fitting_algorithm":"least_squares"}
        primitive_type is of: cylinder, plane, sphere, all

        """
        if params is None:
            params = {}
        m = Model3D()
        m.me = self.me.FitPrimitivesObjectModel3d(_list2tuple(params.keys()), _list2tuple(params.values()))
        return m


    def prepare(self, purpose, overwrite=True, params=None):
        if overwrite:
            overwrite = b"true"
        else:
            overwrite = b"false"
        if params is None:
            params = {}
        self.me.PrepareObjectModel3d(bytes(purpose, "utf-8"), overwrite, _list2tuple(params.keys()), _list2tuple(params.values()))

    def segment(self, params=None):
        """
        Segment a set of 3D points
        params are: "fitting_algorithm", "max_curvature_diff", "max_orientation_diff", "max_radius", "min_area", "min_radius", "output_point_coord", "output_xyz_mapping", "primitive_type"
        """
        if params is None:
            params = {}
        m = Model3D()
        print("RUN with", params.keys(), params.values())
        m.me = self.me.SegmentObjectModel3d(_list2tuple(params.keys()), _list2tuple(params.values()))
        return m

    def segment2(self, params=None):
        if params is None:
            params = {}
        print("RUN with", params.keys(), params.values())
        cdef cpp.HObjectModel3DArray ar = cpp.HObjectModel3DArray(&self.me, 1)
        cdef cpp.HObjectModel3DArray results = cpp.HObjectModel3D.SegmentObjectModel3d(ar, _list2tuple(params.keys()), _list2tuple(params.values()))
        return _model_array_to_model_list(results)

    def distance(self, Model3D model, double max_dist=0, params=None):
        """
        Compute distance. result is stored in attribute 'distance'
        """
        if params is None:
            params = {}
        self.me.DistanceObjectModel3d(model.me, cpp.HPose(), cpp.HTuple(max_dist), _list2tuple(params.keys()), _list2tuple(params.values()))

    def get_attribute(self, params):
        """
        return value of one or more attributes hacon tuple
        possible values in halcon 13 are: 
"blue", "bounding_box1", "center", "diameter_axis_aligned_bounding_box", "extended_attribute_names", "green", "has_distance_computation_data", "has_extended_attribute", "has_lines", "has_point_normals", "has_points", "has_polygons", "has_primitive_data", "has_primitive_rms", "has_segmentation_data", "has_shape_based_matching_3d_data", "has_surface_based_matching_data", "has_triangles", "has_xyz_mapping", "lines", "mapping_col", "mapping_row", "neighbor_distance", "num_extended_attribute", "num_lines", "num_neighbors", "num_neighbors_fast", "num_points", "num_polygons", "num_primitive_parameter_extension", "num_triangles", "point_coord_x", "point_coord_y", "point_coord_z", "point_normal_x", "point_normal_y", "point_normal_z", "polygons", "primitive_parameter", "primitive_parameter_extension", "primitive_parameter_pose", "primitive_pose", "primitive_rms", "primitive_type", "red", "reference_point", "score", "triangles"
        """
        if not isinstance(params, list):
            params = [params]
        tup = HTuple()
        tup.me = self.me.GetObjectModel3dParams(_list2tuple(params))
        return tup
    
    def get_connected_components(self, feature, double value):
        """
        Get connected componented computed using one of following features:
        "angle", "distance_3d", "distance_mapping", "lines", "mesh"
        """
        cdef cpp.HObjectModel3DArray results = self.me.ConnectionObjectModel3d(feature.encode(), value) 
        return _model_array_to_model_list(results)
 
    def triangulate(self, method):
        """
        Method (string): 'polygon_triangulation', 'greedy' or 'implicit'
        """
        m = Model3D()
        names = []
        vals = []
        cdef long* info
        m.me = self.me.TriangulateObjectModel3d(method.encode(), _list2tuple(names), _list2tuple(vals), info)
        return m

cdef class Plane(Model3D):
    def __init__(self, HPose pose, xext_vect, yext_vect):
        Model3D.__init__(self)
        xext_vect = list(xext_vect)
        yext_vect = list(yext_vect)
        self.me.GenPlaneObjectModel3d(pose.me, _list2tuple(xext_vect), _list2tuple(yext_vect))


cdef class Sphere(Model3D):
    def __init__(self, double x, double y , double z, double radius):
        Model3D.__init__(self)
        self.me.GenSphereObjectModel3dCenter(x, y, z, radius)


cdef class Box(Model3D):
    def __init__(self, HPose pose, double x, double y, double z):
        Model3D.__init__(self)
        self.me.GenBoxObjectModel3d(pose.me, x, y, z)


cdef class Region:

    cdef cpp.HRegion me

    def __cinit__(self, path=None):
        self.me = cpp.HRegion()


cdef class Image:

    cdef cpp.HImage me

    def __cinit__(self, path=None):
        if path is None:
            self.me = cpp.HImage()
        else:
            self.me = cpp.HImage(cpp.HString(path.encode()))

    def to_array(self):
        #FIXME
        cdef int n = self.me.Width()[0].L() * self.me.Height()[0].L()
        #result = cnp.empty(n, dtype=np.long_t)
        #for i in range(n):
            #result[i] = self.me[i]
        #result.shape = self.me.Width()[0].L(), self.me.Height()[0].L()
        #return result

    def write(self, str path, fmt=None, long fillcolor=0):
        """
        suggested format: 
"tiff", "tiff mask", "tiff alpha", "tiff deflate 9", "tiff deflate 9 alpha", "tiff jpeg 90", "tiff lzw", "tiff lzw alpha ", "tiff packbits", "bigtiff", "bigtiff mask", "bigtiff alpha", "bigtiff deflate 9", "bigtiff deflate 9 alpha", "bigtiff jpeg 90", "bigtiff lzw", "bigtiff lzw alpha ", "bigtiff packbits", "bmp", "jpeg", "jpeg 100", "jpeg 80", "jpeg 60", "jpeg 40", "jpeg 20", "jp2", "jp2 50", "jp2 40", "jp2 30", "jp2 20", "jpegxr", "jpegxr 50", "jpegxr 40", "jpegxr 30", "jpegxr 20", "png", "png best", "png fastest", "png none", "ima", "hobj"
        """
        if fmt is None:
            if path[-4:] == '.jpg':
                fmt = b'jpeg 60'
            elif path[-4:] == '.png':
                fmt = b'png'
            elif path[-5:] == '.tiff':
                fmt = b'tiff'
            elif path[-4:] == '.bmp':
                fmt = b'bmp'
            else:
                raise Exception('Please specifiy format')
        self.me.WriteImage(cpp.HString(fmt), fillcolor, cpp.HString(path.encode()))


    @staticmethod
    def from_file(str path):
        im = Image()
        im.me = cpp.HImage(cpp.HString(path.encode()))
        return im

    @property
    def width(self):
        ht = HTuple()
        ht.me = self.me.Width()
        return ht.to_list()[0]
 
    @property
    def height(self):
        ht = HTuple()
        ht.me = self.me.Height()
        return ht.to_list()[0]

    def croped(self, int x1, int y1, int x2, int y2):
        im = Image()
        im.me = self.me.CropRectangle1(x1, y1, x2, y2)
        return im

    def inspect_shape_model(self, int level, int contrast):
        reg = Region()
        im = Image()
        im.me = self.me.InspectShapeModel(&reg.me, level, contrast)
        return im, reg



