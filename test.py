from IPython import embed
import halpy

if __name__ == "__main__":
    e = halpy.HTuple()
    d = halpy.HTuple(0.199)
    #d.append(9.2)
    s = halpy.HTuple("totot is back")
    print("LENGTH empty", e.length())
    #embed()
    print("LENGTH double", d.length())
    print("val[0]", d[0])
    print("val array", d.to_array())
    #print("string array", d.to_string())
    print("String", s.length())
    print("String", s[0])
    #print("String", s.to_string())
    print("Try read model")
    halpy.read_object_model_3d("arm_base.stl", "mm", None, None)
