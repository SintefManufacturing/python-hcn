import halpy

if __name__ == "__main__":
    t = halpy.HTuple()
    d = halpy.HTuple.from_double(0.199)
    print("LENGTH", t.length())
    print("LENGTH", d.length())
    print("val[0]", d[0])
    print("val array", d.to_array())
