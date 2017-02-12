import numpy as np
import hcn
from IPython import embed, get_ipython

import vtk_visualizer as vv

if __name__ == "__main__":
    m = hcn.Model3D.from_file("b4.obj", "m")
    m = m.select_x(0, 1)
    m = m.select_y(0, 1)
    m = m.select_z(0.01, 1)
    #vv.plotxyz(m.to_array(), block=True)
    #sm = m.smoothed(knn=1000, order=3)
    #vv.plotxyz(sm.to_array(), block=True)
    #vv.plotxyz(m.to_array(), block=True)
    #ch = m.get_convex_hull()
    #vv.plotxyz(ch.to_array(), block=True)
    #new = sm.sampled(0.01)
    #vv.plotxyz(new.to_array(), block=True)
    #nn = m.compute_normals(20, 2)
    #a = nn.to_array()
    #print(a)
    #b = nn.normals_to_array()
    #print(b)
    #res = np.hstack((a, b))
    #print(res)
    #vv.plothh(res, block=True)
    s = m.create_surface_model(0.1)
    embed()
