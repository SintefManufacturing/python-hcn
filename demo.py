import numpy as np
import hcn
from IPython import embed, get_ipython

import vtk_visualizer as vv

if __name__ == "__main__":
    m = hcn.Model3D.from_file("b4.obj", "m")
    #vv.plotxyz(m.to_array(), block=True)
    #sm = m.smoothed(knn=200, order=1)
    #vv.plotxyz(sm.to_array(), block=True)
    #new = sm.sampled(0.01)
    #vv.plotxyz(new.to_array(), block=True)
    nn = m.compute_normals(20, 2)
    a = nn.to_array()
    print(a)
    b = nn.normals_to_array()
    print(b)
    res = np.hstack((a, b))
    print(res)
    vv.plothh(res, block=True)
    #embed()
