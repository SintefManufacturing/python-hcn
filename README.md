
LGPL Cython based Python wrapper for the proprietary vision system Halcon.

It exposes only the point cloud related API. Not all methods are exposed but the ground work is done and exposing new methods should only require exposing them in hcn.pyx and maybe pxd.

Exposing new classes is a bit more work but not much 

Example code for 3D matching:

    import numpy as np
    import hcn

    if __name__ == "__main__":
        # first read CAD model and create surface model
        mod = hcn.Model3D.from_file("CAD/KA1.STL", "mm")
        mod = mod.sampled("fast_compute_normals", 0.001)
        surf = mod.create_surface_model(0.001, invert_normals="true")

        #now read our scene
        scene = hcn.Model3D.from_file("punktskyer/scene_ka1_simple.ply", "mm")
        scene = scene.compute_normals(60, 2)

        poses, score = surf.find_surface_model(scene, 0.001, 0.2, min_score=0, params={"num_matches":1})
        print("Found ", len(score), "matches: ", poses)

FAQ:
* Is it usable?

yes we use it to build prototypes

* Why not use Hirsch bindings: https://github.com/dov/hirsch

Hirsch is only python2 and the 2D halcon API. We wanted Python3 and 3D API. So our choice was either extend it or write our own. The developers of hirsch did not answer our questions so we chose to develop our own since very little of hirsch seemed to be usefull to us

* Why not autogereating code as Hirsch does?

We could autogenerate pxd and a patch is welcome.
The pyx file implements custom python classes and most of it seems to be hard to automate while keeping a pythonic API
