
LGPL Cython based Python wrapper for the proprietary vision system Halcon.

It exposes only the point cloud related API. Not all methods are exposed but the ground work is done and exposing new methods should only require exposing them in hcn.pyx and maybe pxd.

Exposing new classes is a bit more work but not much 


FAQ:
* Is it usable?
yes we use it to build prototypes

* Why not use Hirsch bindings: https://github.com/dov/hirsch
Hirsch is only python2 and the 2D halcon API. We wanted Python3 and 3D API. So our choice was either extend it or write our own. The developers of hirsch did not answer our questions so we chose to develop our own since very little of hirsch seemed to be usefull to us

* Why not autogereating code as Hirsch does?
We could autogenerate pxd and a patch is welcome.
pyx filen create custom python classes and most of it seems to be hard to automate while keeping a pythonic API
