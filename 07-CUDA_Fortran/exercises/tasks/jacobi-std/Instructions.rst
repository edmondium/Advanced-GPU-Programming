Instructions
============

The file ``poisson2d.f90`` is to be GPU-accelerated with do concurrent.

Jacobi solver with with do concurrent
-----------------------------------------

Please have a look at the file and work on the indicated lines (see `TODO`s).

*  `DO CONCURRENT (...) [locality-spec]` can be used to accelerate the loop.

    .. code-block:: Fortran

		do concurrent (i = 1: n, j=1,m)


* Compare the results with the explicit and CUF kernel versions.

* Be sure to load the custom modules of this task.

    .. code-block:: bash

    	source setup.sh

* For compilation, use  

    .. code-block:: bash

		make

* To run your code, call ``srun`` with the correct parameters. A shortcut is given via  

    .. code-block:: bash

		make run





