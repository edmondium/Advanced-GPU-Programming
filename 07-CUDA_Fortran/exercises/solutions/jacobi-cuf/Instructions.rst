Instructions
============

The file ``poisson2d.cuf`` is to be GPU-accelerated using Kernel Loop Directive (CUF kernels).

Jacobi solver with Kernel Loop Directives
-----------------------------------------

Please have a look at the file and work on the indicated lines (see `TODO`s).

*  `!$cuf kernel do[(n)] <<< grid, block[optional stream] >>>` can be used to accelerate the loop. `n` is the depth of the loop.

    .. code-block:: Fortran

		!$CUF Kernel Do(3) <<<*,*>>>



* Use Fortran array notation to transfer data.  

    .. code-block:: Fortran

    	myArr=myArr_d

* Compare the results with the explicit kernel version.

* Be sure to load the custom modules of this task.

    .. code-block:: bash

    	source setup.sh

* For compilation, use  

    .. code-block:: bash

		make

* To run your code, call ``srun`` with the correct parameters. A shortcut is given via  

    .. code-block:: bash

		make run





