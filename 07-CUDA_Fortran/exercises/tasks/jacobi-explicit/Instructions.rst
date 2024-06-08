Instructions
============

The file ``poisson2d.cuf`` is to be GPU-accelerated using explicit Kernels.

Jacobi solver with explicit Kernels
-----------------------------------

Please have a look at the poisson2d.cuf file and work on the indicated lines (see 'TODO's).


* Use ``device`` attribute to declare device arrays.  

    .. code-block:: Fortran

    	integer,allocatable,device  :: myInt(:)


* Use Fortran array notation for data transfer.  

    .. code-block:: Fortran

    	myArr=myArr_d


* The global attribute is necessary to specify your kernels.  

    .. code-block:: Fortran

		attributes(global) subroutine mySub

* Use ``<<<,>>>`` to lunch your kernels.

    .. code-block:: Fortran

    	call yourKernel<<<gridDim,blockDim>>>(arg1,arg2)

* Use ``dim3`` type to define variables for your lunch configuration.

    .. code-block:: Fortran

    	type(dim3) :: blockDim, gridDim

* Be sure to load the custom modules of this task.

    .. code-block:: bash

        source setup.sh

* For compilation, use  

    .. code-block:: bash

		make

* To run your code, call ``srun`` with the correct parameters. A shortcut is given via  

    .. code-block:: bash

		make run

