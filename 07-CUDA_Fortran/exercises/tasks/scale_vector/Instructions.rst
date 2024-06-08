Instructions
============

In this exercise, we'll scale a vector (array) of single-precision numbers by a scalar. You'll learn 
to write simple CUDA Fortran codes employing explicit data transfer as well as unified memory.

The first CUDA Fortran program
------------------------------

Take a look at ``scale_vector.cuf`` and ``scale_vector_um.cuf`` and look for TODOs.

* The global attribute is necessary to specify your kernels.  

    .. code-block:: Fortran

		attributes(global) subroutine mySub

* Since Fortran 2003, you can pass argument by value using ``value``  type declaration attribute.  

    .. code-block:: Fortran

 		integer,value :: myInt

* Use ``device`` attribute to declare device arrays.  

    .. code-block:: Fortran

    	integer,allocatable,device  :: myInt(:)

* Use ``managed`` attribute to declare  unified memory arrays.  

    .. code-block:: Fortran

    	integer,managed,allocatable  :: myInt(:)

* If necessary, use native Fortran copy notation to transfer data.  

    .. code-block:: Fortran

    	myArr=myArr_d

* Be sure to load the custom modules of this task.

    .. code-block:: bash

        source setup.sh


* For compilation, use  

    .. code-block:: bash

		make

* To run your code, call ``srun`` with the correct parameters. A shortcut is given via  

    .. code-block:: bash

		make run
		or
		make run-um




