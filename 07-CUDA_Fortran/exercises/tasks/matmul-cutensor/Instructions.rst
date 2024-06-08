Instructions
============

The file ``mmul-GPU.cuf `` is to be GPU-accelerated  using cutensorex library.

Fortran array intrinsics using Tensor Cores
-----------------------------------

Please have a look at the mmul-GPU.cuf  file and work on the indicated lines (see 'TODO's).


* Use ``device`` attribute to declare device arrays.  

    .. code-block:: Fortran

    	integer,allocatable,device  :: myInt(:)


* Use Fortran array notation for data transfer.  

    .. code-block:: Fortran

    	myArr=myArr_d


* Remember to include  cutensorex! 

    .. code-block:: Fortran

    	use cutensorex

* Be sure to load the custom modules of this task.

    .. code-block:: bash

        source setup.sh

* For compilation, use  

    .. code-block:: bash

		make

* To run your code, call ``srun`` with the correct parameters. A shortcut is given via  

    .. code-block:: bash

		make run

