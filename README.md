===========================================================================
                          Source code for the artice 
   
      A single cell spiking model for the origin of grid-cell patterns
                    by Tiziano D'Albis and Richard Kempter
                      doi: 10.1371/journal.pcbi.1005782

============================================================================


REQUIREMENTS
============

All source code is in Python language and was tested in the following environmnent:

- Python 2.7.12
- Brian2 2.0b4
- Matplotlib 1.4.3 
- NumPy 1.9.3 
- Scipy 0.17.1	

The easiest way to run the code is to create an Anaconda environment using the file 'conda_env.yml' provided in the root folder of this package.
This can be done with the command:

conda env create -f conda_env.yml

This will create a new conda environment named 'grid_origin'. Note that this environment may contain more packages
then actually needed (i.e., it provides more than the minimal set of required dependencies).

Note that you need approximately XXX GB of free space on disk to store all the required simulation results.


GENERATING FIGURES
==================

Each figure of the paper is generated via a separate script named figXX.py where XX is a number from 1 to 11 (plus eventually an extra suffix for subplots).
Scripts should be run from within the 'code' subfolder; e.g., type 'python fig1.py' to generate Figure 1. 
Figures in EPS format will be saved in a 'figures' folder that is automatically generated at the first run. 
Note that only figXX.py scripts need to be run; all the other Python files are automaticallly imported and called from these scripts.

Most figures require data that is generated at the end of one or more numerical simulations.
If the required simulation data is not yet present on disk, it will be automatically created by the figure script itself and saved in the 'results' folder.
Note that running all simulations requires several days of computation (estimated running time for each simulatin is printed on the console at launch).
Simulations results are permenantly saved to disk for future plotting ('results' folder). 


CONFIGURATION
=============

Some of the scripts launch multiple jobs in parallel to reduce waiting time. The maximum number of parallel jobs is set to 7 by the default.
This number can be customized (and be made machine dependent) by adding an antry in the 'procs_by_host' dictionary within the 'grid_batch.py'file: the key is hostname of the machine, the value is the number of parallel processes to be run. 

The default parameter values of all simulations are found in the file 'grid_params.py'.

