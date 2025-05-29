.. numq documentation master file, created by
   sphinx-quickstart on Sat Dec 12 00:32:13 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to numq's documentation!
================================

This project accelerates the simulation of quantum circuits through two backends: NumPy for CPU, and CuPy for GPU (CUDA based). A brief introduction is available from the project's home page. Within this documentation, we list main functions exposed by this module.

.. note::

   This project is a reserach project.

.. note::
   
   Due to the singledispath mechanism, many functions are repeated, though with different signature, in this documentation. Only one of them is documented and other alternative implementations would perform the same calcualtion but on different type of inputs.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

List of functions
--------------------

..
   TODO: currently functions are manually added. Later consider auto-generation of them.

.. autosummary::
   numq.apply_isometry_to_density_matrices
   numq.apply_kraus_ops_to_density_matrices
   numq.apply_unitary_transformation_to_density_matrices
   numq.commutator
   numq.format_wavefunction
   numq.get_random_ru
   numq.get_random_wf
   numq.get_randu
   numq.load_state_into_mqb_start_from_lqb
   numq.load_states_into
   numq.make_density_matrix
   numq.partial_trace
   numq.partial_trace_wf
   numq.partial_trace_wf_keep_first
   numq.rand_herm

.. automodule:: numq
   :members:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
