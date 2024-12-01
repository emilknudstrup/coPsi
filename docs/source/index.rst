.. coPsi documentation master file, created by
   sphinx-quickstart on Thu Sep 22 16:34:10 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

coPsi
=================================

**Stellar rotation, inclination, and orientation**

coPsi is made to derive stellar rotation periods from photometric light curves, then *properly* calculate the stellar inclination, and ultimately measure the obliquity :math:`\psi` through:

.. math::
   \cos \psi = \sin i_\star \sin i_{\rm o} \cos \lambda + \cos i_\star \cos i_{\rm o} \, ,

where :math:`i_\star` is the stellar inclination, :math:`i_{\rm o}` is the orbital inclination, and :math:`\lambda` is the projected obliquity.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. toctree::
   :maxdepth: 1
   :caption: Usage

   examples/empirical.ipynb
   examples/mcmc.ipynb
   examples/rotation.ipynb
   examples/rotation_pg.ipynb

.. toctree::
   :maxdepth: 2
   :caption: API
   
   API/Phot
   API/Prot
   API/coPsi
   API/priors

.. toctree::
   :maxdepth: 2
   :caption: References

   references

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
