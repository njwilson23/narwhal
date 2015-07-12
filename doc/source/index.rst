.. Narwhal documentation master file, created by
   sphinx-quickstart on Sun Jul 12 14:16:11 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Narwhal documentation
=====================

.. toctree::
   :maxdepth: 2

   index

.. image:: http://ironicmtn.com/images/narwhal.png
    :alt: Narwhal logo

Casts and CastCollections
-------------------------

Data in *Narwhal* are organized as instances of ``Cast`` or ``CastCollection``.
Casts are individual profiles, with oceanographic variables measured along some
vertical coordinate. CastCollections are agglomerations of casts representing an
individual transect or an entire cruise.

.. autoclass:: narwhal.cast.Cast
   :members:

.. autofunction:: narwhal.cast.CTDCast

.. autofunction:: narwhal.cast.LADCP

.. autofunction:: narwhal.cast.XBTCast

.. autoclass:: narwhal.cast.CastCollection
   :members:

Convenience plotting
--------------------

These functions create quick plots useful for scanning data during analysis.

.. automodule:: narwhal.plotting.plotting
   :members:

Lower-level plotting
--------------------

These classes and functions can be adapted to construct publication-quality data
visualizations.

.. automodule:: narwhal.plotting.section_plot
   :members:

.. automodule:: narwhal.plotting.property_plot
   :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

