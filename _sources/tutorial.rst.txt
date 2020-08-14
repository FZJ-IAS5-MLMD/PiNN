Tutorial
========

The following tutorial explains basic concepts in mdppp with an
example of velocity autocorrelation function.

FrameData
---------

``FrameData`` is the abstraction of a trajectory, or in general, a
collection of framewise arrays that can be loaded (at least)
iteratively. "Children" of ``FrameData`` are called ``FrameArray``\s,
which will be elaborated below.

.. code-block:: python
		
   from mdppp.io import load_multi_dump
   from mdppp.ops import tacf
   data = load_multi_dump(['prod.dump'])
   print(data, data['speed'])

FrameArray
----------

A ``FrameArray`` means a numpy-array-like data structure which is
updated whenever a ``FrameData`` (called its parent) is updated.
For instance, the values in the ``FrameData`` are ``FrameArray``\s.

The ``FrameArray`` behaves much like a numpy array, with the exception
that its mathematical operations procudes ``FrameArray``\s (also
updated according to the udpated value of itself every frame).

The actual value of a ``FrameArray`` can be retrieved with
``.eval()`` method.

.. code-block:: python
		
   v = data['speed']
   sin_v = np.sin(v)
   for i in range(3):
       data.run(1)
       print(v.eval(), sin_v.eval())
     

Subclasses of FrameArray
++++++++++++++++++++++++

The static analysis of structures can often be represented as
operations of ``FrameArray``, which performs the same operation to
each frame.

However, analysis of trajectories often involve time correlatoins (for
instance, the time average of a ``FrameArray`` or its auto-correlation
function). Those analysis requires subclasses of ``FrameArray`` which
behaves just like ``FrameArray`` but are updated different when new
frame is loaded.

Some of the subclasses are:

- ``UnwrappedCoord``: unwrapped positions of atoms
- ``TCACHE``: cached ``FrameArray``
- ``TAVG``: time average of ``FrameArray``

Operations
++++++++++

Most analysis tasks can be composed of a small number of common
operations. Those operations are implemented in mdppp and used to
write most of the recipes.

.. code-block:: python
		
   from mdppp.ops import tcache, tavg
   v_cache = tcache(v, 2)
   v_avg = tavg(v)
   for i in range(3):
       data.run(1)
       print(v.eval(), cache_v.eval(), v_avg.eval())

Although subclassing FrameArray provides great flexibility, it is
recommanded to write new operations and recipes with existing
operations instead of creating new subclass for each operation. For
instance, a (simplified) time-autocorrelation function is just a
combination of tcache and tavg.

.. code-block:: python

   def tacf(var, cache_size):
       var_cache = tcache(var, cache_size)
       acf = var_cache * var[np.newaxis, :]
       acf = np.mean(np.sum(acf, axis=2), axis=1)
       tacf = tavg(acf)
       return tacf		

Example: computing VACF
-----------------------

The following script calculates the velocity autocorrelation function with
a time window of 50 frames, on segments of 100 frames.
 
.. code-block:: python
		
   import matplotlib.pyplot as plt
   for i in range(3):
       vacf = tacf(v, 50)
       data.run(100)
       plt.plot(vacf.eval())
