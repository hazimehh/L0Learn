`fit` function
----------------

.. autofunction:: l0learn.fit


`cvfit` function
----------------

.. autofunction:: l0learn.cvfit


FitModels
---------
.. autoclass:: l0learn.models.FitModel


CVFitModels
-----------
.. autoclass:: l0learn.models.CVFitModel


Generating Functions
--------------------
.. autofunction:: l0learn.models.gen_synthetic
.. autofunction:: l0learn.models.gen_synthetic_high_corr
.. autofunction:: l0learn.models.gen_synthetic_logistic

Scoring Functions
--------------------
These functions are called by :py:meth:`l0learn.models.FitModel.score` and :py:meth:`l0learn.models.CVFitModel.score`.

.. autofunction:: l0learn.models.regularization_loss
.. autofunction:: l0learn.models.squared_error
.. autofunction:: l0learn.models.logistic_loss
.. autofunction:: l0learn.models.squared_hinge_loss



