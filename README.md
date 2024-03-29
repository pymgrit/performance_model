# Parallel-in-time performance analysis

Python tool for task graph based performance analysis of parallel-in-time methods. Currently the methods

* Parareal [[1]](#1)
* Parallel full approximation scheme in space and time (PFASST) [[2]](#2)
* Multigrid reduction in time (MGRIT) [[3]](#3)

are supported.

The model was compared to the following parallel-in-time libraries:

* PararealF90 [[4]](#4)
* LibPFASST [[5]](#5)
* PySDC [[6]](#6)
* XBRAID [[7]](#7)
* PyMGRIT [[8]](#8)

## References
<a id="1">[1]</a> 
Jacques-Louis Lions, Yvon Maday, and Gabriel Turinici. “Résolution d’EDP par un schéma en temps
“pararéel””. In: Comptes Rendus de l’Académie des Sciences. Série I. Mathématique 332.7 (2001),
pp. 661–668.

<a id="2">[2]</a> 
Matthew Emmett and Michael Minion. “Toward an efficient parallel in time method for partial differ-
ential equations”. In: Communications in Applied Mathematics and Computational Science 7.1 (2012),
pp. 105 –132.

<a id="3">[3]</a> 
Robert D. Falgout, Stephanie Friedhoff, Tzanio Kolev, Scott MacLachlan, and Jacob B. Schroder.
“Parallel time integration with multigrid”. In: SIAM Journal on Scientific Computing 36.6 (2014),
pp. C635–C661.

<a id="3">[4]</a> 
https://github.com/Parallel-in-Time/PararealF90

<a id="3">[5]</a> 
https://github.com/libpfasst/LibPFASST

<a id="3">[6]</a> 
https://github.com/Parallel-in-Time/pySDC

<a id="3">[7]</a> 
https://www.llnl.gov/casc/xbraid

<a id="3">[8]</a> 
https://github.com/pymgrit/pymgrit