# murphy

This is a library for structured prediction. It was primarily developed by Taylor Berg-Kirkpatrick. Other contributors include John DeNero, Aria Haghighi, Dan Klein, Jonathan Kummerfeld, and Adam Pauls.

To use the library, download it one of these ways, and include it in your code as described below:

- [Download .zip](https://github.com/tberg12/murphy/zipball/master)
- [Download .tar.gz](https://github.com/tberg12/murphy/tarball/master)
- `git clone https://github.com/tberg12/murphy.git`

This library was initially released as supplementary material for the experiments described in:

  [An Empirical Analysis of Optimization for Max-Margin NLP](https://aclweb.org/anthology/D15-1369)
  Jonathan K. Kummerfeld, Taylor Berg-Kirkpatrick and Dan Klein
  EMNLP 2015

## Using the library

You will need to implement the following:

- Code that creates a LossAugmentedLearner, e.g. PrimalSubgradientSVMLearner, and calls train. This is your main interface to the learning code. At creation you set parameters such as the learning rate and regularization, and when you call train you provide the data, initial weights, and number of iterations.
- A class that extends LossAugmentedLinearModel. This is the interface through which the learning code calls your inference procedure. Given an instance, your code will find the best structure (b) under your model with loss-augmentation relative to the gold (g). You will return an UpdateBundle that includes the loss of b relative to g, and the features active in b and g.
- [if using sparse updates] Modifications to your inference procedure to use getCount on a LazyAdaGradResult to get weights. This handles the delayed updates (see the paper above for further informtation).

Then include this code on your classpath, compile, and you're ready to go!

TODO:
When to use floatstructpred v structpred v lazystructpred ?
