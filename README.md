# NormFlow
Normalizing flow code in Matlab, based on the material in "Normalizing Flows for Probabilistic Modeling and Inference" by Papamakarios et al. The current implementation is forward-feed, fitting a distribution to lots of generated data with the goal of being able to generate more using the simple transformations in NF rather than whatever more computationally intense method (e.g. Euler-Maruyama integration).

The only code guaranteed to work right now is the affine autoregressive NF, corresponding to equation (32) in the paper. It's in one dimension and the jacobian can be analytically calculated. The non-affine NF file is much more ambitions, hoping for it to work in arbitrary dimensions with arbitrary numbers of layers and neural depth. It's a work in progress!
