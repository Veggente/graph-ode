# graph-ode
Simulation code for GeneNetWeaver ODE model data fitting

Use this module to generate the data fitting results in the paper ["From graph topology to ODE models for gene regulatory networks" by Kang et al.](https://doi.org/10.1101/2020.01.22.916114)
1. Install Python packages `pandas`, `numpy`, `scipy`, `lmfit`, and `matplotlib`.
2. Use `fit_synthetic(niter=100, tol=0.001)` to generate data fitting result in row 1, column 1 of Table 4.
3. To generate other entries, modify the following options.
    * For other columns, use `num_exp=<num_exp>`, where `<num_exp>` is the number of experiments.
    * For the 2nd row, use `module_type_fit=[2, 4]`.
    * For the 3rd row, use `col1a_activates_e1_fit=False`.
    * For the 4th row, use `ode_func_fit=repressilator_ode`.
    * For the 5th row, use `gen_net='brownian'`.
