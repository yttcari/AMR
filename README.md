An adaptive refinement code.

Sample Usage:
```
import problem

mesh_old = problem.run_blast(Nx=32, max_level=2, init=True, mode='between')
plot(mesh_old)

mesh_new = problem.run_blast(Nx=32, max_level=2, init=True, mode='in_step')
plot(mesh_new)

RMSE_error(mesh_old, mesh_uniform, plot_var='rho')
```