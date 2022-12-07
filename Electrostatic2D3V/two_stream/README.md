```
source ~/spack/share/spack/setup-env.sh
spack load nektar 
spack load hipsycl 
spack load hdf5 
spack load googletest
wget https://raw.githubusercontent.com/will-saunders-ukaea/NESO-workspace/main/Electrostatic2D3V/two_stream/two_stream_conditions.xml
wget https://raw.githubusercontent.com/will-saunders-ukaea/NESO-workspace/main/Electrostatic2D3V/two_stream/two_stream_mesh.xml
OMP_NUM_THREADS=1 mpirun -n 12 /NESO/Electrostatic2D3V.x two_stream_conditions.xml two_stream_mesh.xml
```

This should produce an output `Electrostatic2D3V_field_trajectory.h5` that can be plotted with

```
python3 plot_energy.py two_stream_conditions.xml electrostatic_two_stream.h5
```



