Generate mesh with:

```
bash mesh_gen.sh periodic_structured_cartesian_square.geo
```

Run with:

```
SYCL_DEVICE_TYPE=host OMP_NUM_THREADS=1 mpirun -n 12 <path-to-Electrostatic2D3V.x> conditions.xml periodic_structured_cartesian_square.xml
```

