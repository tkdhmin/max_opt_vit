## How to build
### Prerequiste
Make sure that you already have downloaded `googletest` on the external directory in the root of the project.
```bash
pwd # Make sure that the current location is a `max_opt_vit` directory.
mkdir external
cd external
git clone https://github.com/google/googletest.git
```

### 1. Create `build` directory
```bash
mkdir build
cd build
```

### 2. Create Makefile using cmake
```bash
cmake ..
```
When you face a problem of path for finding CUDA or CUDAtoolkit, then adjust your path correctly.

It is also possible to add prefix path to cmake like the following:

```bash
cmake -DCMAKE_PREFIX_PATH=/usr/local/cuda ..
```

### 3. Build!
```bash
cmake --build .
```

### 4. Execute the binary
```bash
./my_exec
```

### 5. Execute googletest for testing the specific functionality
```bash
./test_launch_kernel
```

### 6. Execute googletest to test all.
```bash
ctest
```