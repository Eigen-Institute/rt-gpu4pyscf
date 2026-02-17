To install rt-GPS on a Linux system, follow these steps:

---
## Using included Dockerfiles
***Using the included Dockerfiles is a robust way to ensure all dependencies (CUDA, CuPy, and PySCF) are compatible. The ubuntu_devel Dockerfile provides a clean development environment.***

#### 1. Prerequisites (on the Host PC)
   * NVIDIA Driver: Installed on the host (check with nvidia-smi).
   * Docker: Installed and running.
   * NVIDIA Container Toolkit: Required to pass the GPU into the container.
```shell
        # Install if missing (Ubuntu/Debian example)
        sudo apt-get install -y nvidia-container-toolkit
        sudo systemctl restart docker
        
        # Clone github repository
        git clone https://github.com/Eigen-Institute/rt-gpu4pyscf.git
```
#### 2. Build the Docker Image
  Navigate to the gpu4pyscf directory and build the base image.
```shell
    cd rt-gpu4pyscf/gpu4pyscf/dockerfiles/ubuntu_devel
    docker build -t gpu4pyscf:base .
```
#### 3. Run the Container and Build gpu4pyscf
  Run the container, mounting the source code from your host into the container. This allows you to edit files on the host while running them in the optimized Docker environment.

```shell
   # Go back to the repository root
   cd ../../../
   
   # Start the container with GPU support
   docker run --gpus all -it -v $(pwd):/workspace -w /workspace gpu4pyscf:base bash
```
#### 4. Compile and Install (Inside the Container)
  Now that you are inside the container's bash shell:

```shell
   # Compile the C++/CUDA extensions specifically for the 1080 Ti (Pascal, CC 6.1)
   cmake -S gpu4pyscf/lib -B build -DCUDA_ARCHITECTURES="61"
   cmake --build build -j 4
   
   # Install the Python package in development mode
   pip3 install -e .
   
   # Verify the GPU is accessible
   python3 -c "import cupy; print('CUDA Device 0:', cupy.cuda.Device(0).attributes)"
```
#### 5. Running RT-TDDFT
  You can now run your calculations within this container:
```shell
   python3 gpu4pyscf/examples/44-rt_tddft.py
```

  *Tip: If you restart the PC or close the container, you can re-enter it using docker start <container_id> and docker exec -it <container_id> bash. Since we used -v $(pwd):/workspace, any changes you make to the code on your host machine will be immediately reflected inside the container.*

----
## Without Docker Files
#### 1. Requirements
   * NVIDIA Driver: Version 450.80.02 or higher (recommended 525+).
   * CUDA Toolkit: Version 11.0 to 12.x (matching your driver).
   * Python: 3.8 or higher.
   * CMake: 3.19 or higher.
#### 2. Installation Steps

##### Step 1: Create a Virtual Environment (Recommended)

```shell
   python3 -m venv gpu4pyscf_env
   source gpu4pyscf_env/bin/activate
   pip install --upgrade pip
```

##### Step 2: Install PySCF and Cupy
  Install the version of cupy that matches your installed CUDA version (e.g., cupy-cuda12x for CUDA 12).
```shell
   pip install pyscf
   pip install cupy-cuda12x  # Replace '12x' with your CUDA version (e.g., 11x)
```

##### Step 3: Clone and Build the Extensions
  Ensure you have the correct CUDA architecture (61 for Pascal).

```shell
   git clone https://github.com/Eigen-Institute/rt-gpu4pyscf.git
   cd rt-gpu4pyscf
   
   # Configure and build C++/CUDA extensions
   cmake -S gpu4pyscf/lib -B build -DCUDA_ARCHITECTURES="61"
   cmake --build build -j 4
```

##### Step 4: Install the Package
```shell
   pip install .
```

#### 3. Verification
  Run a quick test to ensure the GPU is detected and working:
```shell
python -c "import cupy; print('GPU:',cupy.cuda.Device(0).attributes['MultiProcessorCount'], 'MPs detected')"
```

#### 4. Running RT-TDDFT
  You can now run the included examples:

```shell
   python gpu4pyscf/examples/44-rt_tddft.py
```