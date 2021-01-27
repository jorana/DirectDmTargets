
# Installing multinest
sudo apt-get install -qq libblas{3,-dev} liblapack{3,-dev} cmake build-essential git gfortran
sudo apt-get install -qq openmpi-bin libopenmpi-dev python-mpi4py
pip install pymultinest
git clone https://github.com/JohannesBuchner/MultiNest
cd MultiNest/build
cmake ..
make
cd ..
ls
pwd
export LD_LIBRARY_PATH=./lib
cd ..
ls
pwd

# Installing others
yes | conda install -c conda-forge emcee
pip install git+https://github.com/jorana/wimprates
pip install git+https://github.com/jorana/verne
