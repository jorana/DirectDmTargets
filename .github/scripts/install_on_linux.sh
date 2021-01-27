
# Installing multinest
echo "do install of multinest"
sudo apt-get install -qq libblas{3,-dev} liblapack{3,-dev} cmake build-essential git gfortran
sudo apt-get install -qq openmpi-bin libopenmpi-dev python-mpi4py
echo "doing pip install"
echo "cloning dir"
git clone https://github.com/JohannesBuchner/MultiNest
#cd MultiNest/build
#cmake ..
#make
#echo "made MultiNest"
#echo "doing cd .. ; ls ; pwd ; cd .."
#cd ..
#ls
#pwd
#cd ..
#echo "Setting evn var"
#export LD_LIBRARY_PATH=/home/runner/work/DirectDmTargets/DirectDmTargets/MultiNest/lib
mkdir -p MultiNest/build; pushd MultiNest/build; cmake .. && make && popd
test -e MultiNest/lib/libmultinest.so
#python setup.py install --user
#python -c 'import pymultinest'|grep 'LD_LIBRARY_PATH
pip install pymultinest
echo "set LD_LIBRARY_PATH to" $LD_LIBRARY_PATH
echo "in that folder is:"
ls $LD_LIBRARY_PATH
echo "go back to installation; ls ; pwd "
cd DirectDmTargets
ls
pwd
echo "Doing other conda and pip:"
# Installing others
yes | conda install -c conda-forge emcee
pip install git+https://github.com/jorana/wimprates
pip install git+https://github.com/jorana/verne
echo "done"
