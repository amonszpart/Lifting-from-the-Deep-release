#!/bin/bash

mkdir saved_sessions
cd saved_sessions

echo 'Downloading models...'
wget http://visual.cs.ucl.ac.uk/pubs/liftingFromTheDeep/res/init_session.tar.gz
wget http://visual.cs.ucl.ac.uk/pubs/liftingFromTheDeep/res/prob_model.tar.gz
wget http://geometry.cs.ucl.ac.uk/projects/2019/imapper/pose_MPI.tar.gz
wget http://geometry.cs.ucl.ac.uk/projects/2019/imapper/person_MPI.tar.gz

echo 'Extracting models...'
tar -xvzf init_session.tar.gz
tar -xvzf prob_model.tar.gz
tar -xvzf pose_MPI.tar.gz
tar -xvzf person_MPI.tar.gz
rm -rf init_session.tar.gz
rm -rf prob_model.tar.gz
rm -rf pose_MPI.tar.gz
rm -rf person_MPI.tar.gz
cd ..

echo 'Installing dependencies...'
pip2 install Cython
pip2 install scikit-image

echo 'Compiling external utilities...'
cd utils/external/
python2 setup_fast_rot.py build
#python2 setup_fast_rot.py build_ext --inplace
cd ../../
ln -sf utils/external/build/lib.linux-x86_64-2.7/upright_fast.so ./

echo 'Done'
