#ubuntu 14.04 server: run setup.sh sets up amazon ubuntu 14.04 server for python3 developemnt
#windows: manually via http://www.lfd.uci.edu/~gohlke/pythonlibs/
# ls -ltrh to show all files with permission and size
# df -h to see disk utilization
# File transfer: pscp -i C:\kaggle\randomfsfswqefs.ppk C:\kaggle\dr\makefile.make ubuntu@ec2-52-91-222-20.compute-1.amazonaws.com:makefile
# pscp -i C:\kaggle\randomfsfswqefs.ppk C:\kaggle\dr\makefile.make ubuntu@ec2-52-90-19-209.compute-1.amazonaws.com:makefile
# pscp -i C:\kaggle\randomfsfswqefs.ppk C:\kaggle\dr\cnn2_test.py ubuntu@ec2-52-90-19-209.compute-1.amazonaws.com:cnn2_test.py
# pscp -i C:\kaggle\randomfsfswqefs.ppk C:\kaggle\dr\cnn2.py ubuntu@ec2-52-90-19-209.compute-1.amazonaws.com:cnn2.py
# pscp -i C:\kaggle\randomfsfswqefs.ppk C:\kaggle\dr\theanoChecker.py ubuntu@ec2-52-90-19-209.compute-1.amazonaws.com:theanoChecker.py
# pscp -i C:\kaggle\randomfsfswqefs.ppk C:\kaggle\dr\cookies.txt ubuntu@ec2-52-90-19-209.compute-1.amazonaws.com:cookies.txt

pre:
	sudo apt-get -y update
	sudo apt-get -y upgrade
	sudo apt-get -qq install -y python3-dev python3-setuptools python3-pip
	sudo apt-get -qq install -y gfortran
	sudo apt-get -qq install -y libblas-dev liblapack-dev
	sudo apt-get -qq install -y g++
	sudo apt-get -qq install -y libpng-dev libfreetype6-dev libxft-dev
	sudo apt-get -qq install -y unzip git

python:
	sudo apt-get install python3-numpy python3-scipy
	sudo pip3 install matplotlib==1.5.1
	sudo pip3 install pandas==0.18.1
	sudo pip3 install scikit-learn==0.17.1

	sudo pip3 install h5py==2.6.0
	sudo pip3 install keras==1.0.2

download:
	mkdir input
	cd input
	wget -x --load-cookies ~/cookies.txt https://www.kaggle.com/c/digit-recognizer/download/train.csv
	wget -x --load-cookies ~/cookies.txt https://www.kaggle.com/c/digit-recognizer/download/test.csv
	mv www.kaggle.com/c/digit-recognizer/download/* .
	rm -rf www.kaggle.com
	cd ..

pre-theano:
	# Verify if you have CUDA-Capable GPU, compare form here: https://developer.nvidia.com/cuda-gpus
	lspci | grep -i nvidia
	# Verify You Have a Supported Version of Linux
	uname -m && cat /etc/*release
	# Verify the System Has gcc Installation.
	# If an error message displays, you need to install the development tools from your Linux distribution or obtain a version of gcc
	gcc --version
	# if anything missing when "Verify the System Has gcc Installation", try the folowing line:
	# sudo apt-get update
	# sudo apt-get upgrade -y
	# sudo apt-get install -y opencl-headers build-essential protobuf-compiler libprotoc-dev libboost-all-dev libleveldb-dev hdf5-tools libhdf5-serial-dev libopencv-core-dev  libopencv-highgui-dev libsnappy-dev libsnappy1 libatlas-base-dev cmake libstdc++6-4.8-dbg libgoogle-glog0 libgoogle-glog-dev libgflags-dev liblmdb-dev git gfortran
	# Verify the System has the Correct Kernel Headers and Development Packages Installed
	uname -r
	# sudo apt-get install -y linux-image-extra-`uname -r` linux-headers-`uname -r` linux-image-`uname -r`
	sudo apt-get install linux-headers-$(uname -r)
	sudo apt-get install -y linux-image-extra-$(uname -r)
	sudo apt-get install -y linux-image-$(uname -r)

cuda:
	# Install Dependencies
	sudo apt-get update
	sudo apt-get upgrade -y
	sudo apt-get install -y opencl-headers build-essential protobuf-compiler
	sudo apt-get install -y libprotoc-dev libboost-all-dev libleveldb-dev hdf5-tools libhdf5-serial-dev
	sudo apt-get install -y libopencv-core-dev libopencv-highgui-dev libsnappy-dev libsnappy1
	sudo apt-get install -y libatlas-base-dev cmake libstdc++6-4.8-dbg libgoogle-glog0 libgoogle-glog-dev
	sudo apt-get install -y libgflags-dev liblmdb-dev git python-pip gfortran
	sudo apt-get clean

	# Get CUDA Repository
	sudo dpkg -i cuda-repo-ubuntu1404_7.5-18_amd64.deb

	# Install CUDA 7.5
	sudo apt-get -y install cuda
	sudo apt-get -y clean

	# Environment Setup for 64-bit operating systems
	export PATH=/usr/local/cuda-7.5/bin:$PATH
	export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:$LD_LIBRARY_PATH

	# Verify if CUDA is installed correctly
	nvidia-smi

	# Install Theano
	sudo pip3 install Theano==0.8.2
