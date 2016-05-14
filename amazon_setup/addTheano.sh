#addTheano.sh adds Theano with CUDA for ubuntu 14.04. Last update in 05/11/2016.
#http://www.ceus-now.com/ubuntu-14-04-gt-750m-install-nvidia-346-72-for-cuda-cudnn-theano/

# Step 1: Pre-install check
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

# Time to Verify: Countdown timer
echo "Verify the above to see if you are qualify. Automatic installation will begin in 30 secs."
secs=$((10 * 3))
while [ $secs -gt 0 ]; do
   echo -ne "$secs\033[0K\r"
   sleep 1
   : $((secs--))
done


# Step 2: Install CUDA
# Get CUDA Repository
sudo apt-get update
sudo apt-get upgrade -y
sudo dpkg -i cuda-repo-ubuntu1404_7.5-18_amd64.deb

# Install CUDA 7.5
sudo apt-get install cuda
sudo apt-get clean

# Environment Setup for 64-bit operating systems
export PATH=/usr/local/cuda-7.5/bin:$PATH #export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:$LD_LIBRARY_PATH #export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH


# Step 3: Verify CUDA
# Verify if CUDA is installed correctly
nvidia-smi # may need "sudo apt-get install nvidia-current" for nVidia driver update


# Step 4: Install Theano
# Install Theano
sudo pip3 install Theano==0.8.2
