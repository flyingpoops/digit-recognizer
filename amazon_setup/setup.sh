#ubuntu 14.04 server: run setup.sh sets up amazon ubuntu 14.04 server for python3 developemnt
#windows: manually via http://www.lfd.uci.edu/~gohlke/pythonlibs/
# File transfer: pscp -i C:\kaggle\randomfsfswqefs.ppk C:\kaggle\dr\setup.sh ubuntu@ec2-52-90-23-192.compute-1.amazonaws.com:setup.sh
#pscp -i C:\kaggle\randomfsfswqefs.ppk C:\kaggle\dr\cookies.txt ubuntu@ec2-52-90-23-192.compute-1.amazonaws.com:cookies.txt
#
#sudo apt-get -y update
#sudo apt-get -y upgrade
hellomake:
	sudo apt-get -qq install -y python3-dev python3-setuptools python3-pip
	sudo apt-get -qq install -y gfortran
	sudo apt-get -qq install -y libblas-dev liblapack-dev
	sudo apt-get -qq install -y g++