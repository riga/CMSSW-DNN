#!/usr/bin/env bash

# Legacy setup script for slc6_amd64_gcc530 in combination with 80X and <=92X.
# Usage:
#   > ./setup.sh

action() {
	local origin="$( /bin/pwd )"

	# check CMSSW installation
	if [ ! -d "$CMSSW_BASE" ]; then
		>&2 echo "please setup CMSSW before calling this script"
		cd "$origin"
		return 1
	fi

	# check SCRAM_ARCH
	if [ "$( echo $SCRAM_ARCH | cut -d_ -f3 )" != "gcc530" ]; then
		>&2 echo "compiler in SCRAM_ARCH '$SCRAM_ARCH' but must be gcc530"
		cd "$origin"
		return 1
	fi

	# check CMSSW_VERSION
	local cv="$( echo $CMSSW_VERSION | cut -d_ -f2-3 )"
	if [ "$cv" != "8_0" ] && [ "$cv" != "9_0" ] && [ "$cv" != "9_1" ] && [ "$cv" != "9_2" ]; then
		>&2 echo "CMSSW_VERSION '$CMSSW_VERSION' not supported"
		cd "$origin"
		return 1
	fi

	# download and unpack the bundle
	echo "setup tensorflow bundle"
	cd "$CMSSW_BASE/src/DNN/Tensorflow"
	if [ -f "/afs/cern.ch/work/m/mharrend/public/tensorflow-cmssw8-0-26.tar.gz" ]; then
		echo "fetch from /afs"
		cp /afs/cern.ch/work/m/mharrend/public/tensorflow-cmssw8-0-26.tar.gz .
	else
		echo "download"
		wget -nv http://www-ekp.physik.uni-karlsruhe.de/~harrendorf/tensorflow-cmssw8-0-26.tar.gz
	fi
	tar -zxf tensorflow-cmssw8-0-26.tar.gz
	mkdir -p python
	mv tensorflow-cmssw8-0-26-patch1/site-packages/* python/
	rm -rf tensorflow-cmssw8-0-26.tar.gz tensorflow-cmssw8-0-26-patch1/
	cd "$origin"
	echo "tensorflow bundle setup successful"

	# setup the custom numpy c-api tool file
	echo "setup tool for numpy C API"
	cd "$CMSSW_BASE/src"
	scram setup "DNN/py2-numpy-c-api.xml"
	eval `scramv1 runtime -sh`
	cd "$origin"
	echo "numpy C API tool setup successful"
}
action "$@"
