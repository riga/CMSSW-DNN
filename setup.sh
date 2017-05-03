#!/usr/bin/env bash

# Setup script.
# Usage:
#   > ./setup.sh

action() {
	local origin="$( /bin/pwd )"

	if [ -d "$CMSSW_BASE" ]; then

		# setup the tensorflow software bundle, only if the CMSSW major version is <= 8
		if [[ "$( echo $CMSSW_VERSION | cut -d_ -f2 )" -le 8 ]]; then
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
		fi

		# setup the custom numpy c-api tool file
		# this is a temporary fix until it was centrally deployed by the cmsdist team
		# see https://github.com/cms-sw/cmsdist/issues/2994
		echo "setup tool for numpy C API"
		cd "$CMSSW_BASE/src"
		scram setup "DNN/Tensorflow/py2-numpy-c-api.xml"
		eval `scramv1 runtime -sh`
		cd "$origin"
		echo "numpy C API tool setup successful"

	else
		echo "please setup CMSSW and do 'cmsenv' before calling this script"
	fi

	cd "$origin"
}
action "$@"
