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
			cd $CMSSW_BASE/src/DNN/Tensorflow
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
			echo "tensorflow bundle setup successful"
		fi

	else
		echo "please setup CMSSW and do 'cmsenv' before calling this script"
	fi

	cd "$origin"
}
action "$@"
