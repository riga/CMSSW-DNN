#!/usr/bin/env bash

action() {
	local origin="$( /bin/pwd )"

	if [ -d "$CMSSW_BASE" ]; then

		# setup the tensorflow interface
		cd $CMSSW_BASE/src/DNN/Tensorflow
		if [ -f "/afs/cern.ch/work/m/mharrend/public/tensorflow-cmssw8-0-26.tar.gz" ]; then
			cp /afs/cern.ch/work/m/mharrend/public/tensorflow-cmssw8-0-26.tar.gz .
		else
			wget http://www-ekp.physik.uni-karlsruhe.de/~harrendorf/tensorflow-cmssw8-0-26.tar.gz
		fi
		tar -zxf tensorflow-cmssw8-0-26.tar.gz
		mkdir -p python
		mv tensorflow-cmssw8-0-26-patch1/site-packages/* python/
		rm -rf tensorflow-cmssw8-0-26.tar.gz tensorflow-cmssw8-0-26-patch1/

	else
		echo "please setup CMSSW and do 'cmsenv' before calling this script"
	fi

	cd "$origin"
}
action "$@"
