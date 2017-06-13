echo "===== Starting test_kitmodel ====" >> job.log
test_kitmodel >> job.log
cmsRun -j FrameworkJobReport.xml -p PSet.py
