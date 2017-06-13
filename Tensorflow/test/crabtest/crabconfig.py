from WMCore.Configuration import Configuration
config = Configuration()

config.section_("General")
config.General.requestName = "testDNNTensorflow"
config.General.workArea = 'crab_testDNNTensorflow'
config.General.transferLogs = True

config.section_("JobType")
config.JobType.pluginName = 'PrivateMC'
config.JobType.psetName = 'pset.py'
config.JobType.disableAutomaticOutputCollection = True
config.JobType.scriptExe = 'jobScript.sh'
config.JobType.outputFiles = ['job.log']
config.JobType.inputFiles = ['jobScript.sh']
config.JobType.maxMemoryMB = 2500
config.JobType.sendPythonFolder= True

config.section_("Data")
config.Data.splitting = 'EventBased'
config.Data.unitsPerJob = 10
config.Data.totalUnits = 30
config.Data.publication = False
config.Data.outputPrimaryDataset = 'testDNNTensorflow'
config.Data.outputDatasetTag = 'crab_testDNNTensorflow'
config.Data.inputDBS = 'phys03'

config.section_("Site")
config.Site.storageSite = 'T2_DE_DESY'
config.Site.whitelist = ['T2_*']

config.section_("User")
## only german users
config.User.voGroup = "dcms"
