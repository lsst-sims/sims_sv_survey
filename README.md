# sims_sv_survey
Simulate the SV survey

Although there is a pyproject.toml file here, not all of the 
requirements will install properly (ts-config-ocs, ts-fbs-utils). 
Installing as 
`pip install -e . --no-deps` seems like a good starting point for now. 

For rubin-nights:
`pip install --upgrade git+https://github.com/lsst-sims/rubin_nights.git  --no-deps` 

For ts-fbs-utils: 
`git clone git@github.com:lsst-ts/ts_fbs_utils.git`
Running from the develop branch seems safe, but running from the last tag is likely better.
`pip install -e . --no-deps` 

For ts-config-ocs: 
`git clone git@github.com:lsst-ts/ts_config_ocs.git`
This has to be run from the current run branch, latest commit. 
The run branch changes every few weeks. 
It can be found in JIRA with a query like: 
"Summary ~ "Support Summit Observing Weeks" and status not in (DONE, Invalid) order by duedate ASC"
(and often looks like the ticket branch starting with DM- that has the highest number)
You can either add 
`ts_config_ocs/Scheduler/feature_scheduler/maintel/fbs_config_sv_survey.py`
to your python path (`sys.path.insert(0, ts_config_ocs/Scheduler/feature_scheduler/maintel/fbs_config_sv_survey.py)`) 
or symlink that file to your simulation working directory (which might
be this directory or the `sv_survey` directory). 

