import logging
import sqlite3
import sys
import warnings
from typing import Any

import astropy.units as u
import lsst.ts.fbs.utils.maintel.sv_config as svc
import pandas as pd
import rubin_nights.dayobs_utils as rn_dayobs
import rubin_nights.rubin_sim_addons as rn_sim
from astroplan import Observer
from astropy.time import Time, TimeDelta
from rubin_nights import connections
from rubin_nights.augment_visits import augment_visits
from rubin_scheduler.scheduler import sim_runner
from rubin_scheduler.scheduler.model_observatory import ModelObservatory
from rubin_scheduler.scheduler.schedulers import CoreScheduler, SimpleBandSched
from rubin_scheduler.scheduler.utils import SchemaConverter
from rubin_scheduler.utils import Site

from . import sv_support as svs

sys.path.insert(0, "/Users/lynnej/lsst_repos/ts_config_ocs/Scheduler/feature_scheduler/maintel/")
import fbs_config_sv_survey

logger = logging.getLogger(__name__)

__all__ = [
    "fetch_previous_sv_visits",
    "setup_sv",
    "run_sv_sim",
]


def fetch_previous_sv_visits(day_obs: int, tokenfile: str | None = None, site: str = "usdf") -> pd.DataFrame:
    """Fetch SV-relevant visits from the Consdb and convert to Opsim format.

    Parameters
    ----------
    day_obs
        The day_obs (integer) of the day on which to start the simulation.
        Will fetch all SV visits *up to* this day_obs.
    tokenfile
        Path to the RSP tokenfile.
        See also `rubin_nights.connections.get_access_token`.
        Default None will use `ACCESS_TOKEN` environment variable.
    site
        The site (`usdf`, `usdf-dev`, `summit` ..) location at
        which to query services. Must match tokenfile origin.

    Returns
    -------
    initial_opsim
        pd.DataFrame containing opsim-formatted visit information from
        the consdb for the SV visits up to day_obs.
    """
    # Get the SV survey visits from the ConsDB and format as opsim visits.
    endpoints = connections.get_clients(tokenfile=tokenfile, site=site)
    consdb = endpoints["consdb"]

    instrument = "lsstcam"
    query = (
        f"select v.*, q.* from cdb_{instrument}.visit1 as v "
        f"left join cdb_{instrument}.visit1_quicklook as q "
        f"on v.visit_id = q.visit_id "
        f"where v.day_obs >= 20250620 and v.day_obs < {day_obs} "
        f"and v.science_program = 'BLOCK-365' "
        f"and (v.observation_reason not like 'block-t548' "
        f"and v.observation_reason != 'field_survey_science')"
    )
    consdb_visits = consdb.query(query)
    # augment visits adds many additional columns
    consdb_visits = augment_visits(consdb_visits, instrument)

    # Convert consdb visits to opsim visits
    initial_opsim = rn_sim.consdb_to_opsim(consdb_visits)
    initial_opsim["note"] = initial_opsim["scheduler_note"].copy()
    return initial_opsim


def setup_sv(
    day_obs: int,
    no_downtime: bool = True,
    filename: str | None = None,
    tokenfile: str | None = None,
    site: str = "usdf",
):
    """Set up the SV survey scheduler and model observatory.
     Read previous SV visits into scheduler for startup.

    Parameters
    ----------
    day_obs
        The day_obs (integer) of the day on which to start the simulation.
        Will fetch all SV visits *up to* this day_obs.
    no_downtime
        If True, will not add any random downtime to the model observatory.
        If False, then random downtime (on the order of 50%) will be added.
        For prenight simulations, this should be True.
        For full SV simulations, this should be False.
    filename
        Get previous visits from a file, instead of directly from
        the ConsDB. If set, then consdb will not be queried.
    tokenfile
        Path to the RSP tokenfile.
        See also `rubin_nights.connections.get_access_token`.
        Default None will use `ACCESS_TOKEN` environment variable.
    site
        The site (`usdf`, `usdf-dev`, `summit` ..) location at
        which to query services. Must match tokenfile origin.

    Returns
    -------
    scheduler : `CoreScheduler`
    observatory : `ModelObservatory`
    survey_info : `dict`
    initial_opsim : `pd.DataFrame`
    """

    # Set up the scheduler from the config file from ts_config_ocs.
    nside, scheduler = fbs_config_sv_survey.get_scheduler()

    # Find the survey information - survey start, downtime simulation ..
    survey_info = svs.survey_times(verbose=True, no_downtime=no_downtime, nside=nside)

    # This isn't strictly necessary for survey_info but adds useful
    # potential footprint information for plotting purposes
    survey_info.update(svc.survey_footprint(survey_start_mjd=survey_info["survey_start"].mjd, nside=nside))

    # Now that we have downtime, set up model observatory.
    observatory = svs.setup_observatory_summit(survey_info)

    if filename is None:
        # Fetch the initial opsim visits from the consdb.
        initial_opsim = fetch_previous_sv_visits(day_obs, tokenfile, site)
    else:
        # Read from the datafile `filename`.
        con = sqlite3.connect(filename)
        initial_opsim = pd.read_sql("select * from observations;", con)

    # Convert opsim visits to ObservationArray and feed the scheduler.
    sch_obs = SchemaConverter().opsimdf2obs(initial_opsim)
    scheduler.add_observations_array(sch_obs)

    # We will NOT update the observatory to the state of the last visit.
    # If you wish to resume a simulation in the middle of the night,
    # please look at rubin_scheduler.scheduler.utils.restore_scheduler

    # Return initial_opsim instead of sch_obs, because initial_opsim
    # easier to work with, plus retains extra information from the ConsDB.
    return scheduler, observatory, survey_info, initial_opsim


def run_sv_sim(
    scheduler: CoreScheduler,
    observatory: ModelObservatory,
    survey_info: dict,
    initial_opsim: pd.DataFrame,
    day_obs: int,
    sim_nights: int | None = None,
    anomalous_overhead_func: Any | None = None,
    run_name: str | None = None,
) -> tuple[pd.DataFrame, CoreScheduler, ModelObservatory, pd.DataFrame, pd.DataFrame, dict]:
    """Run the (already set up) scheduler and observatory for
    the appropriate length of time.

    Parameters
    ----------
    scheduler
        The CoreScheduler, ready to run the simulation with previous visits,
        if applicable.
    observatory
        The ModelObservatory, configured to run the simulation.
    survey_info
        Dictionary containing survey_start and (potentially) survey_end
        astropy Time dates. Returned from `sv_support.survey_times`.
    day_obs
        The integer dayobs on which to start the simulation.
    sim_nights
        The number of nights to run the simulation. If None, then run
        to the end of survey specified in survey_info.
    anomalous_overhead_func
        A function or callable object that takes the visit time and slew time
        (in seconds) as argument, and returns and additional offset (also
        in seconds) to be applied as additional overhead between exposures.
        Defaults to None.
    run_name
        String to use as the run name, to save the results to disk.
        If None, defaults to `sv_{day_obs}.db`.

    Returns
    -------
    visits, scheduler, observatory, rewards, obs_rewards, survey_info

    Notes
    -----
    For the standard cronjob prenight simulation suites, I imagine
    this would be replaced by `run_prenights`, but note we need
    to find a way to properly set the filters in use for the night
    (and starting after sunset might not trigger the band scheduler?..
    but for prenights, we probably should be setting this to match summit
    maintenance schedule somehow.).
    """

    # Set up the filter scheduler (we should check this against reality)
    # This simple and scheduler just changes based on lunar phase.
    band_scheduler = SimpleBandSched(illum_limit=40)

    # Start at dayobs sunset minus a tiny bit of time
    # (ensure band scheduler changes if needed and that we start on time)
    day_obs_str = rn_dayobs.day_obs_int_to_str(day_obs)
    day_obs_time = Time(f"{day_obs_str}T12:00:00", format="isot", scale="utc")

    observer = Observer.at_site(Site("LSST").to_earth_location())
    sunset = Time(
        observer.sun_set_time(day_obs_time, which="next", horizon=-6 * u.deg),
        format="jd",
    )
    sunrise = Time(
        observer.sun_rise_time(day_obs_time, which="next", horizon=-6 * u.deg),
        format="jd",
    )

    sim_start = sunset.mjd - 15 / 60 / 24
    if sim_nights is not None:
        # end at sunrise after sim_nights
        sim_end = (sunrise + TimeDelta(sim_nights, format="jd")).mjd
    else:
        # end at end of SV
        sim_end = survey_info["survey_end"].mjd

    # Set observatory MJD
    observatory.mjd = sim_start

    keep_rewards = False
    # The scheduler is noisy.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        vals = sim_runner(
            observatory,
            scheduler,
            band_scheduler=band_scheduler,
            sim_start_mjd=sim_start,
            sim_duration=sim_end - sim_start,
            record_rewards=keep_rewards,
            verbose=True,
            anomalous_overhead_func=anomalous_overhead_func,
        )
    # Separate outputs.
    observatory = vals[0]
    scheduler = vals[1]
    sim_observations = vals[2]
    if len(vals) == 5:
        rewards = vals[3]
        obs_rewards = vals[4]
    else:
        rewards = []
        obs_rewards = []

    if run_name is None:
        run_name = f"sv_{day_obs}"

    # Save (all) visits to disk and join initial + new observations
    visits = svs.save_opsim(observatory, sim_observations, initial_opsim, f"{run_name}.db")

    # Check remaining observations to ensure we scheduled all available time.
    survey_info = svs.count_obstime(sim_observations, survey_info)

    return visits, scheduler, observatory, rewards, obs_rewards, survey_info


def sv_sim(
    day_obs: int,
    tokenfile: str | None = None,
    site: str = "usdf",
) -> tuple[pd.DataFrame, dict]:
    """Run an updated SV simulation to end of SV period, starting at day_obs.

    Parameters
    ----------
    day_obs
        The integer dayobs on which to start the simulation.
    tokenfile
        Path to the RSP tokenfile.
        See also `rubin_nights.connections.get_access_token`.
        Default None will use `ACCESS_TOKEN` environment variable.
    site
        The site (`usdf`, `usdf-dev`, `summit` ..) location at
        which to query services. Must match tokenfile origin.

    Returns
    -------
    visits, survey_info : `pd.DataFrame`, `dict`
    """
    scheduler, observatory, survey_info, initial_opsim = setup_sv(
        day_obs, no_downtime=False, filename=None, tokenfile=tokenfile, site=site
    )

    visits, scheduler, observatory, rewards, obs_rewards, survey_info = run_sv_sim(
        scheduler,
        observatory,
        survey_info,
        initial_opsim,
        day_obs,
        sim_nights=None,
        anomalous_overhead_func=None,
        run_name=None,
    )

    return visits, survey_info
