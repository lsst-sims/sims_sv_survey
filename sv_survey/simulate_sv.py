import argparse
import getpass
import logging
import pickle
import sqlite3
import warnings
from typing import Any

import astropy.units as u
import lsst.ts.fbs.utils.maintel.sv_config as svc
import numpy as np
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
from rubin_scheduler.scheduler.utils import (
    SchemaConverter,
)
from rubin_scheduler.utils import Site
from rubin_sim.sim_archive import make_sim_archive_dir, transfer_archive_dir
from rubin_sim.sim_archive.make_snapshot import get_scheduler_from_config

from . import sv_support as svs

CONFIG_SCRIPT_PATH = "fbs_config_sv_survey.py"

# For backwards compatibility
if getpass.getuser() == "lynnej":
    LYNNES_DIR = "/Users/lynnej/lsst_repos/ts_config_ocs/Scheduler/feature_scheduler/maintel/"
    CONFIG_SCRIPT_PATH = "/".join([LYNNES_DIR, CONFIG_SCRIPT_PATH])

LOGGER = logging.getLogger(__name__)

__all__ = [
    "fetch_previous_sv_visits",
    "setup_sv",
    "run_sv_sim",
    "fetch_sv_visits_cli",
    "make_sv_scheduler_cli",
    "make_model_observatory_cli",
    "make_band_scheduler_cli",
    "run_sv_sim_cli",
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
) -> tuple:
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
    nside, scheduler = get_scheduler_from_config(CONFIG_SCRIPT_PATH)

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


def _simple_run_sv_sim(
    scheduler: CoreScheduler,
    observatory: ModelObservatory,
    survey_info: dict,
    day_obs: int,
    sim_nights: int | None = None,
    anomalous_overhead_func: Any | None = None,
    keep_rewards: bool = False,
    delay: float = 0,
) -> tuple[np.recarray, CoreScheduler, ModelObservatory, pd.DataFrame, pd.DataFrame, dict]:
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
    keep_rewards
        If True, will compute and return rewards.
    delay
        Number of minutes by which simulated observing should be delayed.

    Returns
    -------
    sim_observations, scheduler, observatory, rewards, obs_rewards, survey_info

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

    observer = Observer(Site("LSST").to_earth_location())
    sunset = Time(
        observer.sun_set_time(day_obs_time, which="next", horizon=-6 * u.deg),
        format="jd",
    )
    sunrise = Time(
        observer.sun_rise_time(day_obs_time, which="next", horizon=-6 * u.deg),
        format="jd",
    )

    # If a delay is requested, set the delay relative to 12 degree twilight.
    # This might not always be correct. Ideally, we might need to start with a
    # mini-simulation to test where the first visit comes out without a delay,
    # then follow it with a second sim starting delayed relative to that.
    if delay > 0:
        nominal_start = Time(
            observer.sun_set_time(day_obs_time, which="next", horizon=-12 * u.deg),
            format="jd",
        ).mjd
        sim_start = nominal_start + delay / (24.0 * 60.0)
    else:
        sim_start = sunset.mjd - 15 / 60 / 24

    if sim_nights is not None:
        # end at sunrise after sim_nights
        sim_end = (sunrise + TimeDelta(sim_nights, format="jd")).mjd
    else:
        # end at end of SV
        sim_end = survey_info["survey_end"].mjd

    # Set observatory MJD
    observatory.mjd = sim_start

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

    # Check remaining observations to ensure we scheduled all available time.
    survey_info = svs.count_obstime(sim_observations, survey_info)

    return sim_observations, scheduler, observatory, rewards, obs_rewards, survey_info


def run_sv_sim(
    scheduler: CoreScheduler,
    observatory: ModelObservatory,
    survey_info: dict,
    initial_opsim: pd.DataFrame,
    day_obs: int,
    sim_nights: int | None = None,
    anomalous_overhead_func: Any | None = None,
    run_name: str | None = None,
    keep_rewards: bool = False,
    delay: float = 0,
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
        If None, defaults to `sv_{day_obs}.db`
    keep_rewards
        If True, will compute and return rewards.
    delay
        Number of minutes by which simulated observing should be delayed.

    Returns
    -------
    visits, scheduler, observatory, rewards, obs_rewards, survey_info
    """

    # The actual work of running the simulation is separated into
    # _simple_run_sv_sim so that it can be run without the final step of
    # combining the simulated visits with the initial_opsim visits and saved,
    # while still preserving the previous behavior of calls to run_sv_sim.

    sim_observations, scheduler, observatory, rewards, obs_rewards, survey_info = _simple_run_sv_sim(
        scheduler,
        observatory,
        survey_info,
        day_obs,
        sim_nights,
        anomalous_overhead_func,
        keep_rewards,
        delay,
    )

    if run_name is None:
        run_name = f"sv_{day_obs}"

    # Save (all) visits to disk and join initial + new observations
    visits = svs.save_opsim(observatory, sim_observations, initial_opsim, f"{run_name}.db")

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


def fetch_sv_visits_cli(cli_args: list = []) -> int:
    parser = argparse.ArgumentParser(description="Query the consdb for completed sv visits")
    parser.add_argument("dayobs", type=int, help="Dayobs before which to query.")
    parser.add_argument("file_name", type=str, help="Name of opsim db file to write.")
    parser.add_argument(
        "--site", type=str, default="usdf", help="site of consdb to query (usdf, usdf-dev, or summit)"
    )
    args = parser.parse_args() if len(cli_args) == 0 else parser.parse_args(cli_args)

    dayobs = args.dayobs
    file_name = args.file_name
    site = args.site

    visits = fetch_previous_sv_visits(dayobs, site=site)

    with sqlite3.connect(file_name) as connection:
        visits.to_sql("observations", connection, index=False)

    return 0


def make_sv_scheduler_cli(cli_args: list = []) -> int:
    parser = argparse.ArgumentParser(description="Create a pickle of an SV scheduler")
    parser.add_argument("file_name", type=str, help="Name of pickle file to write.")
    parser.add_argument("--opsim", type=str, default="", help="Name of opsim visits file to load.")
    parser.add_argument(
        "--config-script",
        type=str,
        default=CONFIG_SCRIPT_PATH,
        help="Path to the config script for the scheduler.",
    )
    args = parser.parse_args() if len(cli_args) == 0 else parser.parse_args(cli_args)
    opsim_fname = args.opsim
    scheduler_fname = args.file_name
    scheduler_config_script = args.config_script

    # Set up the scheduler from the config file from ts_config_ocs.
    nside, scheduler = get_scheduler_from_config(scheduler_config_script)
    print("NSIDE: ", nside)

    # Read opsim visits and add to the scheduler
    if len(opsim_fname) > 0:
        obs = SchemaConverter().opsim2obs(opsim_fname)
        scheduler.add_observations_array(obs)

    # Save to a pickle
    with open(scheduler_fname, "wb") as sched_io:
        pickle.dump(scheduler, sched_io)

    return 0


def make_model_observatory_cli(cli_args: list = []) -> int:
    parser = argparse.ArgumentParser(description="Create a pickle of a model observatory")
    parser.add_argument("file_name", type=str, help="Name of pickle file to write.")
    parser.add_argument("--nside", type=int, default=32, help="nside for the model observatory.")
    parser.add_argument(
        "--no-downtime", action="store_true", dest="no_downtime", help="Include scheduled downtime"
    )
    parser.add_argument("--seeing", type=float, default=0, help="Seeing to use")
    args = parser.parse_args() if len(cli_args) == 0 else parser.parse_args(cli_args)
    observatory_fname = args.file_name
    nside = args.nside
    no_downtime = args.no_downtime
    seeing = None if args.seeing == 0 else args.seeing

    # Find the survey information - survey start, downtime simulation ..
    survey_info = svs.survey_times(verbose=True, no_downtime=no_downtime, nside=nside)

    # This isn't strictly necessary for survey_info but adds useful
    # potential footprint information for plotting purposes
    survey_info.update(svc.survey_footprint(survey_start_mjd=survey_info["survey_start"].mjd, nside=nside))

    # Now that we have downtime, set up model observatory.
    observatory = svs.setup_observatory_summit(survey_info, seeing=seeing)

    # Save to a pickle
    with open(observatory_fname, "wb") as observatory_io:
        pickle.dump(observatory, observatory_io)

    return 0


def make_band_scheduler_cli(cli_args: list = []) -> int:
    parser = argparse.ArgumentParser(description="Create a pickle of a band scheduler")
    parser.add_argument("file_name", type=str, help="Name of pickle file to write.")
    parser.add_argument("--illum_limit", type=float, default=40, help="Illumination limit.")
    args = parser.parse_args() if len(cli_args) == 0 else parser.parse_args(cli_args)
    file_name = args.file_name
    illum_limit = args.illum_limit

    band_scheduler = SimpleBandSched(illum_limit=illum_limit)

    with open(file_name, "wb") as bs_io:
        pickle.dump(band_scheduler, bs_io)

    return 0


def run_sv_sim_cli(cli_args: list = []) -> int:
    parser = argparse.ArgumentParser(description="Run an SV simulation.")
    parser.add_argument("scheduler", type=str, help="scheduler pickle file.")
    parser.add_argument("observatory", type=str, help="model observatory pickle file.")
    parser.add_argument("initial_opsim", type=str, help="initial opsim database.")
    parser.add_argument("day_obs", type=int, help="start day obs.")
    parser.add_argument("sim_nights", type=int, help="number of nights to run.")
    parser.add_argument("run_name", type=str, help="Run (also db output) name.")
    parser.add_argument(
        "--no-downtime", action="store_true", dest="no_downtime", help="Include scheduled downtime"
    )
    parser.add_argument("--keep_rewards", action="store_true", help="Compute rewards data.")
    parser.add_argument(
        "--archive", type=str, default="", help="URI of the archive in which to store the results"
    )
    parser.add_argument("--telescope", type=str, default="simonyi", help="The telescope simulated.")
    parser.add_argument(
        "--capture_env",
        action="store_true",
        help="Record the current environment as the simulation environment.",
    )
    parser.add_argument("--label", type=str, default="", help="The tags on the simulation.")
    parser.add_argument("--delay", type=float, default=0.0, help="Minutes after nominal to start.")
    parser.add_argument("--tags", type=str, default=[], nargs="*", help="The tags on the simulation.")
    args = parser.parse_args() if len(cli_args) == 0 else parser.parse_args(cli_args)

    with open(args.scheduler, "rb") as sched_io:
        scheduler = pickle.load(sched_io)

    with open(args.observatory, "rb") as obsv_io:
        observatory = pickle.load(obsv_io)

    initial_opsim = None
    if len(args.initial_opsim) > 0:
        converter = SchemaConverter()
        initial_obs = converter.opsim2obs(args.initial_opsim)
        initial_opsim = converter.obs2opsim(initial_obs)

    day_obs = args.day_obs
    sim_nights = args.sim_nights
    run_name = args.run_name
    no_downtime = args.no_downtime
    nside = observatory.nside
    archive_uri = args.archive
    keep_rewards = args.keep_rewards
    tags = args.tags
    label = args.label
    capture_env = args.capture_env
    telescope = args.telescope
    delay = args.delay

    if keep_rewards:
        scheduler.keep_rewards = keep_rewards

    survey_info = svs.survey_times(verbose=True, no_downtime=no_downtime, nside=nside)

    if len(archive_uri) == 0:
        run_sv_sim(
            scheduler,
            observatory,
            survey_info,
            initial_opsim,
            day_obs,
            sim_nights,
            anomalous_overhead_func=None,
            run_name=run_name,
            keep_rewards=keep_rewards,
            delay=delay,
        )
    else:
        LOGGER.info("Starting simulation")
        observations, scheduler, observatory, rewards, obs_rewards, survey_info = _simple_run_sv_sim(
            scheduler,
            observatory,
            survey_info,
            day_obs,
            sim_nights,
            anomalous_overhead_func=None,
            keep_rewards=keep_rewards,
            delay=delay,
        )
        LOGGER.info("Simualtion complete.")

        data_path = make_sim_archive_dir(
            observations,
            rewards,
            obs_rewards,
            in_files={"scheduler": args.scheduler, "observatory": args.observatory},
            tags=tags,
            label=label,
            capture_env=capture_env,
            opsim_metadata={"telescope": telescope},
        )
        LOGGER.info(f"Created simulation archived directory: {data_path.name}")

        sim_archive_uri = transfer_archive_dir(data_path.name, archive_uri)
        LOGGER.info(f"Transferred {data_path} to {sim_archive_uri}")

    return 0
