import sqlite3

import astropy
import numpy as np
import pandas as pd
from astropy.time import Time

astropy.utils.iers.conf.iers_degraded_accuracy = "ignore"

from rubin_scheduler.scheduler.model_observatory import ModelObservatory, tma_movement
from rubin_scheduler.scheduler.utils import (
    ObservationArray,
    SchemaConverter,
    run_info_table,
)
from rubin_scheduler.site_models import Almanac, ConstantSeeingData
from rubin_scheduler.utils import DEFAULT_NSIDE, Site

__all__ = [
    "survey_times",
    "lst_over_survey_time",
    "setup_observatory",
    "setup_observatory_summit",
    "count_obstime",
    "save_opsim",
]


def survey_times(
    verbose: bool = True,
    random_seed: int = 55,
    early_dome_closure: float = 2.0,
    no_downtime: bool = False,
    nside: int = DEFAULT_NSIDE,
) -> dict:
    """Set up basic SV survey conditions.

    Parameters
    ----------
    verbose
        Print information about start/end times and downtime fraction.
    random_seed
        Random value to seed downtimes with
    early_dome_close
        Close the dome (start downtime) `early_dome_closure` hours before
        0-degree sunrise. A closure 2 hours before sunrise aligns with
        current operational guidelines.
    no_downtime
        Create additional downtime (True) - currently creates about
        a 50% downtime, which is roughly in line with operational
        overhead at present, on nights which open. Set to True for whole
        SV simulations.
        For single night simulations, set to False.
    nside
        Should be set from the scheduler configuration, but will use
        default value otherwise. Is used for ModelObservatory setup.

    Returns
    -------
    survey_info
        Returns a dictionary with keys containing information about the
        survey. Among others, this includes:
        `survey_start` - Time to use as survey_start value for simulations
        (for the survey footprint objects).
        `survey_end` - a very approximate end-time for SV survey, to use
        for whole-SV simulations.
        `downtimes` - the downtimes to feed to the ModelObservatory.
    """
    survey_start = Time("2025-06-20T12:00:00", format="isot", scale="utc")
    survey_end = Time("2025-09-22T12:00:00", format="isot", scale="utc")
    survey_length = int(np.ceil((survey_end - survey_start).jd))

    survey_mid = Time("2025-07-01T12:00:00", format="isot", scale="utc")

    if verbose:
        print("Survey start", survey_start.iso)
        print("Survey end", survey_end.isot)
        print("for a survey length of (nights)", survey_length)

    site = Site("LSST")
    almanac = Almanac(mjd_start=survey_start.mjd)
    alm_start = np.where(abs(almanac.sunsets["sunset"] - survey_start.mjd) < 0.5)[0][0]
    alm_end = np.where(abs(almanac.sunsets["sunset"] - survey_end.mjd) < 0.5)[0][0]
    alm_mid = np.where(abs(almanac.sunsets["sunset"] - survey_mid.mjd) < 0.5)[0][0]
    mid_offset = alm_mid - alm_start

    sunsets = almanac.sunsets[alm_start:alm_end]["sun_n12_setting"]
    actual_sunsets = almanac.sunsets[alm_start:alm_end]["sunset"]
    actual_sunrises = almanac.sunsets[alm_start:alm_end]["sunrise"]
    sunrises = almanac.sunsets[alm_start:alm_end]["sun_n12_rising"]
    moon_set = almanac.sunsets[alm_start:alm_end]["moonset"]
    moon_rise = almanac.sunsets[alm_start:alm_end]["moonrise"]
    moon_illum = almanac.interpolators["moon_phase"](sunsets)

    survey_info = {
        "survey_start": survey_start,
        "survey_end": survey_end,
        "survey_length": survey_length,
        "almanac": almanac,
        "sunsets18": almanac.sunsets[alm_start:alm_end]["sun_n18_setting"],
        "sunrises18": almanac.sunsets[alm_start:alm_end]["sun_n18_rising"],
        "sunsets12": sunsets,
        "sunrises12": sunrises,
        "sunsets": actual_sunsets,
        "sunrises": actual_sunrises,
        "moonsets": moon_set,
        "moonrises": moon_rise,
        "moon_illum": moon_illum,
        "site": site,
        "nside": 32,
        "early_dome_closure": early_dome_closure,
    }

    # Add time limits and downtime
    # early_dome_closure is the time ahead of 0-deg sunrise to close
    # 2 hours per night for SV survey from June 15 - July 1
    # choose best time (moon down)
    # whole night from July 1 - September 15 with random downtime
    # 50% downtime ?

    if not no_downtime:
        rng = np.random.default_rng(seed=random_seed)

        night_start = sunsets
        night_end = actual_sunrises - early_dome_closure / 24.0
        # Almanac always counts moon rise and set in the current night, after sunset
        # dark ends with either sunrise or moonrise
        dark_end = np.where(moon_rise < night_end, moon_rise, night_end)
        # dark starts with either sunset or moonset
        dark_start = np.where(moon_set < night_end, moon_set, night_start)
        # but sometimes the moon can be up the whole night
        up_all_night = np.where(dark_end < dark_start)
        dark_start[up_all_night] = 0
        dark_end[up_all_night] = 0

        two_hours = "random"
        if two_hours == "random":
            # Choose a random 2 hours within these times,
            obs_start = np.zeros(len(dark_start))
            good = np.where((dark_end - dark_start - 2.5 / 24 > 0) & (night_start < survey_mid.mjd))
            obs_start[good] = rng.uniform(low=dark_start[good], high=dark_end[good] - 2.5 / 24)
            # Add downtime from the start of the night until the observations
            down_starts = actual_sunsets[good]
            down_ends = obs_start[good]
            # Add downtime from after the observation until sunrise
            down_starts = np.concatenate([down_starts, obs_start[good] + 2.1 / 24.0])
            down_ends = np.concatenate([down_ends, actual_sunrises[good]])
            # if moon up all night AND before survey_mid, remove the whole night
            bad = np.where((dark_end - dark_start - 2.5 / 24 <= 0) & (night_start < survey_mid.mjd))
            down_starts = np.concatenate([down_starts, actual_sunsets[bad]])
            down_ends = np.concatenate([down_ends, actual_sunrises[bad]])

        # Then let's add random periods of downtime within each night,
        # Assume some chance of having some amount of downtime
        # (but this is simplistic and assumes downtime ~twice per night)
        random_downtime = 0
        for count in range(3):
            threshold = 1.0 - (count / 5)
            prob_down = rng.random(len(night_start))
            time_down = rng.gumbel(loc=0.5, scale=1, size=len(night_start))  # in hours
            # apply probability of having downtime or not
            time_down = np.where(prob_down <= threshold, time_down, 0)
            avail_in_night = (night_end - night_start) * 24
            time_down = np.where(time_down >= avail_in_night, avail_in_night, time_down)
            time_down = np.where(time_down <= 0, 0, time_down)
            d_starts = rng.uniform(low=night_start, high=night_end - time_down / 24)
            d_ends = d_starts + time_down / 24.0
            # But only use these after July 15 -
            d_starts = d_starts[mid_offset:]
            d_ends = d_ends[mid_offset:]
            random_downtime += ((d_ends - d_starts) * 24).sum()
            night_hours = avail_in_night[mid_offset:].sum()
            print(
                "cycle",
                count,
                random_downtime,
                night_hours,
                random_downtime / night_hours,
            )
            # combine previous expected downtime and random downtime
            down_starts = np.concatenate([down_starts, d_starts])
            down_ends = np.concatenate([down_ends, d_ends])

        # And mask the final hour of the night through the survey
        # (already done for 0:mid)
        if early_dome_closure > 0:
            down_starts = np.concatenate(
                [down_starts, actual_sunrises[mid_offset:] - early_dome_closure / 24]
            )
            down_ends = np.concatenate([down_ends, actual_sunrises[mid_offset:]])

    else:
        # Only early dome closure
        down_starts = actual_sunrises - early_dome_closure / 24
        down_ends = actual_sunrises

    # Turn into an array of downtimes for sim_runner
    # down_starts/ down_ends should be mjd times for internal-ModelObservatory use
    downtimes = np.array(
        list(zip(down_starts, down_ends)),
        dtype=list(zip(["start", "end"], [float, float])),
    )
    downtimes.sort(order="start")

    # Eliminate overlaps (just in case)
    diff = downtimes["start"][1:] - downtimes["end"][0:-1]
    while np.min(diff) < 0:
        print("found overlap")
        # Should be able to do this without a loop, but this works
        for i, dt in enumerate(downtimes[0:-1]):
            if downtimes["start"][i + 1] < dt["end"]:
                new_end = np.max([dt["end"], downtimes["end"][i + 1]])
                downtimes[i]["end"] = new_end
                downtimes[i + 1]["end"] = new_end

        good = np.where(downtimes["end"] - np.roll(downtimes["end"], 1) != 0)
        downtimes = downtimes[good]
        diff = downtimes["start"][1:] - downtimes["end"][0:-1]

    # Count up downtime within each night
    dayobsmjd = np.arange(survey_start.mjd, survey_start.mjd + survey_length, 1)
    downtime_per_night = np.zeros(len(sunrises))
    for start, end in zip(downtimes["start"], downtimes["end"]):
        idx = np.where((start > dayobsmjd) & (end < dayobsmjd + 1))
        if start < sunsets[idx]:
            dstart = sunsets[idx]
        else:
            dstart = start
        if end > sunrises[idx]:
            dend = sunrises[idx]
        else:
            dend = end
        downtime_per_night[idx] += (dend - dstart) * 24

    survey_info["downtimes"] = downtimes
    survey_info["dayobsmjd"] = dayobsmjd
    hours_in_night = (sunrises - sunsets) * 24
    survey_info["hours_in_night"] = hours_in_night
    survey_info["downtime_per_night"] = downtime_per_night
    survey_info["avail_per_night"] = hours_in_night - downtime_per_night
    survey_info["efficiency_after_midpoint"] = (
        1
        - survey_info["downtime_per_night"][mid_offset:].sum()
        / survey_info["avail_per_night"][mid_offset:].sum()
    )
    if verbose:
        print(f"Max length of night {hours_in_night.max()} min length of night {hours_in_night.min()}")
        print(
            f"Total nighttime {hours_in_night.sum()}, "
            f"total downtime {downtime_per_night.sum()}, "
            f"available time {hours_in_night.sum() - downtime_per_night.sum()}"
        )
        print(f"Efficiency after midpoint {survey_info['efficiency_after_midpoint']}")

    return survey_info


def lst_over_survey_time(survey_info: dict) -> None:
    # Some informational stuff to help define the footprint
    loc = survey_info["site"].to_earth_location()
    idx = int(survey_info["survey_length"] / 2)
    sunsets = survey_info["sunsets"]
    sunrises = survey_info["sunrises"]
    mid_survey = Time(
        sunsets[idx] + (sunrises[idx] - sunsets[idx]) / 2,
        format="mjd",
        scale="utc",
        location=loc,
    )
    mid_lst = mid_survey.sidereal_time("mean")
    idx = 0
    start_lst = Time(
        sunsets[idx] + (sunrises[idx] - sunsets[idx]) / 2,
        format="mjd",
        scale="utc",
        location=loc,
    ).sidereal_time("mean")
    idx = -1
    end_lst = Time(
        sunsets[idx] + (sunrises[idx] - sunsets[idx]) / 2,
        format="mjd",
        scale="utc",
        location=loc,
    ).sidereal_time("mean")

    print(
        "lst midnight @ start",
        start_lst.deg,
        "lst midnight @ mid",
        mid_lst.deg,
        "lst midnight @ end",
        end_lst.deg,
    )
    idx = int(survey_info["survey_length"] / 2)
    sunset_mid_lst = Time(sunsets[idx], format="mjd", scale="utc", location=loc).sidereal_time("mean")
    sunrise_mid_lst = Time(sunrises[idx], format="mjd", scale="utc", location=loc).sidereal_time("mean")
    print("lst sunset @ mid", sunset_mid_lst.deg, "lst sunrise @ mid", sunrise_mid_lst.deg)


def setup_observatory(survey_info: dict, seeing: float | None = None) -> ModelObservatory:
    """Configure a 40%-tma-movement model observatory.
    Probably not useful for SV simulations, except as a faster comparison.

    Parameters
    ----------
    survey_info
        The survey_info dictionary returned by `survey_times`
    seeing
        If specified (as a float), then the constant seeing model will be
        used, delivering atmospheric seeing with `seeing` arcsecond values.

    Returns
    -------
    model_observatory
    """
    if seeing is not None:
        seeing_data = ConstantSeeingData()
        seeing_data.fwhm_500 = seeing
    else:
        # Else use standard seeing data
        seeing_data = None

    model_obs = ModelObservatory(
        nside=survey_info["nside"],
        mjd=survey_info["survey_start"].mjd,
        mjd_start=survey_info["survey_start"].mjd,
        cloud_data="ideal",  # noclouds
        seeing_data=seeing_data,
        wind_data=None,
        downtimes=survey_info["downtimes"],
        sim_to_o=None,
    )
    # Slow the telescope down with smaller jerk/acceleration and maxvel
    tma = tma_movement(40)
    model_obs.setup_telescope(**tma)
    return model_obs


def setup_observatory_summit(survey_info: dict, seeing: float | None = None) -> ModelObservatory:
    """Configure a `summit-10` model observatory.
    This approximates average summit performance at present.

    Parameters
    ----------
    survey_info
        The survey_info dictionary returned by `survey_times`
    seeing
        If specified (as a float), then the constant seeing model will be
        used, delivering atmospheric seeing with `seeing` arcsecond values.

    Returns
    -------
    model_observatory
    """
    if seeing is not None:
        seeing_data = ConstantSeeingData()
        seeing_data.fwhm_500 = seeing
    else:
        # Else use standard seeing data
        seeing_data = None

    observatory = ModelObservatory(
        nside=survey_info["nside"],
        mjd=survey_info["survey_start"].mjd,
        mjd_start=survey_info["survey_start"].mjd,
        cloud_data="ideal",  # no clouds
        seeing_data=seeing_data,
        wind_data=None,
        downtimes=survey_info["downtimes"],
        sim_to_o=None,
    )
    # "10 percent TMA" - but this is a label from the summit, not 10% in all
    observatory.setup_telescope(
        azimuth_maxspeed=1.0,
        azimuth_accel=1.0,
        azimuth_jerk=4.0,
        altitude_maxspeed=4.0,
        altitude_accel=1.0,
        altitude_jerk=4.0,
        settle_time=3,  # more like current settle average
    )
    observatory.setup_camera(band_changetime=130, readtime=3.07)

    return observatory


def count_obstime(observations: ObservationArray, survey_info: dict) -> dict:
    """Utility function to compare the scheduled observation time
    to the available time per night and check for unscheduled time.

    Parameters
    ----------
    observations
        The observations array returned by the scheduler when running
        a simulation.
    survey_info
        The survey_info dictionary returned by `survey_times`

    Returns
    -------
    survey_info
        Adds the observed time per night to the survey_info object.
    """
    obs_time = np.zeros(len(survey_info["sunrises"]))
    for i in range(len(survey_info["sunrises"])):
        idx = np.where(
            (observations["mjd"] >= survey_info["sunsets"][i])
            & (observations["mjd"] <= survey_info["sunrises"][i])
        )[0]
        obs_time[i] = (observations["visittime"][idx] + observations["slewtime"][idx]).sum() / 60 / 60

    tnight = float(survey_info["hours_in_night"].sum())
    tdown = float(survey_info["downtime_per_night"].sum())
    tavail = float(survey_info["avail_per_night"].sum())
    tobs = float((observations["visittime"].sum() + observations["slewtime"].sum()) / 60 / 60)

    print(f"Total night time (hours): {tnight:.2f}")
    print(f"Total down time (hours): {tdown:.2f}")
    print(f"Total available time (hours): {tavail:.2f}")
    print(f"Total time in observations + slew (hours): {tobs:.2f}")
    print(f"Unscheduled time (hours): {(tavail - tobs):.2f}")

    survey_info["obs_time_per_night"] = obs_time
    return survey_info


def save_opsim(
    observatory: ModelObservatory,
    observations: ObservationArray,
    initial_opsim: pd.DataFrame | None,
    filename: str | None = None,
) -> pd.DataFrame:
    """Combine the initial (opsim formatted) visits with the observation array
    returned from the scheduler simulation.  Optionally saves the result
    to a standard opsim sqlite database.

    Parameters
    ----------
    observatory
        Model Observatory from the simulation.
        Used to save the metadata about the model observatory and site models
        to the sqlite file.
    observations
        The ObservationsArray created by the scheduler in the simulation.
    initial_opsim
        The initial opsim visits fed into the scheduler to start the
        simulation.
        Often (especially for SV) these are likely to be visits converted
        from the ConsDB. Note that extra columns are fine! We can keep
        extra consdb information available if desired (this can be useful
        for double-checking).
    filename
        If provided, this is the filename into which to save the
        (complete) opsim results. Recommend naming this something like
        `sv_{day_obs}.db` where dayobs is the integer dayobs on which the
        new sv simulation was

    Returns
    -------
    visits_df
        A DataFrame of both initial and simulated visits, in opsim format.
    """
    sim_visits = SchemaConverter().obs2opsim(observations)
    visits_df = pd.concat([initial_opsim, sim_visits])
    if filename is not None:
        con = sqlite3.connect(filename)
        visits_df.to_sql("observations", con, index=False, if_exists="replace")
        info = run_info_table(observatory)
        df_info = pd.DataFrame(info)
        df_info.to_sql("info", con, if_exists="replace")
        con.close()
    return visits_df
