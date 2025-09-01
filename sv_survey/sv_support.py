import sqlite3

import astropy
import numpy as np
import pandas as pd
from astropy.time import Time

astropy.utils.iers.conf.iers_degraded_accuracy = "ignore"

import rubin_nights.dayobs_utils as rn_dayobs
from rubin_nights import connections
from rubin_nights.augment_visits import augment_visits
from rubin_scheduler.scheduler.model_observatory import ModelObservatory, tma_movement
from rubin_scheduler.scheduler.utils import (
    ObservationArray,
    SchemaConverter,
    run_info_table,
)
from rubin_scheduler.site_models import Almanac, ConstantSeeingData, SeeingModel
from rubin_scheduler.utils import DEFAULT_NSIDE, Site

__all__ = [
    "survey_times",
    "lst_over_survey_time",
    "setup_observatory_summit",
    "count_obstime",
    "save_opsim",
]


def survey_times(
    verbose: bool = True,
    random_seed: int = 55,
    early_dome_closure: float = 2.0,
    no_downtime: bool = False,
    real_downtime: bool = False,
    visits: pd.DataFrame | None = None,
    day_obs: int | None = None,
    nside: int = DEFAULT_NSIDE,
    tokenfile: str | None = None,
    site: str = "usdf",
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
        Do not add downtime (True). If False, currently creates about
        a 50% downtime, which is roughly in line with operational
        overhead at present, on nights which open.
        Set to False for whole SV simulations.
        For single night simulations, set to True.
    real_downtime
        Use the time the SV survey was on-sky operational to determine
        the downtime up to "now". This is only useful to check simulations
        running from the start of survey (without using real visits) match
        the appropriate general characteristics of the real survey to date.
        It does also give a more realistic view of up/down time prior to
        the current date.
    visits
        Option to pass in the visits from the consdb, instead of querying
        directly. Only needed if real_downtime is True.
    day_obs
        Use real downtime up until day_obs (visits.day_obs < day_obs) and
        then cut over to simulated downtime.
    nside
        Should be set from the scheduler configuration, but will use
        default value otherwise. Is used for ModelObservatory setup.
    tokenfile
        Path to the RSP tokenfile.
        See also `rubin_nights.connections.get_access_token`.
        Default None will use `ACCESS_TOKEN` environment variable.
    site
        The site (`usdf`, `usdf-dev`, `summit` ..) location at
        which to query services. Must match tokenfile origin.

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

    if verbose:
        print("Survey start", survey_start.iso)
        print("Survey end", survey_end.isot)
        print("for a survey length of (nights)", survey_length)

    lsst_site = Site("LSST")
    almanac = Almanac(mjd_start=survey_start.mjd)
    alm_start = np.where(abs(almanac.sunsets["sunset"] - survey_start.mjd) < 0.5)[0][0]
    alm_end = np.where(abs(almanac.sunsets["sunset"] - survey_end.mjd) < 0.5)[0][0]

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
        "site": lsst_site,
        "nside": nside,
        "early_dome_closure": early_dome_closure,
    }

    # Add time limits and downtime
    # early_dome_closure is the time ahead of 0-deg sunrise to close

    # We always have early dome closure
    down_starts = actual_sunrises - early_dome_closure / 24
    down_ends = actual_sunrises

    # And we might as well throw in being slow on sky
    d_starts = actual_sunsets
    d_ends = sunsets + 0.5 / 24

    down_starts = np.concatenate([down_starts, d_starts])
    down_ends = np.concatenate([down_ends, d_ends])

    # Generate simulated downtime throughout SV
    if not no_downtime:
        # 2 hours per night for SV survey from June 15 - July 1
        # choose best time (moon down)
        survey_transition = Time("2025-07-01T12:00:00", format="isot", scale="utc")
        alm_transition = np.where(abs(almanac.sunsets["sunset"] - survey_transition.mjd) < 0.5)[0][0]
        transition_offset = alm_transition - alm_start
        # whole night from July 1 - September 15 with random downtime
        # 50% downtime ?
        # Added as of 20250730
        # Take complete night for aos
        aos_test_start_total = Time("2025-08-27T12:00:00", scale="utc", format="isot")
        aos_test_end_total = Time("2025-08-28T12:00:00", scale="utc", format="isot")
        # Take some hours per night for aos test
        aos_hours_min = 4
        aos_hours_max = 7
        aos_test_start = Time("2025-08-28T12:00:00", scale="utc", format="isot")
        aos_test_end = Time("2025-09-06T12:00:00", scale="utc", format="isot")

        # More weather
        weather_starts = [
            Time("2025-08-29T12:00:00", scale="utc"),
            Time("2025-09-12T12:00:00", scale="utc"),
        ]
        weather_ends = [
            Time("2025-09-01T12:00:00", scale="utc"),
            Time("2025-09-14T12:00:00", scale="utc"),
        ]

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

        # Choose a random 2 hours for operation at start within nights
        # 20250620 - 20250701
        # This might not be right ... but it's done now.
        obs_start = np.zeros(len(dark_start))
        good = np.where((dark_end - dark_start - 2.5 / 24 > 0) & (night_start < survey_transition.mjd))
        obs_start[good] = rng.uniform(low=dark_start[good], high=dark_end[good] - 2.5 / 24)
        # Add downtime from the start of the night until the observations
        down_starts = np.concatenate([down_starts, actual_sunsets[good]])
        down_ends = np.concatenate([down_ends, obs_start[good]])

        # Add downtime from after the observation until sunrise
        down_starts = np.concatenate([down_starts, obs_start[good] + 2.1 / 24.0])
        down_ends = np.concatenate([down_ends, actual_sunrises[good]])
        # if moon up all night AND before survey_mid, remove the whole night
        bad = np.where((dark_end - dark_start - 2.5 / 24 <= 0) & (night_start < survey_transition.mjd))
        down_starts = np.concatenate([down_starts, actual_sunsets[bad]])
        down_ends = np.concatenate([down_ends, actual_sunrises[bad]])

        # Then let's add random periods of downtime within each night,
        # Assume some chance of having some amount of downtime
        # (but this is simplistic and assumes downtime ~twice per night)
        random_downtime = 0
        match = np.where(night_start > survey_transition.mjd)[0]
        for count in range(5):
            threshold = 1.0 - (count / 5)
            prob_down = rng.random(len(match))
            time_down = rng.gumbel(loc=0.4, scale=1, size=len(match))  # in hours
            # apply probability of having downtime or not -
            # But always at least 2 minutes
            time_down = np.where(prob_down <= threshold, time_down, 2 / 60 / 24)
            avail_in_night = (night_end[match] - night_start[match]) * 24
            time_down = np.where(time_down >= avail_in_night, avail_in_night, time_down)
            time_down = np.where(time_down <= 0, 3 / 60 / 24, time_down)
            d_starts = rng.uniform(low=night_start[match], high=night_end[match] - time_down / 24)
            d_ends = d_starts + time_down / 24.0
            random_downtime += ((d_ends - d_starts) * 24).sum()
            night_hours = avail_in_night.sum()
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

        # Remove whole nights days for AOS test
        # (this would work to just use start/end, but plot later is not good)
        d_starts = np.arange(aos_test_start_total.mjd, aos_test_end_total.mjd - 1, 1)
        d_ends = np.arange(aos_test_start_total.mjd + 1, aos_test_end_total.mjd, 1)
        down_starts = np.concatenate([down_starts, d_starts])
        down_ends = np.concatenate([down_ends, d_ends])
        # And add aos hours per night during AOS test
        match = np.where((night_start >= aos_test_start.mjd) & (night_end <= (aos_test_end.mjd + 1)))[0]
        aos_time_test = rng.uniform(low=aos_hours_min, high=aos_hours_max, size=len(match)) / 24
        aos_time_start = rng.uniform(low=night_start[match] + 0.5 / 24, high=night_end[match] - aos_time_test)
        d_starts = aos_time_start
        d_ends = aos_time_start + aos_time_test
        down_starts = np.concatenate([down_starts, d_starts])
        down_ends = np.concatenate([down_ends, d_ends])

        # Remove some additional whole nights for weather
        for ws, we in zip(weather_starts, weather_ends):
            d_starts = np.arange(ws.mjd, we.mjd - 1, 1)
            d_ends = np.arange(ws.mjd + 1, we.mjd, 1)
            down_starts = np.concatenate([down_starts, d_starts])
            down_ends = np.concatenate([down_ends, d_ends])

        # Sort and then remove overlaps
        sorted_order = np.argsort(down_starts)
        down_starts = down_starts[sorted_order]
        down_ends = down_ends[sorted_order]

        # Remove overlaps
        diff = down_starts[1:] - down_ends[0:-1]
        while np.min(diff) < 0:
            tts = down_starts[0:-1].copy()
            tte = down_ends[0:-1].copy()
            for i, (ds, de) in enumerate(zip(tts, tte)):
                if down_starts[i + 1] < de:
                    new_end = np.max([de, down_ends[i + 1]])
                    down_ends[i] = new_end
                    down_ends[i + 1] = new_end

            good = np.where(down_ends - np.roll(down_ends, 1) != 0)
            down_ends = down_ends[good]
            down_starts = down_starts[good]
            diff = down_starts[1:] - down_ends[0:-1]

    if not no_downtime and real_downtime:
        if day_obs is None:
            day_obs = rn_dayobs.day_obs_str_to_int(rn_dayobs.time_to_day_obs(Time.now()))
        day_obs_mjd = rn_dayobs.day_obs_to_time(day_obs).mjd
        if visits is None:
            # Fetch the visits if not already provided
            endpoints = connections.get_clients(tokenfile, site)
            query = (
                "select *, q.* from cdb_lsstcam.visit1 left join cdb_lsstcam.visit1_quicklook as q "
                "on visit1.visit_id = q.visit_id "
                "where science_program = 'BLOCK-365' and observation_reason != 'block-t548' and "
                f"observation_reason != 'field_survey_science' and visit1.day_obs < {day_obs}"
            )
            visits = endpoints["consdb"].query(query)
            visits = augment_visits(visits, "lsstcam")

        survey_info["consdb_visits"] = visits

        # Identify gaps/downtime starts
        if "obs_start_mjd" in visits.columns:
            obs_start_mjd_key = "obs_start_mjd"
        else:
            obs_start_mjd_key = "observationStartMJD"
        edges = np.where(np.diff(visits[obs_start_mjd_key].values) > 250 / 60 / 60 / 24)[0]
        dropout_starts = visits.iloc[edges]["obs_end_mjd"].values
        dropout_ends = visits.iloc[edges + 1][obs_start_mjd_key].values - 150 / 60 / 60 / 24

        dropout_starts = np.concatenate([dropout_starts, np.array([visits.obs_end_mjd.max()])])
        dropout_ends = np.concatenate([dropout_ends, np.array([day_obs_mjd - 0.001])])

        dayobsmjd = np.arange(survey_start.mjd, survey_start.mjd + survey_length, 1)
        d_starts = []
        d_ends = []
        for ds, de in zip(dropout_starts, dropout_ends):
            idx_s = np.where(ds >= dayobsmjd)[0][-1]
            idx_e = np.where(de >= dayobsmjd)[0][-1]
            if idx_s == idx_e:
                d_starts += [ds]
                d_ends += [de]
            else:
                idx_s = idx_s + 1
                if idx_e == idx_s:
                    d_starts += [ds]
                    d_starts += [dayobsmjd[idx_s]]
                    d_ends += [dayobsmjd[idx_e]]
                    d_ends += [de]
                else:
                    idx_e = idx_e + 1
                    d_starts += [ds]
                    d_starts += list(dayobsmjd[idx_s:idx_e])
                    d_ends += list(dayobsmjd[idx_s:idx_e])
                    d_ends += [de]

        # Use real downtime where we have that information, but continue
        # with sim downtime
        keep_starts = down_starts[np.where(down_starts >= day_obs_mjd)]
        keep_ends = down_ends[np.where(down_ends > day_obs_mjd)]

        down_starts = np.concatenate([d_starts, keep_starts])
        down_ends = np.concatenate([d_ends, keep_ends])

    # Trim all of these to sunrise/sunset
    dayobsmjd = np.arange(survey_start.mjd, survey_start.mjd + survey_length, 1)
    downtime_starts = []
    downtime_ends = []
    eps = 0.0001
    for ds, de in zip(down_starts, down_ends):
        idx = np.where(ds >= dayobsmjd)[0][-1]
        if ds < actual_sunsets[idx]:
            downtime_starts.append(actual_sunsets[idx])
        else:
            downtime_starts.append(ds)
        idx = np.where(de > dayobsmjd)[0][-1]
        # If we ended up on a day boundary, have to back up one night
        if np.abs(de - dayobsmjd[idx]) < eps:
            idx = idx - 1
        if de > actual_sunrises[idx]:
            downtime_ends.append(actual_sunrises[idx])
        else:
            downtime_ends.append(de)

    # Turn into an array of downtimes for sim_runner
    # down_starts/ down_ends should be mjd times for internal-ModelObservatory use
    downtimes = np.array(
        list(zip(downtime_starts, downtime_ends)),
        dtype=list(zip(["start", "end"], [float, float])),
    )
    downtimes.sort(order="start")

    # Eliminate overlaps (just in case)
    diff = downtimes["start"][1:] - downtimes["end"][0:-1]
    while np.min(diff) < 0:
        print("found overlap")
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
        if len(idx[0]) == 0:
            print(start, end, Time(start, format="mjd").iso, Time(end, format="mjd").iso)
            continue
        if len(idx[0]) > 1:
            print(start, end, Time(start, format="mjd").iso, Time(end, format="mjd").iso)
            print(idx, dayobsmjd[idx], sunsets[idx], sunrises[idx])
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
    survey_info["system_availability"] = np.nanmean(
        survey_info["avail_per_night"] / survey_info["hours_in_night"]
    )
    if verbose:
        print(f"Max length of night {hours_in_night.max()} min length of night {hours_in_night.min()}")
        print(
            f"Total nighttime {hours_in_night.sum()}, "
            f"total downtime {downtime_per_night.sum()}, "
            f"available time {hours_in_night.sum() - downtime_per_night.sum()}"
        )
        print(f"Average availability {survey_info['system_availability']}")

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


def setup_observatory_summit(
    survey_info: dict, seeing: float | None = None, clouds: bool = False
) -> ModelObservatory:
    """Configure a `summit-10` model observatory.
    This approximates average summit performance at present.

    Parameters
    ----------
    survey_info
        The survey_info dictionary returned by `survey_times`
    seeing
        If specified (as a float), then the constant seeing model will be
        used, delivering atmospheric seeing with `seeing` arcsecond values.
    clouds
        If True, use our standard cloud downtime model.
        If False, use the 'ideal' cloud model resulting in no downtime.

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
    # Use a bigger system contribution
    seeing_model = SeeingModel(telescope_seeing=0.52)

    if clouds:
        cloud_data = None
    else:
        cloud_data = "ideal"

    observatory = ModelObservatory(
        nside=survey_info["nside"],
        mjd=survey_info["survey_start"].mjd,
        mjd_start=survey_info["survey_start"].mjd,
        cloud_data=cloud_data,
        seeing_data=seeing_data,
        wind_data=None,
        downtimes=survey_info["downtimes"],
        sim_to_o=None,
    )
    observatory.seeing_model = seeing_model
    # "10 percent TMA" - but this is a label from the summit, not 10% in all
    observatory.setup_telescope(
        azimuth_maxspeed=1.0,
        azimuth_accel=1.0,
        azimuth_jerk=4.0,
        altitude_maxspeed=4.0,
        altitude_accel=1.0,
        altitude_jerk=4.0,
        settle_time=3.45,  # more like current settle average
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
    if initial_opsim is not None:
        visits_df = pd.concat([initial_opsim, sim_visits])
    else:
        visits_df = sim_visits
    if filename is not None:
        con = sqlite3.connect(filename)
        visits_df.to_sql("observations", con, index=False, if_exists="replace")
        info = run_info_table(observatory)
        df_info = pd.DataFrame(info)
        df_info.to_sql("info", con, if_exists="replace")
        con.close()
    return visits_df
