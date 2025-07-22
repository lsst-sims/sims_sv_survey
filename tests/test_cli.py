import os
import pickle
import unittest
from contextlib import contextmanager
from tempfile import TemporaryDirectory

from astropy.time import Time
from rubin_scheduler.scheduler.model_observatory.model_observatory import ModelObservatory
from rubin_scheduler.scheduler.schedulers.core_scheduler import CoreScheduler
from rubin_scheduler.scheduler.schedulers.filter_scheduler import BandSwapScheduler
from rubin_scheduler.scheduler.utils import SchemaConverter

from sv_survey import simulate_sv

# export PYTHONPATH=/sdf/data/rubin/user/neilsen/devel/ts_fbs_utils/python:${PYTHONPATH}
# export PYTHONPATH=/sdf/data/rubin/user/neilsen/devel/ts_config_ocs/Scheduler/feature_scheduler/maintel:${PYTHONPATH}


@contextmanager
def temp_cwd() -> None:
    with TemporaryDirectory() as temp_dir:
        old_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            yield
        finally:
            os.chdir(old_cwd)


class TestSVCLI(unittest.TestCase):

#    @unittest.skip("redundant")
    def test_make_sv_scheduler_cli(self) -> None:
        with temp_cwd():
            sceduler_pickle = "scheduler.p"
            return_status = simulate_sv.make_sv_scheduler_cli([sceduler_pickle])
            assert return_status == 0

            with open(sceduler_pickle, "rb") as pickle_io:
                scheduler = pickle.load(pickle_io)

            assert isinstance(scheduler, CoreScheduler)

#    @unittest.skip("redundant")
    def test_make_model_observatory_cli(self) -> None:
        with temp_cwd():
            observatory_pickle = "observatory.p"
            return_status = simulate_sv.make_model_observatory_cli([observatory_pickle])
            assert return_status == 0

            with open(observatory_pickle, "rb") as pickle_io:
                observatory = pickle.load(pickle_io)

            assert isinstance(observatory, ModelObservatory)

    def test_make_band_scheduler_cli(self) -> None:
        with temp_cwd():
            band_scheduler_pickle = "band_scheduler.p"
            return_status = simulate_sv.make_band_scheduler_cli([band_scheduler_pickle])
            assert return_status == 0

            with open(band_scheduler_pickle, "rb") as pickle_io:
                band_scheduler = pickle.load(pickle_io)

            assert isinstance(band_scheduler, BandSwapScheduler)

    def test_run_sv_sim_cli(self) -> None:
        with temp_cwd():
            scheduler_pickle = "scheduler.p"
            return_status = simulate_sv.make_sv_scheduler_cli([scheduler_pickle])
            assert return_status == 0
            with open(scheduler_pickle, "rb") as pickle_io:
                scheduler = pickle.load(pickle_io)
            assert isinstance(scheduler, CoreScheduler)
            nside = scheduler.nside

            observatory_pickle = "observatory.p"
            return_status = simulate_sv.make_model_observatory_cli(
                [observatory_pickle, "--nside", f"{nside}"]
            )
            assert return_status == 0
            with open(observatory_pickle, "rb") as pickle_io:
                observatory = pickle.load(pickle_io)
            assert isinstance(observatory, ModelObservatory)

            init_opsim = ""
            day_obs = "20250720"
            sim_nights = "1"
            run_name = "test_opsim_output"
            simulate_sv.run_sv_sim_cli(
                [scheduler_pickle, observatory_pickle, init_opsim, day_obs, sim_nights, run_name]
            )

            obs = SchemaConverter().opsim2obs(f"{run_name}.db")
            assert len(obs) > 500
            assert Time(obs["mjd"] - 0.5, format="mjd").min().datetime.strftime("%Y%m%d") == day_obs
