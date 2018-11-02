from countingworkspace import create_workspace
from countingworkspace.examples import NCATEGORIES, NPROCESS, NTRUE, EFFICIENCIES, EXPECTED_BKG_CAT, LUMI
import numpy as np
import pytest


def test_create_workspace():
    ws = create_workspace(NCATEGORIES, NPROCESS, NTRUE, EFFICIENCIES, EXPECTED_BKG_CAT)

    assert ws
    for ncat in range(NCATEGORIES):
        for nproc in range(NPROCESS):
            np.testing.assert_allclose(
                ws.var("eff_cat%d_proc%d" % (ncat, nproc)).getVal(),
                EFFICIENCIES[ncat][nproc],
            )


def test_create_workspace_raise():
    with pytest.raises(ValueError):
        create_workspace(
            NCATEGORIES - 1, NPROCESS, NTRUE, EFFICIENCIES, EXPECTED_BKG_CAT
        )
    with pytest.raises(ValueError):
        create_workspace(
            NCATEGORIES, NPROCESS + 1, NTRUE, EFFICIENCIES, EXPECTED_BKG_CAT
        )

def test_create_workspace_luminosity():
    ws = ROOT.RooWorkspace()
    ws.factory('lumi[%s]' % LUMI)
