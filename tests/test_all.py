from countingworkspace import create_workspace
from countingworkspace.examples import NCATEGORIES, NPROCESS, NTRUE, EFFICIENCIES, EXPECTED_BKG_CAT, LUMI
import numpy as np
import pytest
import countingworkspace.utils
import ROOT


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


def test_generate_toys():
    ws = create_workspace(NCATEGORIES, NPROCESS, NTRUE, EFFICIENCIES, EXPECTED_BKG_CAT)
    toys = countingworkspace.utils.generate_toys(ws, 100)
    assert toys.numEntries() == 100


def test_generate_and_fit():
    ws = create_workspace(NCATEGORIES, NPROCESS, NTRUE, EFFICIENCIES, EXPECTED_BKG_CAT)
    countingworkspace.utils.generate_and_fit(ws, 10)


def test_toy_study():
    ws = create_workspace(NCATEGORIES, NPROCESS, NTRUE, EFFICIENCIES, EXPECTED_BKG_CAT)
    countingworkspace.utils.toy_study(ws, 10, seed=42)
    f = ROOT.TFile.Open('result_42.root')
    tree = f.Get("results")
    assert tree
    assert tree.GetEntries() == 10
    branches = [k.GetName() for k in tree.GetListOfBranches()]
    assert 'nll' in branches
    assert 'status' in branches
    for nproc in range(NPROCESS):
        assert ("nsignal_gen_proc%d" % nproc) in branches
        assert ("nsignal_gen_proc%d_error" % nproc) in branches
        assert ("nsignal_gen_proc%d_error_up" % nproc) in branches
        assert ("nsignal_gen_proc%d_error_down" % nproc) in branches
