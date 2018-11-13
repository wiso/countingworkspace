from countingworkspace import create_workspace
import countingworkspace
from countingworkspace.examples import NCATEGORIES, NPROCESS, NTRUE, EFFICIENCIES, EXPECTED_BKG_CAT, LUMI
import numpy as np
import pytest
import countingworkspace.utils
import ROOT


def test_create_variable_scalar():
    ws = ROOT.RooWorkspace()
    countingworkspace.create_variables(ws, 'lumi', values=10.)
    lumi = ws.var('lumi')
    assert lumi
    assert lumi.isConstant()
    assert lumi.getVal() == 10.

    countingworkspace.create_variables(ws, 'lumi2', values=11., ranges=[1., 20.])
    lumi2 = ws.var('lumi2')
    assert lumi2
    assert not lumi2.isConstant()
    assert lumi2.getVal() == 11.
    assert lumi2.getMin() == 1.
    assert lumi2.getMax() == 20.


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


def test_free_variables():
    ws = create_workspace(NCATEGORIES, NPROCESS, NTRUE, EFFICIENCIES, EXPECTED_BKG_CAT)
    free_variables = countingworkspace.utils.get_free_variables(ws)
    assert free_variables.getSize() == NPROCESS


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
