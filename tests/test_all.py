from countingworkspace import create_workspace
import countingworkspace
from countingworkspace.examples import NCATEGORIES, NPROCESS, NTRUE, EFFICIENCIES, EXPECTED_BKG_CAT, LUMI, XSECFID_X_BR_PRODUCTION_MODES
import numpy as np
import pytest
import countingworkspace.utils
from countingworkspace.utils import iter_collection
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


def test_create_expected_true():
    ws = ROOT.RooWorkspace()
    countingworkspace.create_variables(ws, 'lumi', values=10.)
    assert ws.var('lumi')
    NPROC = 4
    xsvalues = np.arange(1, NPROC + 1)
    countingworkspace.create_variables(ws, 'xsec_{index0}', bins=NPROC, values=xsvalues)
    assert ws.allVars().getSize() == NPROC + 1


def test_create_efficiencies():
    ws = ROOT.RooWorkspace()
    eff = np.arange(6).reshape(2, 3)
    countingworkspace.create_efficiencies(ws, eff, 'myeff_cat{cat}_proc{proc}')

    for y in range(2):
        for x in range(3):
            v = ws.var('myeff_cat{cat}_proc{proc}'.format(cat=y, proc=x))
            assert v
            assert v.getVal() == eff[y][x]

    bins_proc = 'A', 'B', 'C'
    bins_cat = 'X', 'Y'
    countingworkspace.create_efficiencies(ws, eff, bins_proc=bins_proc, bins_cat=bins_cat)
    for icat, cat in enumerate(bins_cat):
        for iproc, proc in enumerate(bins_proc):
            v = ws.var('eff_cat{cat}_proc{proc}'.format(cat=cat, proc=proc))
            assert v
            assert v.getVal() == eff[icat][iproc]


def test_create_formula():
    ws = ROOT.RooWorkspace()
    countingworkspace.create_variables(ws, 'a', values=10.)
    assert ws.var('a').getVal() == 10.
    countingworkspace.create_variables(ws, 'prod:X(a, b[20])')
    assert ws.var('b').getVal() == 20.
    assert ws.obj('X').getVal() == 10. * 20.

    NPROC = 4
    xsvalues = np.arange(1, NPROC + 1)
    countingworkspace.create_variables(ws, 'xsec_{index0}', bins=NPROC, values=xsvalues)
    countingworkspace.create_variables(ws, 'prod:ntrue_{index0}(lumi[100], xsec_{index0})', bins=NPROC)
    assert ws.obj('ntrue_0').getVal() == 100 * 1
    assert ws.obj('ntrue_1').getVal() == 100 * 2
    assert ws.obj('ntrue_2').getVal() == 100 * 3
    assert ws.obj('ntrue_3').getVal() == 100 * 4


def test_dot():
    ws = ROOT.RooWorkspace()
    a = np.arange(10)
    b = np.arange(10) - 1.5
    countingworkspace.create_variables(ws, 'a_{index0}', bins=10, values=a)
    countingworkspace.create_variables(ws, 'b_{index0}', bins=10, values=b)
    countingworkspace.dot(ws, 'a_{index0}', 'b_{index0}', nvar=10)
    for i, c in enumerate(a * b):
        assert ws.obj('a_x_b_%d' % i).getVal() == c

    countingworkspace.dot(ws, 'a_{index0}', 'b_{index0}', 'd_{index0}')
    for i, c in enumerate(a * b):
        assert ws.obj('d_%d' % i).getVal() == c


def test_sum():
    ws = ROOT.RooWorkspace()
    a = np.arange(10)
    b = np.arange(10) - 1.5
    countingworkspace.create_variables(ws, 'a_{index0}', bins=10, values=a)
    countingworkspace.create_variables(ws, 'b_{index0}', bins=10, values=b)
    countingworkspace.sum(ws, 'a_{index0}', 'b_{index0}', nvar=10)
    for i, c in enumerate(a + b):
        assert ws.obj('a_plus_b_%d' % i).getVal() == c

    countingworkspace.sum(ws, 'a_{index0}', 'b_{index0}', 'd_{index0}')
    for i, c in enumerate(a + b):
        assert ws.obj('d_%d' % i).getVal() == c


def test_create_workspace():
    ws = create_workspace(NCATEGORIES, NPROCESS, NTRUE, EFFICIENCIES, EXPECTED_BKG_CAT)

    assert ws
    for ncat in range(NCATEGORIES):
        for nproc in range(NPROCESS):
            np.testing.assert_allclose(
                ws.var("eff_cat%d_proc%d" % (ncat, nproc)).getVal(),
                EFFICIENCIES[ncat][nproc],
            )

def test_asimov_roostats():
    ws = create_workspace(NCATEGORIES, NPROCESS, NTRUE, EFFICIENCIES, EXPECTED_BKG_CAT)
    obs = ws.set('all_obs')
    pdf = ws.obj('model')
    assert(obs)
    assert(pdf)
    data_asimov = ROOT.RooStats.AsymptoticCalculator.GenerateAsimovData(pdf, obs)
    assert(data_asimov)
    assert(data_asimov.numEntries() == 1)

    for ivar, v in enumerate(iter_collection(data_asimov.get(0))):
        if type(v) != ROOT.RooCategory:
            np.testing.assert_allclose(v.getVal(), ws.obj('nexp_cat%s' % ivar).getVal())


def test_asimov():
    ws = create_workspace(NCATEGORIES, NPROCESS, NTRUE, EFFICIENCIES, EXPECTED_BKG_CAT)
    data_asimov = countingworkspace.utils.generate_asimov(ws)
    assert data_asimov
    assert data_asimov.numEntries() == 1
    d = data_asimov.get(0)
    assert d.getSize() == NCATEGORIES
    np.testing.assert_allclose([x.getVal() for x in iter_collection(d)], np.dot(EFFICIENCIES, NTRUE) + EXPECTED_BKG_CAT)


def test_fit_asimov():
    ws = create_workspace(NCATEGORIES, NPROCESS, NTRUE, EFFICIENCIES, EXPECTED_BKG_CAT)
    obs = ws.set('all_obs')
    pdf = ws.obj('model')
    assert obs
    assert pdf
    data_asimov = ROOT.RooStats.AsymptoticCalculator.GenerateAsimovData(pdf, obs)
    pois = ws.obj('ModelConfig').GetParametersOfInterest()
    assert pois

    # not start the fit from the true values
    for poi in iter_collection(pois):
        poi.setVal(poi.getVal() * 1.1)

    fr = pdf.fitTo(data_asimov, ROOT.RooFit.Save())
    assert(fr.status() == 0)
    pois_fitted = fr.floatParsFinal()
    for ntrue, poi_fitted in zip(NTRUE, iter_collection(pois_fitted)):
        np.testing.assert_allclose(ntrue, poi_fitted.getVal(), rtol=0.002)


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

    # workspace where nsignal_gen[p] = xsec[p] * lumi
    ws_with_lumi = ROOT.RooWorkspace()
    ws_with_lumi.factory('lumi[%f]' % LUMI)
    countingworkspace.create_variables(ws_with_lumi, 'xsec_{index0}',
                                       bins=NPROCESS,
                                       values=XSECFID_X_BR_PRODUCTION_MODES,
                                       ranges=[-1000, 10000])

    create_workspace(NCATEGORIES, NPROCESS, None, EFFICIENCIES, EXPECTED_BKG_CAT,
                     expression_nsignal_gen='prod:nsignal_gen_proc{index0}(lumi, xsec_{index0})',
                     ws=ws_with_lumi)

    # workspace where nsignal_gen[p] = mu[p] * xsec[p] * lumi
    ws_with_4mu = ROOT.RooWorkspace()
    ws_with_4mu.factory('lumi[%f]' % LUMI)
    countingworkspace.create_variables(ws_with_4mu, 'xsec_{index0}',
                                       bins=NPROCESS,
                                       values=XSECFID_X_BR_PRODUCTION_MODES)
    create_workspace(NCATEGORIES, NPROCESS, None, EFFICIENCIES, EXPECTED_BKG_CAT,
                     expression_nsignal_gen='prod:nsignal_gen_proc{index0}(mu_{index0}[1, -4, 5], lumi, xsec_{index0})',
                     ws=ws_with_4mu)

    # workspace where nsignal_gen[p] = mu * mu[p] * xsec[p] * lumi
    # where true yield is created externally
    ws_with_4mu_x_mu = ROOT.RooWorkspace()
    ws_with_4mu_x_mu.factory('lumi[%f]' % LUMI)
    countingworkspace.create_variables(ws_with_4mu_x_mu, 'xsec_{index0}',
                                       bins=NPROCESS,
                                       values=XSECFID_X_BR_PRODUCTION_MODES)
    countingworkspace.create_variables(ws_with_4mu_x_mu,
                                       'prod:nsignal_gen_proc{index0}(mu[1, -4, 5], mu_{index0}[1, -4, 5], lumi, xsec_{index0})',
                                       bins=NPROCESS)

    create_workspace(NCATEGORIES, NPROCESS, None, EFFICIENCIES, EXPECTED_BKG_CAT,
                     expression_nsignal_gen='nsignal_gen_proc{index0}',
                     ws=ws_with_4mu_x_mu)

    # same, but with names
    ws_with_4mu_x_mu_names = ROOT.RooWorkspace()
    ws_with_4mu_x_mu_names.factory('lumi[%f]' % LUMI)
    countingworkspace.create_variables(ws_with_4mu_x_mu_names, 'xsec_{index0}',
                                       bins=list(map(str, range(NPROCESS))),
                                       values=XSECFID_X_BR_PRODUCTION_MODES)
    countingworkspace.create_variables(ws_with_4mu_x_mu_names,
                                       'prod:nsignal_gen_proc{index0}(mu[1, -4, 5], mu_{index0}[1, -4, 5], lumi, xsec_{index0})',
                                       bins=list(map(str, range(NPROCESS))))

    create_workspace(list(map(str, range(NCATEGORIES))),
                     list(map(str, range(NPROCESS))), None, EFFICIENCIES, EXPECTED_BKG_CAT,
                     expression_nsignal_gen='nsignal_gen_proc{index0}',
                     ws=ws_with_4mu_x_mu_names)

    # nominal workspace for reference
    ws = create_workspace(NCATEGORIES, NPROCESS, NTRUE, EFFICIENCIES, EXPECTED_BKG_CAT)

    all_vars = ws.allVars()
    for v in iter_collection(all_vars):
        v_lumi = ws_with_lumi.obj(v.GetName())
        assert v_lumi
        np.testing.assert_allclose(v.getVal(), v_lumi.getVal())

        v_4mu = ws_with_4mu.obj(v.GetName())
        assert v_4mu
        np.testing.assert_allclose(v.getVal(), v_4mu.getVal())

        v_4mu_x_mu = ws_with_4mu_x_mu.obj(v.GetName())
        assert v_4mu_x_mu
        np.testing.assert_allclose(v.getVal(), v_4mu_x_mu.getVal())

        v_4mu_x_mu_names = ws_with_4mu_x_mu_names.obj(v.GetName())
        assert v_4mu_x_mu_names
        np.testing.assert_allclose(v.getVal(), v_4mu_x_mu_names.getVal())

    all_f = ws.allFunctions()
    for f in iter_collection(all_f):
        f_lumi = ws_with_lumi.obj(f.GetName())
        assert f_lumi
        np.testing.assert_allclose(f.getVal(), f_lumi.getVal())

        f_4mu = ws_with_4mu.obj(f.GetName())
        assert f_4mu
        np.testing.assert_allclose(f.getVal(), f_4mu.getVal())

        f_4mu_x_mu = ws_with_4mu_x_mu.obj(f.GetName())
        assert f_4mu_x_mu
        np.testing.assert_allclose(f.getVal(), f_4mu_x_mu.getVal())

        f_4mu_x_mu_names = ws_with_4mu_x_mu_names.obj(f.GetName())
        assert f_4mu_x_mu_names
        np.testing.assert_allclose(f.getVal(), f_4mu_x_mu_names.getVal())

    all_pdf = ws.allPdfs()
    for p in iter_collection(all_pdf):
        p_lumi = ws_with_lumi.pdf(p.GetName())
        assert p_lumi
        np.testing.assert_allclose(p.getVal(), p_lumi.getVal())

        p_4mu = ws_with_4mu.pdf(p.GetName())
        assert p_4mu
        np.testing.assert_allclose(p.getVal(), p_4mu.getVal())

        p_4mu_x_mu = ws_with_4mu.pdf(p.GetName())
        assert p_4mu_x_mu
        np.testing.assert_allclose(p.getVal(), p_4mu_x_mu.getVal())

        p_4mu_x_mu_names = ws_with_4mu_x_mu_names.pdf(p.GetName())
        assert p_4mu_x_mu_names
        np.testing.assert_allclose(p.getVal(), p_4mu_x_mu_names.getVal())

    assert countingworkspace.utils.get_free_variables(ws).getSize() == \
           countingworkspace.utils.get_free_variables(ws_with_lumi).getSize() == \
           countingworkspace.utils.get_free_variables(ws_with_4mu).getSize() == \
           countingworkspace.utils.get_free_variables(ws_with_4mu_x_mu).getSize() - 1 == \
           countingworkspace.utils.get_free_variables(ws_with_4mu_x_mu_names).getSize() - 1 == \
           NPROCESS


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
    list(countingworkspace.utils.generate_and_fit(ws, 10))


def test_toy_study():
    ws = create_workspace(NCATEGORIES, NPROCESS, NTRUE, EFFICIENCIES, EXPECTED_BKG_CAT)
    NTOYS = 10
    countingworkspace.utils.toy_study(ws, NTOYS, seed=42)
    f = ROOT.TFile.Open('result_42.root')
    tree = f.Get("results")
    assert tree
    assert tree.GetEntries() == NTOYS
    branches = [k.GetName() for k in tree.GetListOfBranches()]
    assert 'nll' in branches
    assert 'status' in branches
    for nproc in range(NPROCESS):
        assert ("nsignal_gen_proc%d" % nproc) in branches
        assert ("nsignal_gen_proc%d_error" % nproc) in branches
        assert ("nsignal_gen_proc%d_error_up" % nproc) in branches
        assert ("nsignal_gen_proc%d_error_down" % nproc) in branches
