from countingworkspace import create_workspace, format_index
import countingworkspace
from countingworkspace.examples import NCATEGORIES, NPROCESS, NTRUE, EFFICIENCIES, EXPECTED_BKG_CAT, LUMI, XSECFID_X_BR_PRODUCTION_MODES
import numpy as np
import pytest
import countingworkspace.utils
from countingworkspace.utils import iter_collection
import ROOT


def test_string_range():
    assert countingworkspace.string_range(2) == ['000', '001']


def test_create_scalar():
    ws = ROOT.RooWorkspace()
    countingworkspace.create_scalar(ws, 'lumi', value=10.)
    lumi = ws.var('lumi')
    assert lumi
    assert lumi.isConstant()
    assert lumi.getVal() == 10.

    countingworkspace.create_scalar(ws, 'lumi2', value=11., ranges=[1., 20.])
    lumi2 = ws.var('lumi2')
    assert lumi2
    assert not lumi2.isConstant()
    assert lumi2.getVal() == 11.
    assert lumi2.getMin() == 1.
    assert lumi2.getMax() == 20.

    countingworkspace.create_scalar(ws, 'expr:double("2*@0", {value})', value=4.)
    d = ws.function('double')
    assert d
    assert d.getVal() == 8.


def test_create_variable_scalar():
    ws = ROOT.RooWorkspace()
    r = countingworkspace.create_variables(ws, 'lumi', values=10.)
    lumi = ws.var('lumi')
    assert lumi
    assert lumi.isConstant()
    assert lumi.getVal() == 10.
    assert r.getVal() == 10.

    countingworkspace.create_variables(ws, 'lumi2', values=11., ranges=[1., 20.])
    lumi2 = ws.var('lumi2')
    assert lumi2
    assert not lumi2.isConstant()
    assert lumi2.getVal() == 11.
    assert lumi2.getMin() == 1.
    assert lumi2.getMax() == 20.

    countingworkspace.create_variables(ws, 'theta', values=0, ranges=(-5, 5))
    theta = ws.var('theta')
    assert theta
    assert not theta.isConstant()
    assert theta.getVal(0) == 0.
    assert theta.getMin() == -5.
    assert theta.getMax() == 5.


def test_create_variable_vector():
    ws = ROOT.RooWorkspace()
    values = [1., 3., 10.]

    r = countingworkspace.create_variables(ws, 'foo_{myindex}', values=values, index_names='myindex')
    v = ws.allVars().selectByName('foo_*')
    assert(v.getSize() == len(values))
    assert(len(r) == len(values))
    for vv1, vv2, rr in zip(iter_collection(v), values, r):
        assert (vv1.getVal() == vv2)
        assert (rr.getVal() == vv2)

    countingworkspace.create_variables(ws, 'bar_{myindex2}', values=values)
    v = ws.allVars().selectByName('bar_*')
    assert(v.getSize() == len(values))
    for vv1, vv2 in zip(iter_collection(v), values):
        assert(vv1.getVal() == vv2)

    countingworkspace.create_variables(ws, 'zoo_{myindex2}', values=values, bins=['one', 'two', 'three'])
    v = ws.allVars().selectByName('bar_*')
    assert(v.getSize() == len(values))
    for vv1, vv2 in zip(iter_collection(v), values):
        assert(vv1.getVal() == vv2)
    assert(ws.var('zoo_one').getVal() == values[0])
    assert(ws.var('zoo_two').getVal() == values[1])
    assert(ws.var('zoo_three').getVal() == values[2])

    bins = list(map(str, range(len(values))))
    countingworkspace.create_variables(ws, 'a_{proc}', bins=bins, values=values, ranges=(-10000, 50000))
    for b, v in zip(bins, values):
        a = ws.var('a_%s' % b)
        assert a
        np.testing.assert_allclose(a.getVal(), v)
        np.testing.assert_allclose(a.getMin(), -10000)
        np.testing.assert_allclose(a.getMax(), 50000)

    countingworkspace.create_variables(ws, 'x_{proc}', bins=[bins], nbins=(len(bins), ), values=values)
    for b, v in zip(bins, values):
        x = ws.var('x_%s' % b)
        assert x
        np.testing.assert_allclose(x.getVal(), v)


def test_create_variable_matrix():
    ws = ROOT.RooWorkspace()
    eff = np.arange(6).reshape(2, 3)
    r = countingworkspace.create_variables(ws, 'myeff_cat{cat}_proc{proc}', values=eff, index_names=('cat', 'proc'))
    assert r

    for r1, v1 in zip(r, eff):
        for r2, v2 in zip(r1, v1):
            assert r2.getVal() == v2

    for y in range(2):
        for x in range(3):
            v = ws.var('myeff_cat{cat}_proc{proc}'.format(cat=format_index(y), proc=format_index(x)))
            assert v
            assert v.getVal() == eff[y][x]

    bins_proc = 'A', 'B', 'C'
    bins_cat = 'X', 'Y'
    countingworkspace.create_variables(ws, 'myeff2_cat{cat}_proc{proc}', values=eff, bins=[bins_cat, bins_proc], index_names=('cat', 'proc'))
    for icat, cat in enumerate(bins_cat):
        for iproc, proc in enumerate(bins_proc):
            v = ws.var('myeff2_cat{cat}_proc{proc}'.format(cat=cat, proc=proc))
            assert v
            assert v.getVal() == eff[icat][iproc]


def test_create_expected_true():
    ws = ROOT.RooWorkspace()
    countingworkspace.create_variables(ws, 'lumi', values=10.)
    assert ws.var('lumi')
    NPROC = 4
    xsvalues = np.arange(1, NPROC + 1)
    countingworkspace.create_variables(ws, 'xsec_{proc}', nbins=NPROC, values=xsvalues)
    assert ws.allVars().getSize() == NPROC + 1


def test_create_efficiencies():
    ws = ROOT.RooWorkspace()
    eff = np.arange(6).reshape(2, 3)
    countingworkspace.create_efficiencies(ws, eff, 'myeff_cat{cat}_proc{proc}')

    for y in range(2):
        for x in range(3):
            v = ws.var('myeff_cat{cat}_proc{proc}'.format(cat=format_index(y), proc=format_index(x)))
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
    countingworkspace.create_variables(ws, 'theta', values=0., ranges=(-5, 5))
    assert ws.var('theta').getVal() == 0.
    assert ws.var('theta').getMin() == -5.
    assert ws.var('theta').getMax() == 5.
    countingworkspace.create_variables(ws, 'prod:X(a, b[20])')
    assert ws.var('b').getVal() == 20.
    assert ws.obj('X').getVal() == 10. * 20.

    NPROC = 4
    xsvalues = np.arange(1, NPROC + 1)
    countingworkspace.create_variables(ws, 'xsec_{proc}', nbins=NPROC, values=xsvalues)
    countingworkspace.create_variables(ws, 'prod:ntrue_{proc}(lumi[100], xsec_{proc})', nbins=NPROC)
    for i, xs in enumerate(xsvalues):
        assert(ws.obj('ntrue_%s' % format_index(i)) == 100 * (i + 1))


def test_dot():
    ws = ROOT.RooWorkspace()
    a = np.arange(10)
    b = np.arange(10) - 1.5
    countingworkspace.create_variables(ws, 'a_{index0}', nbins=10, values=a)
    countingworkspace.create_variables(ws, 'b_{index0}', nbins=10, values=b)
    countingworkspace.dot(ws, 'a_{index0}', 'b_{index0}', nvar=10)
    for i, c in enumerate(a * b):
        assert ws.obj('a_x_b_%s' % format_index(i)).getVal() == c

    countingworkspace.dot(ws, 'a_{index0}', 'b_{index0}', 'd_{index0}')
    for i, c in enumerate(a * b):
        assert ws.obj('d_%s' % format_index(i)).getVal() == c


def test_add():
    ws = ROOT.RooWorkspace()
    a = np.arange(10)
    b = np.arange(10) - 1.5
    countingworkspace.create_variables(ws, 'a_{index0}', nbins=10, values=a)
    countingworkspace.create_variables(ws, 'b_{index0}', nbins=10, values=b)
    countingworkspace.add(ws, 'a_{index0}', 'b_{index0}', nvar=10)
    for i, c in enumerate(a + b):
        assert ws.obj('a_plus_b_%s' % format_index(i)).getVal() == c

    countingworkspace.add(ws, 'a_{index0}', 'b_{index0}', 'd_{index0}')
    for i, c in enumerate(a + b):
        assert ws.obj('d_%s' % format_index(i)).getVal() == c


def test_create_workspace():
    ws = create_workspace(NCATEGORIES, NPROCESS, NTRUE, EFFICIENCIES, EXPECTED_BKG_CAT)

    assert ws
    for cat in range(NCATEGORIES):
        for nproc in range(NPROCESS):
            np.testing.assert_allclose(
                ws.var("eff_cat%s_proc%s" % (format_index(cat), format_index(nproc))).getVal(),
                EFFICIENCIES[cat][nproc],
            )

    all_nexp_cat = np.dot(EFFICIENCIES, NTRUE) + EXPECTED_BKG_CAT

    for cat, nexp_cat in zip(range(NCATEGORIES), all_nexp_cat):
        v = ws.obj('nexp_cat{cat}'.format(cat=format_index(cat)))
        assert(v)
        v1 = v.getVal()
        np.testing.assert_allclose(v1, nexp_cat)

    model_config = ws.obj('ModelConfig')
    obs = model_config.GetObservables()
    assert obs
    assert obs.getSize() == NCATEGORIES
    poi = model_config.GetParametersOfInterest()
    assert poi
    assert poi.getSize() == NPROCESS


def test_create_workspace_systematics_nsignal_gen():
    systematics_nsignal_gen = np.ones(NPROCESS) * 0.01
    systematics_nsignal_gen[0] *= 2

    ws = create_workspace(NCATEGORIES, NPROCESS, NTRUE, EFFICIENCIES, EXPECTED_BKG_CAT,
                          systematics_nsignal_gen=[{'name': 'lumi', 'values': systematics_nsignal_gen}])

    assert ws
    for cat in range(NCATEGORIES):
        for nproc in range(NPROCESS):
            np.testing.assert_allclose(
                ws.var("eff_cat%s_proc%s" % (format_index(cat), format_index(nproc))).getVal(),
                EFFICIENCIES[cat][nproc],
            )

    all_nexp_cat = np.dot(EFFICIENCIES, NTRUE) + EXPECTED_BKG_CAT

    for cat, nexp_cat in zip(range(NCATEGORIES), all_nexp_cat):
        v = ws.obj('nexp_cat{cat}'.format(cat=format_index(cat)))
        assert(v)
        v1 = v.getVal()
        np.testing.assert_allclose(v1, nexp_cat)

    ws.var('theta_lumi').setVal(1)
    all_nexp_cat = np.dot(EFFICIENCIES, NTRUE * (1. + systematics_nsignal_gen)) + EXPECTED_BKG_CAT

    for cat, nexp_cat in zip(range(NCATEGORIES), all_nexp_cat):
        v = ws.obj('nexp_cat{cat}'.format(cat=format_index(cat)))
        assert(v)
        v1 = v.getVal()
        np.testing.assert_allclose(v1, nexp_cat)

    ws.var('theta_lumi').setVal(2)
    all_nexp_cat = np.dot(EFFICIENCIES, NTRUE * (1. + 2 * systematics_nsignal_gen)) + EXPECTED_BKG_CAT

    for cat, nexp_cat in zip(range(NCATEGORIES), all_nexp_cat):
        v = ws.obj('nexp_cat{cat}'.format(cat=format_index(cat)))
        assert(v)
        v1 = v.getVal()
        np.testing.assert_allclose(v1, nexp_cat)


def test_create_workspace_systematics_efficiencies():
    systematics_efficiencies = np.ones_like(EFFICIENCIES) * 0.01
    systematics_efficiencies[0] *= 2

    ws = create_workspace(NCATEGORIES, NPROCESS, NTRUE, EFFICIENCIES, EXPECTED_BKG_CAT,
                          systematic_efficiencies=[{'name': 'lumi', 'values': systematics_efficiencies}])

    assert ws
    for cat in range(NCATEGORIES):
        for nproc in range(NPROCESS):
            np.testing.assert_allclose(
                ws.var("eff_cat%s_proc%s" % (format_index(cat), format_index(nproc))).getVal(),
                EFFICIENCIES[cat][nproc],
            )

    all_nexp_cat = np.dot(EFFICIENCIES, NTRUE) + EXPECTED_BKG_CAT

    for cat, nexp_cat in zip(range(NCATEGORIES), all_nexp_cat):
        v = ws.obj('nexp_cat{cat}'.format(cat=format_index(cat)))
        assert(v)
        v1 = v.getVal()
        np.testing.assert_allclose(v1, nexp_cat)

    ws.var('theta_lumi').setVal(2)
    all_nexp_cat = np.dot(EFFICIENCIES * (1 + 2 * systematics_efficiencies), NTRUE) + EXPECTED_BKG_CAT
    for cat, nexp_cat in zip(range(NCATEGORIES), all_nexp_cat):
        v = ws.obj('nexp_cat{cat}'.format(cat=format_index(cat)))
        assert(v)
        v1 = v.getVal()
        np.testing.assert_allclose(v1, nexp_cat)

    constrain = ws.pdf('constrain_syslumi')
    assert constrain


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
        if not isinstance(v, ROOT.RooCategory):
            np.testing.assert_allclose(v.getVal(), ws.obj('nexp_cat%s' % format_index(ivar)).getVal())


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
    data_asimov = countingworkspace.utils.generate_asimov(ws)
    pois = ws.obj('ModelConfig').GetParametersOfInterest()
    assert pois

    # not start the fit from the true values
    for poi in iter_collection(pois):
        poi.setVal(poi.getVal() * 1.1)

    ws.Print()

    fr = pdf.fitTo(data_asimov, ROOT.RooFit.Save())
    assert(fr.status() == 0)
    pois_fitted = fr.floatParsFinal()
    for ntrue, poi_fitted in zip(NTRUE, iter_collection(pois_fitted)):
        np.testing.assert_allclose(ntrue, poi_fitted.getVal(), rtol=0.002)


def test_fit_asimov_syst():
    systematics_nsignal_gen = np.ones(NPROCESS) * 0.05
    systematics_nsignal_gen[0] *= 2
    systematics_nsignal_gen2 = np.ones(NPROCESS) * 0.06
    systematics_nsignal_gen2[1] *= 2

    ws = create_workspace(NCATEGORIES, NPROCESS, NTRUE, EFFICIENCIES, EXPECTED_BKG_CAT,
                          systematics_nsignal_gen=[{'name': 'lumi', 'values': systematics_nsignal_gen},
                                                   {'name': 'lumi2', 'values': systematics_nsignal_gen2}])

    obs = ws.set('all_obs')
    pdf = ws.obj('model')
    assert obs
    assert pdf
    data_asimov = countingworkspace.utils.generate_asimov(ws)
    pois = ws.obj('ModelConfig').GetParametersOfInterest()
    assert pois

    # not start the fit from the true values
    for poi in iter_collection(pois):
        poi.setVal(poi.getVal() * 1.1)

    fr = pdf.fitTo(data_asimov, ROOT.RooFit.Save(), ROOT.RooFit.Hesse(True))
    assert(fr.status() == 0)
    pois_fitted = fr.floatParsFinal()
    all_errors = []
    for ntrue, poi_fitted in zip(NTRUE, iter_collection(pois_fitted)):
        np.testing.assert_allclose(ntrue, poi_fitted.getVal(), rtol=0.002)
        all_errors.append(poi_fitted.getError())

    ws.loadSnapshot('initial')
    for theta in iter_collection(ws.allVars().selectByName('theta*')):
        theta.setVal(0)
        theta.setConstant()

    fr = pdf.fitTo(data_asimov, ROOT.RooFit.Save(), ROOT.RooFit.Hesse(True))
    assert(fr.status() == 0)
    pois_fitted = fr.floatParsFinal()
    all_errors_stat = []
    for ntrue, poi_fitted in zip(NTRUE, iter_collection(pois_fitted)):
        np.testing.assert_allclose(ntrue, poi_fitted.getVal(), rtol=0.002)
        all_errors_stat.append(poi_fitted.getError())

    sys_only_errors = (np.sqrt(np.array(all_errors)**2 - np.array(all_errors_stat)**2) / NTRUE)
    np.testing.assert_allclose(sys_only_errors, np.sqrt(systematics_nsignal_gen**2 + systematics_nsignal_gen2**2), rtol=5E-2)


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
    countingworkspace.create_variables(ws_with_lumi, 'xsec_{proc}',
                                       nbins=NPROCESS,
                                       values=XSECFID_X_BR_PRODUCTION_MODES,
                                       ranges=[-1000, 10000])

    create_workspace(NCATEGORIES, NPROCESS, None, EFFICIENCIES, EXPECTED_BKG_CAT,
                     expression_nsignal_gen='prod:nsignal_gen_proc{proc}(lumi, xsec_{proc})',
                     ws=ws_with_lumi)

    # workspace where nsignal_gen[p] = mu[p] * xsec[p] * lumi
    ws_with_4mu = ROOT.RooWorkspace()
    ws_with_4mu.factory('lumi[%f]' % LUMI)
    countingworkspace.create_variables(ws_with_4mu, 'xsec_{proc}',
                                       nbins=NPROCESS,
                                       values=XSECFID_X_BR_PRODUCTION_MODES)
    create_workspace(NCATEGORIES, NPROCESS, None, EFFICIENCIES, EXPECTED_BKG_CAT,
                     expression_nsignal_gen='prod:nsignal_gen_proc{proc}(mu_{proc}[1, -4, 5], lumi, xsec_{proc})',
                     ws=ws_with_4mu)

    # workspace where nsignal_gen[p] = mu * mu[p] * xsec[p] * lumi
    # where true yield is created externally
    ws_with_4mu_x_mu = ROOT.RooWorkspace()
    ws_with_4mu_x_mu.factory('lumi[%f]' % LUMI)
    countingworkspace.create_variables(ws_with_4mu_x_mu, 'xsec_{proc}',
                                       nbins=NPROCESS,
                                       values=XSECFID_X_BR_PRODUCTION_MODES)
    countingworkspace.create_variables(ws_with_4mu_x_mu,
                                       'prod:nsignal_gen_proc{proc}(mu[1, -4, 5], mu_{proc}[1, -4, 5], lumi, xsec_{proc})',
                                       nbins=NPROCESS)

    create_workspace(NCATEGORIES, NPROCESS, None, EFFICIENCIES, EXPECTED_BKG_CAT,
                     expression_nsignal_gen='nsignal_gen_proc{proc}',
                     ws=ws_with_4mu_x_mu)

    # same, but with names
    ws_with_4mu_x_mu_names = ROOT.RooWorkspace()
    ws_with_4mu_x_mu_names.factory('lumi[%f]' % LUMI)
    countingworkspace.create_variables(ws_with_4mu_x_mu_names, 'xsec_{proc}',
                                       bins=list(map(format_index, range(NPROCESS))),
                                       values=XSECFID_X_BR_PRODUCTION_MODES)
    countingworkspace.create_variables(ws_with_4mu_x_mu_names,
                                       'prod:nsignal_gen_proc{proc}(mu[1, -4, 5], mu_{proc}[1, -4, 5], lumi, xsec_{proc})',
                                       bins=list(map(format_index, range(NPROCESS))))

    create_workspace(list(map(format_index, range(NCATEGORIES))),
                     list(map(format_index, range(NPROCESS))), None, EFFICIENCIES, EXPECTED_BKG_CAT,
                     expression_nsignal_gen='nsignal_gen_proc{proc}',
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
    assert(len(list(countingworkspace.utils.generate_and_fit(ws, ntoys=10))) == 10)


def test_generate_and_fit_crossed():
    ws_generate = create_workspace(NCATEGORIES, NPROCESS, NTRUE, EFFICIENCIES, EXPECTED_BKG_CAT)
    ws_fit = create_workspace(NCATEGORIES, NPROCESS, NTRUE, EFFICIENCIES / 2., EXPECTED_BKG_CAT)
    assert(len(list(countingworkspace.utils.generate_and_fit(ws_generate, ws_fit, ntoys=10))) == 10)


def test_toy_study():
    ws = create_workspace(NCATEGORIES, NPROCESS, NTRUE, EFFICIENCIES, EXPECTED_BKG_CAT)
    ntoys = 10
    countingworkspace.utils.toy_study(ws, ntoys=ntoys, seed=42,
                                      plugins=[countingworkspace.utils.ToyStudyError(save_asym=True)])
    f = ROOT.TFile.Open('result_42.root')
    tree = f.Get("results")
    assert tree
    assert tree.GetEntries() == ntoys
    branches = [k.GetName() for k in tree.GetListOfBranches()]
    assert 'nll' in branches
    assert 'status' in branches
    for nproc in range(NPROCESS):
        assert ("nsignal_gen_proc%s" % format_index(nproc)) in branches
        assert ("nsignal_gen_proc%s_error" % format_index(nproc)) in branches
        assert ("nsignal_gen_proc%s_error_up" % format_index(nproc)) in branches
        assert ("nsignal_gen_proc%s_error_down" % format_index(nproc)) in branches


def test_toy_coverage():
    ws = create_workspace(NCATEGORIES, NPROCESS, NTRUE, EFFICIENCIES, EXPECTED_BKG_CAT)
    ntoys = 10
    countingworkspace.utils.toy_study(ws, ntoys=ntoys, seed=42,
                                      plugins=[countingworkspace.utils.ToyStudyError(save_asym=True),
                                               countingworkspace.utils.ToyStudyCoverage(ws.obj('ModelConfig').GetParametersOfInterest(),
                                                                                        NTRUE, significance=1, output_var='isCoveredAll'),
                                               countingworkspace.utils.ToyStudyCoverage(ws.obj('ModelConfig').GetParametersOfInterest(),
                                                                                        NTRUE, significance=2, output_var='isCoveredAll2sigma')

                                               ])
    f = ROOT.TFile.Open('result_42.root')
    tree = f.Get("results")
    assert tree
    assert tree.GetEntries() == ntoys
    branches = [k.GetName() for k in tree.GetListOfBranches()]
    assert 'nll' in branches
    assert 'status' in branches
    assert 'isCoveredAll' in branches
    for nproc in range(NPROCESS):
        assert ("nsignal_gen_proc%s" % format_index(nproc)) in branches
        assert ("nsignal_gen_proc%s_error" % format_index(nproc)) in branches
        assert ("nsignal_gen_proc%s_error_up" % format_index(nproc)) in branches
        assert ("nsignal_gen_proc%s_error_down" % format_index(nproc)) in branches

    h1sigma = ROOT.TH1F("h1sigma", "h1sigma", 2, 0, 2)
    h2sigma = ROOT.TH1F("h2sigma", "h2sigma", 2, 0, 2)
    tree.Draw("isCoveredAll>>h1sigma")
    tree.Draw("isCoveredAll2sigma>>h2sigma")
    assert(h1sigma.GetMean() <= h2sigma.GetMean())

def test_trivial_unfolding():
    ncat = 4
    nprocess = 4
    efficiencies = np.eye(4)
    ntrue = np.array([10, 20, 30, 40])
    bkg = np.zeros(4)
    ws = create_workspace(ncat, nprocess, ntrue, efficiencies, bkg)
    data_asimov = countingworkspace.utils.generate_asimov(ws)
    pdf = ws.obj('model')
    fr = pdf.fitTo(data_asimov, ROOT.RooFit.Save())
    assert(fr.status() == 0)

    for nproc, t in enumerate(ntrue):
        var_name = "nsignal_gen_proc%s" % format_index(nproc)
        assert ws.obj(var_name).getVal() == pytest.approx(t)
        assert ws.obj(var_name).getError() == pytest.approx(np.sqrt(t))


@pytest.mark.skip
def test_trivial_unfolding_gaus():
    ncat = 4
    nprocess = 4
    efficiencies = np.eye(4)
    ntrue = np.array([10, 20, 30, 40])
    bkg = np.zeros(4)
    ws = create_workspace(ncat, nprocess, ntrue, efficiencies, bkg, factory_model_cat='gaussian')

    data_asimov = countingworkspace.utils.generate_asimov(ws)
    pdf = ws.obj('model')
    fr = pdf.fitTo(data_asimov, ROOT.RooFit.Save())
    assert(fr.status() == 0)

    for nproc, t in enumerate(ntrue):
        var_name = "nsignal_gen_proc%s" % format_index(nproc)
        assert ws.obj(var_name).getVal() == pytest.approx(t)
        assert ws.obj(var_name).getError() == pytest.approx(np.sqrt(t))


def test_trivial_unfolding_gaus_fixed_error():
    ncat = 4
    nprocess = 4
    efficiencies = np.eye(4)
    ntrue = np.array([10, 20, 30, 40])
    bkg = np.zeros(4)
    ws = ROOT.RooWorkspace()
    expected_errors = [5., 10., 1., 0.001]
    for icat, error in enumerate(expected_errors):
        ws.factory('error_cat%s[%f]' % (format_index(icat), error))
    ws = create_workspace(ncat, nprocess, ntrue, efficiencies, bkg,
                          factory_model_cat='gaussian',
                          expression_nobs_err_cat='error_cat{cat}',
                          ws=ws
                          )

    data_asimov = countingworkspace.utils.generate_asimov(ws)
    pdf = ws.obj('model')
    fr = pdf.fitTo(data_asimov, ROOT.RooFit.Save())
    assert(fr.status() == 0)

    for nproc, t in enumerate(ntrue):
        var_name = "nsignal_gen_proc%s" % format_index(nproc)
        expected_error = ws.obj('error_cat%s' % format_index(nproc)).getVal()
        assert ws.obj(var_name).getVal() == pytest.approx(t, rel=0.1)
        assert ws.obj(var_name).getError() == pytest.approx(expected_error, rel=0.1)

