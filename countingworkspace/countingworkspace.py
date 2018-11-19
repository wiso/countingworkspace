import ROOT
import numpy as np
import logging
from itertools import product
from . import utils
logging.basicConfig(level=logging.INFO)


def string_range(n):
    return [str(nn) for nn in range(n)]


def safe_factory(func):
    def wrapper(self, *args, **kwargs):
        result = func(self, *args)
        if not result:
            raise ValueError('invalid factory input "%s"' % args)
        return result
    return wrapper


ROOT.RooWorkspace.factory = safe_factory(ROOT.RooWorkspace.factory)


def create_observed_number_of_events(ws, bins_cat, expression='nobs_cat{cat}', nmax=100000):
    if type(bins_cat) == int:
        bins_cat = string_range(bins_cat)
    logging.info('adding observables for {ncat} categories'.format(ncat=len(bins_cat)))

    all_obs = ROOT.RooArgSet()
    for b in bins_cat:
        _ = ws.factory((expression + '[0, {nmax}]').format(cat=b, nmax=nmax))
        all_obs.add(_)
    ws.defineSet('all_obs', all_obs)


def create_variables(ws, expression, bins=None, values=None, ranges=None):
    is_formula = ':' in expression

    if type(bins) is int:
        bins = string_range(bins)

    if is_formula:
        if values is not None:
            raise ValueError('cannot specify values for formula %s' % expression)
        if bins is None:
            ws.factory(expression)
        else:
            for b in bins:
                ws.factory(expression.format(index0=b))
    else:
        if bins is None and values is None:
            raise ValueError('need to specify bins and/or values')
        if values is None:
            values = np.zeros(len(bins))
        values = np.atleast_1d(values)
        if bins is None:
            bins = string_range(len(values))

        if ranges is None:
            ranges = None, None
        ranges = np.asarray(ranges)
        if ranges.shape == (2, ):
            ranges = ((ranges[0], ranges[1]),) * len(bins)

        for b, value, (min_range, max_range) in zip(bins, values, ranges):
            if min_range is None and max_range is None:
                ws.factory((expression + '[{value}]').format(index0=b, value=value))
            else:
                ws.factory((expression + '[{value}, {m}, {M}]').format(index0=b, value=value,
                                                                       m=min_range, M=max_range))


def create_efficiencies(ws, efficiencies, expression='eff_cat{cat}_proc{proc}',
                        bins_proc=None, bins_cat=None):
    ncat, nproc = efficiencies.shape
    if bins_proc is None or type(bins_proc) is int:
        bins_proc = string_range(nproc)
        if type(bins_proc) is int and bins_proc != nproc:
            raise ValueError("Number of processes (%d) don't match number efficiency shape (%d, %d)" % (bins_proc, ncat, nproc))
    if bins_cat is None or type(bins_cat) is int:
        bins_cat = string_range(ncat)
        if type(bins_cat) is int and bins_cat != ncat:
            raise ValueError("Number of categories (%d) don't match number efficiency shape (%d, %d)" % (bins_cat, ncat, nproc))
    logging.info('adding efficiencies for {ncat} categories and {nproc} processes'.format(ncat=ncat, nproc=nproc))
    for icat, cat in enumerate(bins_cat):
        for iproc, proc in enumerate(bins_proc):
            value = efficiencies[icat][iproc]
            ws.factory((expression + '[{value}]').format(cat=cat, proc=proc, value=value))


def create_expected_number_of_signal_events(ws, bins_cat, bins_proc,
                                            expression_nexp='prod:nexp_signal_cat{cat}_proc{proc}(nsignal_gen_proc{proc}, eff_cat{cat}_proc{proc})'):
    if type(bins_cat) is int:
        bins_cat = string_range(bins_cat)
    if type(bins_proc) is int:
        bins_proc = string_range(bins_proc)
    ncat, nproc = len(bins_cat), len(bins_proc)
    logging.info('adding expected events for {ncat} categories and {nproc} processes'.format(ncat=ncat, nproc=nproc))
    all_expected = ROOT.RooArgSet()
    for cat, proc in product(bins_cat, bins_proc):
        # expected events for given category and process
        all_expected.add(ws.factory(expression_nexp.format(cat=cat, proc=proc)))
    all_expected.setName('all_expected')
    ws.defineSet('all_signal_expected', all_expected)


def create_model(ws, ncat, nproc,
                 expression_nexpected_signal_cat_proc='nexp_signal_cat{cat}_proc{proc}',
                 expression_nexpected_bkg_cat_proc='nexp_bkg_cat{cat}',
                 expression_nexpected_cat='nexp_cat{cat}',
                 expression_nobserved='nobs_cat{cat}',
                 expression_model_cat='model_cat{cat}',
                 expression_model='model'
                 ):
    all_poissons = []
    all_exp = ROOT.RooArgSet()
    for icat in range(ncat):
        s = ','.join([expression_nexpected_signal_cat_proc.format(cat=icat, proc=iproc) for iproc in range(nproc)])
        if expression_nexpected_bkg_cat_proc is not None:
            s += ',' + expression_nexpected_bkg_cat_proc.format(cat=icat)
        var_expected = ws.factory('sum:{expression_nexpected_cat}({s})'.format(expression_nexpected_cat=expression_nexpected_cat.format(cat=icat), s=s))
        all_exp.add(var_expected)
        model = 'Poisson:{model_cat}({nobs_cat}, {nexp_cat})'.format(model_cat=expression_model_cat,
                                                                     nobs_cat=expression_nobserved,
                                                                     nexp_cat=expression_nexpected_cat)

        all_poissons.append(str(ws.factory(model.format(cat=icat)).getTitle()))
    ws.defineSet('all_exp', all_exp)
    ws.factory('PROD:%s(%s)' % (expression_model, ','.join(all_poissons)))


def dot(ws, var1, var2, name=None, nvar=None, operation='prod'):
    results = ROOT.RooArgSet()
    if name is None:
        name = var1.replace('_{index0}', '') + '_x_' + var2
    if nvar is None:
        s1 = ws.allVars().selectByName(var1.replace('{index0}', '*')).getSize()
        s2 = ws.allVars().selectByName(var2.replace('{index0}', '*')).getSize()
        if s1 == 0 or s2 == 0 or s1 != s2:
            raise ValueError('cannot find variables %s %s of same size' % (var1, var2))
        nvar = s1
    for ivar in range(nvar):
        v1_name = var1.format(index0=ivar)
        v2_name = var2.format(index0=ivar)
        prod_name = name.format(index0=ivar)
        if ws.obj(v1_name) is None:
            raise ValueError('cannot find "%s"' % v1_name)
        if ws.obj(v2_name) is None:
            raise ValueError('cannot find "%s"' % v2_name)
        v = ws.factory('{}:{}({}, {})'.format(operation, prod_name, v1_name, v2_name))
        results.add(v)
    return results


def sum(ws, var1, var2, name=None, nvar=None):
    if name is None:
        name = var1.replace('_{index0}', '') + '_plus_' + var2
    return dot(ws, var1, var2, name, nvar, operation='sum')


def create_workspace(ncategories, nprocess,
                     ntrue_signal_yield=None,
                     efficiencies=None,
                     expected_bkg_cat=None,
                     expression_nobs='nobs_cat{cat}',
                     expression_efficiency='eff_cat{cat}_proc{proc}',
                     expression_nsignal_gen='nsignal_gen_proc{index0}',
                     expression_nexp_bkg_cat='nexp_bkg_cat{index0}',
                     ws=None):
    if ncategories <= 0 or nprocess <= 0:
        raise ValueError('ncategories and nprocess must be positive')
    efficiencies = np.asarray(efficiencies)
    if efficiencies.shape != (ncategories, nprocess):
        raise ValueError('shape of efficiencies should match (ncategories, nprocess) = ()%d, %d)' % (ncategories, nprocess))

    ws = ws or ROOT.RooWorkspace()
    create_observed_number_of_events(ws, ncategories, expression=expression_nobs)
    create_efficiencies(ws, efficiencies, expression=expression_efficiency)
    # create the number of signal event at true level, only if they are not all present
    if not all(ws.obj(expression_nsignal_gen.format(index0=icat)) for icat in range(nprocess)):
        create_variables(ws, expression_nsignal_gen, nprocess, ntrue_signal_yield, (-10000, 50000))
    create_variables(ws, expression_nexp_bkg_cat, ncategories, expected_bkg_cat)
    create_expected_number_of_signal_events(ws, ncategories, nprocess)
    create_model(ws, ncategories, nprocess)
    ws.saveSnapshot('initial', ws.allVars())

    model_config = ROOT.RooStats.ModelConfig('ModelConfig', ws)
    model_config.SetPdf('model')
    model_config.SetObservables(ws.set('all_obs'))
    poi = utils.get_free_variables(ws)
    model_config.SetParametersOfInterest(poi)
    getattr(ws, 'import')(model_config)

    return ws
