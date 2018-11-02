import ROOT
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)


def safe_factory(func):
    def wrapper(self, *args, **kwargs):
        result = func(self, *args)
        if not result:
            raise ValueError('invalid factory input "%s"' % args)
        return result
    return wrapper


ROOT.RooWorkspace.factory = safe_factory(ROOT.RooWorkspace.factory)


def loop_iterator(iterator):
    object = iterator.Next()
    while object:
        yield object
        object = iterator.Next()


def iter_collection(rooAbsCollection):
    iterator = rooAbsCollection.createIterator()
    return loop_iterator(iterator)


def create_observed_number_of_events(ws, ncat, expression='nobs_cat{cat}', nmax=100000):
    logging.info('adding observables for {ncat} categories'.format(ncat=ncat))

    all_obs = ROOT.RooArgSet()
    for icat in range(ncat):
        _ = ws.factory((expression + '[0, {nmax}]').format(cat=icat, nmax=nmax))
        all_obs.add(_)
    ws.defineSet('all_obs', all_obs)


def create_variables(ws, expression, NVAR, values=None, ranges=None):
    values = values if values is not None else np.zeros(NVAR)
    if ranges is None:
        ranges = None, None
    ranges = np.asarray(ranges)
    if ranges.shape == (2, ):
        ranges = ((ranges[0], ranges[1]),) * NVAR
    for ivar, (value, (min_range, max_range)) in enumerate(zip(values, ranges)):
        if min_range is None and max_range is None:
            ws.factory((expression + '[{value}]').format(index0=ivar, value=value))
        else:
            ws.factory((expression + '[{value}, {m}, {M}]').format(index0=ivar, value=value,
                                                                   m=min_range, M=max_range))


def create_efficiencies(ws, efficiencies, expression='eff_cat{cat}_proc{proc}'):
    ncat, nprocesses = efficiencies.shape
    logging.info('adding efficiencies for {ncat} categories and {nproc} processes'.format(ncat=ncat, nproc=nprocesses))
    for icat in range(ncat):
        for iprocess in range(nprocesses):
            value = efficiencies[icat][iprocess]
            ws.factory((expression + '[{value}]').format(cat=icat, proc=iprocess, value=value))


def create_expected_number_of_signal_events(ws, ncat, nproc,
                                            expression_nexp='prod:nexp_signal_cat{cat}_proc{proc}(nsignal_gen_proc{proc}, eff_cat{cat}_proc{proc})'):
    logging.info('adding expected events for {ncat} categories and {nproc} processes'.format(ncat=ncat, nproc=nproc))
    all_expected = ROOT.RooArgSet()
    for icat in range(ncat):
        for iprocess in range(nproc):
            # expected events for given category and process
            all_expected.add(ws.factory(expression_nexp.format(cat=icat, proc=iprocess)))
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


def create_workspace(ncategories, nprocess, ntrue, efficiencies, expected_bkg_cat,
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
    # create the number of signal event at true level
    create_variables(ws, expression_nsignal_gen, nprocess, ntrue, (-10000, 50000))
    create_variables(ws, expression_nexp_bkg_cat, ncategories, expected_bkg_cat)
    create_expected_number_of_signal_events(ws, ncategories, nprocess)
    create_model(ws, ncategories, nprocess)
    return ws
