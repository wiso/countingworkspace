import ROOT
import numpy as np
import logging
import string
from itertools import product
from . import utils
logging.basicConfig(level=logging.INFO)


def string_range(n):
    if type(n) is not int or n < 0:
        raise ValueError("parameters should be a non-negative integer")
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


def create_variables(ws, expression, bins=None, values=None, ranges=None, index_name=None):
    is_formula = ':' in expression

    if np.iterable(values) and bins is None:
        bins = len(values)

    if type(bins) is int:
        bins = string_range(bins)

    if (bins is not None or np.iterable(values)) and index_name is None:
        possible_indexes = set([tup[1] for tup in string.Formatter().parse(expression) if tup[1] is not None])
        if values is not None:
            if 'value' in possible_indexes:
                possible_indexes.remove('value')
        if len(possible_indexes) == 0:
            raise ValueError('there is no index in expression %s' % expression)
        if len(possible_indexes) != 1:
            raise ValueError('cannot determine which index to use in expression %s, possible indexes: %s' % (expression, possible_indexes))
        index_name = list(possible_indexes)[0]

    if is_formula:
        if bins is None:
            ws.factory(expression.format(value=values))
        else:
            if values is not None:
                if len(values) != len(bins):
                    raise ValueError('size of bins (%d) and values (%d) is different' % (len(bins), len(values)))
                for b, value in zip(bins, values):
                    ws.factory(expression.format(**{index_name: b, 'value': value}))
            else:
                for b in bins:
                    ws.factory(expression.format(**{index_name: b}))
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
                ws.factory((expression + '[{value}]').format(**{index_name: b, 'value': value}))
            else:
                ws.factory((expression + '[{value}, {m}, {M}]').format(**{index_name: b, 'value': value,
                                                                          'm': min_range, 'M': max_range}))


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
                                            expression_nsignal_gen='nsignal_gen_proc{proc}_with_sys',
                                            expression_efficiency='eff_cat{cat}_proc{proc}',
                                            expression_nexp='nexp_signal_cat{cat}_proc{proc}'):
    expression = 'prod:%s(%s, %s)' % (expression_nexp, expression_nsignal_gen, expression_efficiency)
    if type(bins_cat) is int:
        bins_cat = string_range(bins_cat)
    if type(bins_proc) is int:
        bins_proc = string_range(bins_proc)
    ncat, nproc = len(bins_cat), len(bins_proc)
    logging.info('adding expected events for {ncat} categories and {nproc} processes'.format(ncat=ncat, nproc=nproc))
    all_expected = ROOT.RooArgSet()
    for cat, proc in product(bins_cat, bins_proc):
        # expected events for given category and process
        all_expected.add(ws.factory(expression.format(cat=cat, proc=proc)))
    all_expected.setName('all_expected')
    ws.defineSet('all_signal_expected', all_expected)


def create_model(ws, categories, processes,
                 expression_nexpected_signal_cat_proc='nexp_signal_cat{cat}_proc{proc}',
                 expression_nexpected_bkg_cat='nexp_bkg_cat{cat}',
                 expression_nexpected_cat='nexp_cat{cat}',
                 expression_nobs_cat='nobs_cat{cat}',
                 expression_model_cat='model_cat{cat}',
                 expression_model='model'):
    if type(categories) is int:
        categories = string_range(categories)
    if type(processes) is int:
        processes = string_range(processes)

    all_poissons = []
    all_exp = ROOT.RooArgSet()
    for cat in categories:
        s = ','.join([expression_nexpected_signal_cat_proc.format(cat=cat, proc=proc) for proc in processes])
        if expression_nexpected_bkg_cat is not None:
            s += ',' + expression_nexpected_bkg_cat.format(cat=cat)
        var_expected = ws.factory('sum:{expression_nexpected_cat}({s})'.format(expression_nexpected_cat=expression_nexpected_cat.format(cat=cat), s=s))
        all_exp.add(var_expected)
        model = 'Poisson:{model_cat}({nobs_cat}, {nexp_cat})'.format(model_cat=expression_model_cat,
                                                                     nobs_cat=expression_nobs_cat,
                                                                     nexp_cat=expression_nexpected_cat)

        all_poissons.append(str(ws.factory(model.format(cat=cat)).getTitle()))
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


def create_workspace(categories, processes,
                     nsignal_gen=None,
                     efficiencies=None,
                     nexpected_bkg_cat=None,
                     systematics_nsignal_gen=None,
                     expression_nobs_cat='nobs_cat{cat}',
                     expression_nsignal_gen='nsignal_gen_proc{proc}',
                     expression_nsignal_gen_with_sys='nsignal_gen_proc{proc}_with_sys',
                     expression_efficiency='eff_cat{cat}_proc{proc}',
                     expression_nexpected_bkg_cat='nexp_bkg_cat{cat}',
                     ws=None):
    if type(categories) is int:
        categories = string_range(categories)
    if type(processes) is int:
        processes = string_range(processes)
    if not len(categories) or not len(processes):
        raise ValueError('ncategories and nprocess must be positive')

    nproc = len(processes)
    ncat = len(categories)
    sysnames = set()

    efficiencies = np.asarray(efficiencies)
    if efficiencies.shape != (len(categories), len(processes)):
        raise ValueError('shape of efficiencies should match (ncategories, nprocess) = ()%d, %d)' % (ncat, nproc))

    ws = ws or ROOT.RooWorkspace()
    create_observed_number_of_events(ws, categories, expression=expression_nobs_cat)
    create_efficiencies(ws, efficiencies, expression=expression_efficiency, bins_proc=processes, bins_cat=categories)
    # create the number of signal event at true level, only if they are not all present
    if not all(ws.obj(expression_nsignal_gen.format(proc=proc)) for proc in processes):
        create_variables(ws, expression_nsignal_gen, processes, nsignal_gen, (-10000, 50000))
    if systematics_nsignal_gen is None:
        expression_nsignal_gen_with_sys = expression_nsignal_gen
    else:
        for sys_info in systematics_nsignal_gen:
            sysname = sys_info['name']
            if sysname not in sysnames:
                create_variables(ws, 'theta_{sysname}'.format(sysname=sysname), values=0, ranges=(-5, 5))
            sysnames.add(sysname)
            sysvalues = sys_info['values']
            if len(sysvalues) != nproc:
                raise ValueError('size of values for systematics {sysname} is different from the number of processes ({nproc})'.format(sysname=sysname, ncat=nproc))
            expression_response = 'expr:response_sys{sysname}_proc{{proc}}("1 + @0 * @1", sigma_{sysname}_proc{{proc}}[{{value}}], theta_{sysname})'.format(sysname=sysname)
            create_variables(ws, expression_response, processes, values=sysvalues)
        sysnames_joint = ','.join(['response_sys{sysname}_proc{{proc}}'.format(sysname=sys_info['name'])
                                    for sys_info in systematics_nsignal_gen])
        create_variables(ws, 'prod:%s(%s, %s)' % (expression_nsignal_gen_with_sys, expression_nsignal_gen, sysnames_joint), bins=processes)
    create_variables(ws, expression_nexpected_bkg_cat, categories, nexpected_bkg_cat)
    create_expected_number_of_signal_events(ws, categories, processes, expression_nsignal_gen=expression_nsignal_gen_with_sys)

    all_constrains = ROOT.RooArgSet()
    for sysname in sysnames:
        _ = ws.factory('RooGaussian:constrain_sys{sysname}(global_{sysname}[0, -5, 5], theta_{sysname}, 1)'.format(sysname=sysname))
        all_constrains.add(_)
    ws.defineSet('constrains', all_constrains)
    ws.factory('PROD:prod_constrains(%s)' % ','.join([v.GetName() for v in utils.iter_collection(all_constrains)]))

    if sysnames:
        create_model(ws, categories, processes, expression_model='model_nosys')
        ws.factory('PROD:model(model_nosys, prod_constrains)')
    else:
        create_model(ws, categories, processes)
    ws.saveSnapshot('initial', ws.allVars())

    model_config = ROOT.RooStats.ModelConfig('ModelConfig', ws)
    model_config.SetPdf('model')
    model_config.SetObservables(ws.set('all_obs'))
    poi = utils.get_free_variables(ws)
    model_config.SetParametersOfInterest(poi)
    getattr(ws, 'import')(model_config)

    return ws
