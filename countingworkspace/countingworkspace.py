import logging
import string
from itertools import product

import numpy as np
import ROOT

from . import utils

logging.basicConfig(level=logging.INFO)


def format_index(n):
    return "%03d" % n


def string_range(n):
    if not np.issubdtype(type(n), np.integer) or n < 0:
        raise ValueError("parameter %s of type %s should be a non-negative integer" % (n, type(n)))
    return [format_index(nn) for nn in range(n)]


def safe_factory(func):
    def wrapper(self, *args, **kwargs):
        result = func(self, *args)
        if not result:
            raise ValueError('invalid factory input "%s"' % args)
        return result
    return wrapper


ROOT.RooWorkspace.factory = safe_factory(ROOT.RooWorkspace.factory)


def create_observed_number_of_events(ws, bins_cat, expression='nobs_cat{cat}', nmax=100000):
    if isinstance(bins_cat, int):
        bins_cat = string_range(bins_cat)
    logging.info('adding observables for %d categories', len(bins_cat))

    all_obs = ROOT.RooArgSet()
    for b in bins_cat:
        _ = ws.factory((expression + '[0, {nmax}]').format(cat=b, nmax=nmax))
        all_obs.add(_)
    ws.defineSet('all_obs', all_obs)


def create_scalar(ws, expression, value=None, ranges=None):
    is_formula = ':' in expression

    if ranges is not None and len(ranges) != 2:
        raise ValueError('size of ranges must be 2')
    if not is_formula and value is None:
        raise ValueError('you should specify a value for expression %s' % expression)
    if is_formula and ranges is not None:
        raise ValueError('cannot specify range for formula in expression %s' % expression)
    if not is_formula:
        if ranges is None:
            return ws.factory('{expression}[{value}]'.format(expression=expression, value=value))
        else:
            if value > ranges[1]:
                logging.warning("value (%f) is not in range (%f, %f), raising range", value, ranges[0], ranges[1])
                ranges[1] = ranges[0] + (value - ranges[0]) * 2
            if value < ranges[0]
                logging.warning("value (%f) is not in range (%f, %f), raising range", value, ranges[0], ranges[1])
                ranges[0] = ranges[1] - (ranges[0] - value) * 2
            return ws.factory('{expression}[{value},{down},{up}]'.format(expression=expression,
                                                                         value=value, down=ranges[0],
                                                                         up=ranges[1]))
    else:
        if value is not None:
            return ws.factory(expression.format(value=value))
        else:
            return ws.factory(expression)


def create_variables(ws, expression, nbins=None, bins=None, values=None, ranges=None, index_names=None):

    if nbins is not None:
        nbins = np.atleast_1d(nbins)

    if bins is not None:
        if not isinstance(bins[0], list) and not isinstance(bins[0], tuple):
            bins = [bins]

    if values is not None:
        values = np.atleast_1d(values)

    if ranges is not None:
        ranges = np.atleast_1d(ranges)

    if np.iterable(values) and nbins is None:
        nbins = np.shape(values)

    if bins is not None and nbins is None:
        nbins = [len(b) for b in bins]

    if nbins is not None and bins is None:
        bins = [string_range(nb) for nb in nbins]

    if nbins is not None and bins is not None:
        if [len(b) for b in bins] != list(nbins):
            raise ValueError('nbins=%s and bins=%s does not match' % (nbins, bins))

    if nbins is None or nbins == (1, ):
        return create_scalar(ws, expression, None if values is None else values[0], ranges)

    if values is not None and values.shape != nbins:
        raise ValueError('values has wrong shape, should be %s, it is %s' % (nbins, values.shape))

    index_names = np.atleast_1d(index_names)
    logging.debug('after normalizations inputs are: expression=%s, nbins=%s, bins=%s, values=%s, ranges=%s, index_names=%s',
                  expression, nbins, bins, values, ranges, index_names)

    if np.sum([b != 1 for b in nbins]) != np.sum([idx is not None for idx in index_names]):
        logging.debug('trying to determine indexes in expression %s', expression)
        possible_indexes = set([tup[1] for tup in string.Formatter().parse(expression) if tup[1] is not None])
        if values is not None:
            if 'value' in possible_indexes:
                possible_indexes.remove('value')
        if len(possible_indexes) == 0:
            raise ValueError('there is no index in expression %s' % expression)
        if len(possible_indexes) != 1:
            raise ValueError('cannot determine which index to use in expression %s, possible indexes: %s' % (expression, possible_indexes))
        index_names = list(possible_indexes)
        logging.debug('index_names = %s', index_names)

    if len(nbins) == 1:
        results = []
        if values is None:
            for b in bins[0]:
                results.append(create_variables(ws, expression.format(**{index_names[0]: b})))
        else:
            for v, b in zip(values, bins[0]):
                results.append(create_variables(ws, expression.format(**{index_names[0]: b, 'value': '{value}'}), values=v, ranges=ranges))
        return results
    else:
        if len(index_names) != len(nbins):
            raise ValueError('cannot find all the index')
        results = []
        if values is None:
            for ib, b in enumerate(bins[0]):
                logging.debug('calling with expression=%s, bins=%s', expression.replace('{%s}' % index_names[0], b), bins[1:])
                results.append(create_variables(ws, expression.replace('{%s}' % index_names[0], b), bins=bins[1:]))
        else:
            for ib, b in enumerate(bins[0]):
                logging.debug('calling with expression=%s, values=%s, bins=%s', expression.replace('{%s}' % index_names[0], b), values[ib, :], bins[1:])
                results.append(create_variables(ws, expression.replace('{%s}' % index_names[0], b), values=values[ib, :], bins=bins[1:]))
        return results


def create_efficiencies(ws, efficiencies, expression='eff_cat{cat}_proc{proc}',
                        bins_proc=None, bins_cat=None):
    ncat, nproc = efficiencies.shape
    if bins_proc is None or isinstance(bins_proc, int):
        bins_proc = string_range(nproc)
        if isinstance(bins_proc, int) and bins_proc != nproc:
            raise ValueError("Number of processes (%d) don't match number efficiency shape (%d, %d)" % (bins_proc, ncat, nproc))
    if bins_cat is None or isinstance(bins_cat, int):
        bins_cat = string_range(ncat)
        if isinstance(bins_cat, int) and bins_cat != ncat:
            raise ValueError("Number of categories (%d) don't match number efficiency shape (%d, %d)" % (bins_cat, ncat, nproc))
    logging.info('adding efficiencies for %d categories and %d processes', ncat, nproc)
    for icat, cat in enumerate(bins_cat):
        for iproc, proc in enumerate(bins_proc):
            value = efficiencies[icat][iproc]
            ws.factory((expression + '[{value}]').format(cat=cat, proc=proc, value=value))


def create_expected_number_of_signal_events(ws, bins_cat, bins_proc,
                                            expression_nsignal_gen='nsignal_gen_proc{proc}_with_sys',
                                            expression_efficiency='eff_cat{cat}_proc{proc}',
                                            expression_nexp='nexp_signal_cat{cat}_proc{proc}'):
    expression = 'prod:%s(%s, %s)' % (expression_nexp, expression_nsignal_gen, expression_efficiency)
    if isinstance(bins_cat, int):
        bins_cat = string_range(bins_cat)
    if isinstance(bins_proc, int):
        bins_proc = string_range(bins_proc)
    ncat, nproc = len(bins_cat), len(bins_proc)
    logging.info('adding expected events for %d categories and %d processes', ncat, nproc)
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
                 factory_model_cat='poisson',
                 expression_nobs_err_cat=None,
                 expression_model='model'):
    if isinstance(categories, int):
        categories = string_range(categories)
    if isinstance(processes, int):
        processes = string_range(processes)

    all_exp = ROOT.RooArgSet()
    for cat in categories:
        s = ','.join([expression_nexpected_signal_cat_proc.format(cat=cat, proc=proc) for proc in processes])
        if expression_nexpected_bkg_cat is not None:
            s += ',' + expression_nexpected_bkg_cat.format(cat=cat)
        var_expected = ws.factory('sum:{expression_nexpected_cat}({s})'.format(expression_nexpected_cat=expression_nexpected_cat.format(cat=cat), s=s))
        all_exp.add(var_expected)

    if factory_model_cat == 'gaussian' and expression_nobs_err_cat is None:
        raise NotImplementedError('not implemented due to a ROOT bug https://sft.its.cern.ch/jira/browse/ROOT-10069')
        expression_nobs_err_cat = 'nobs_err_cat{cat}'
        for cat in categories:
            ws.factory('expr:{nobs_err_cat}("sqrt(@0)", {nexp_cat})'.format(nobs_err_cat=expression_nobs_err_cat.format(cat=cat),
                                                                            nexp_cat=expression_nexpected_cat.format(cat=cat)))

    all_pdfs = []
    for cat in categories:

        if factory_model_cat == 'poisson':
            model = 'Poisson:{model_cat}({nobs_cat}, {nexp_cat})'.format(model_cat=expression_model_cat,
                                                                         nobs_cat=expression_nobs_cat,
                                                                         nexp_cat=expression_nexpected_cat)
        elif factory_model_cat == 'gaussian':
            model = 'RooGaussian:{model_cat}({nobs_cat}, {nexp_cat}, {nobs_err_cat})'.format(model_cat=expression_model_cat,
                                                                                             nobs_cat=expression_nobs_cat,
                                                                                             nexp_cat=expression_nexpected_cat,
                                                                                             nobs_err_cat=expression_nobs_err_cat)

        else:
            model = factory_model_cat
        all_pdfs.append(str(ws.factory(model.format(cat=cat)).getTitle()))
    ws.defineSet('all_exp', all_exp)
    ws.factory('PROD:%s(%s)' % (expression_model, ','.join(all_pdfs)))


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
        v1_name = var1.format(index0=format_index(ivar))
        v2_name = var2.format(index0=format_index(ivar))
        prod_name = name.format(index0=format_index(ivar))
        if ws.obj(v1_name) is None:
            raise ValueError('cannot find "%s"' % v1_name)
        if ws.obj(v2_name) is None:
            raise ValueError('cannot find "%s"' % v2_name)
        v = ws.factory('{}:{}({}, {})'.format(operation, prod_name, v1_name, v2_name))
        results.add(v)
    return results


def add(ws, var1, var2, name=None, nvar=None):
    if name is None:
        name = var1.replace('_{index0}', '') + '_plus_' + var2
    return dot(ws, var1, var2, name, nvar, operation='sum')


def create_workspace(categories, processes,
                     nsignal_gen=None,
                     efficiencies=None,
                     nexpected_bkg_cat=None,
                     systematics_nsignal_gen=None,
                     systematic_efficiencies=None,
                     expression_nobs_cat='nobs_cat{cat}',
                     expression_nsignal_gen='nsignal_gen_proc{proc}',
                     expression_nsignal_gen_with_sys='nsignal_gen_proc{proc}_with_sys',
                     expression_efficiency='eff_cat{cat}_proc{proc}',
                     expression_efficiency_with_sys='eff_cat{cat}_proc{proc}_with_sys',
                     expression_nexpected_bkg_cat='nexp_bkg_cat{cat}',
                     factory_model_cat='poisson',
                     expression_nobs_err_cat=None,
                     name_constrain='constrain_sys{sysname}',
                     ws=None):
    if isinstance(categories, int):
        categories = string_range(categories)
    if isinstance(processes, int):
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
    all_nuisances = ROOT.RooArgSet()
    create_observed_number_of_events(ws, categories, expression=expression_nobs_cat)
    create_efficiencies(ws, efficiencies, expression=expression_efficiency, bins_proc=processes, bins_cat=categories)
    if systematic_efficiencies is None:
        expression_efficiency_with_sys = expression_efficiency
    else:
        for sys_info in systematic_efficiencies:
            sysname = sys_info['name']
            if sysname not in sysnames:
                create_variables(ws, 'theta_{sysname}'.format(sysname=sysname), values=0, ranges=(-5, 5))
                all_nuisances.add(ws.var('theta_{sysname}'.format(sysname=sysname)))
            sysnames.add(sysname)
            sysvalues = np.asarray(sys_info['values'])
            if sysvalues.shape != (len(categories), len(processes)):
                raise ValueError('shape of efficiency systematic %s should match (ncategories, nprocess) = (%d, %d))' % (sysname, ncat, nproc))
            expression_response = 'expr:response_sys{sysname}_efficiency_cat{{cat}}_proc{{proc}}("1 + @0 * @1", sigma_{sysname}_efficiency_cat{{cat}}_proc{{proc}}[{{value}}], theta_{sysname})'.format(sysname=sysname)
            create_variables(ws, expression_response, bins=(categories, processes), values=sysvalues, index_names=('cat', 'proc'))
        sysnames_joint = ','.join(['response_sys{sysname}_efficiency_cat{{cat}}_proc{{proc}}'.format(sysname=sys_info['name'])
                                   for sys_info in systematic_efficiencies])
        create_variables(ws, 'prod:%s(%s, %s)' % (expression_efficiency_with_sys, expression_efficiency, sysnames_joint),
                         bins=(categories, processes), index_names=('cat', 'proc'))
    # create the number of signal event at true level, only if they are not all present
    if not all(ws.obj(expression_nsignal_gen.format(proc=proc)) for proc in processes):
        create_variables(ws, expression_nsignal_gen, bins=processes, values=nsignal_gen, ranges=(-10000, 50000))
    if systematics_nsignal_gen is None:
        expression_nsignal_gen_with_sys = expression_nsignal_gen
    else:
        for sys_info in systematics_nsignal_gen:
            sysname = sys_info['name']
            if sysname not in sysnames:
                create_variables(ws, 'theta_{sysname}'.format(sysname=sysname), values=0, ranges=(-5, 5))
                all_nuisances.add(ws.var('theta_{sysname}'.format(sysname=sysname)))
            sysnames.add(sysname)
            sysvalues = sys_info['values']
            if len(sysvalues) != nproc:
                raise ValueError('size of values for systematics {sysname} is different from the number of processes ({nproc})'.format(sysname=sysname, nproc=nproc))
            expression_response = 'expr:response_sys{sysname}_nprod_proc{{proc}}("1 + @0 * @1", sigma_{sysname}_nprod_proc{{proc}}[{{value}}], theta_{sysname})'.format(sysname=sysname)
            create_variables(ws, expression_response, bins=processes, values=sysvalues)
        sysnames_joint = ','.join(['response_sys{sysname}_nprod_proc{{proc}}'.format(sysname=sys_info['name'])
                                   for sys_info in systematics_nsignal_gen])
        create_variables(ws, 'prod:%s(%s, %s)' % (expression_nsignal_gen_with_sys, expression_nsignal_gen, sysnames_joint), bins=processes)
    create_variables(ws, expression_nexpected_bkg_cat, bins=categories, values=nexpected_bkg_cat)
    create_expected_number_of_signal_events(ws, categories, processes,
                                            expression_efficiency=expression_efficiency_with_sys,
                                            expression_nsignal_gen=expression_nsignal_gen_with_sys)

    all_constrains = ROOT.RooArgSet()
    all_globals = ROOT.RooArgSet()
    for sysname in sysnames:
        global_obs = ws.factory('global_{sysname}[0, -5, 5]'.format(sysname=sysname))
        global_obs.setConstant()
        nc = name_constrain.format(sysname=sysname)
        _ = ws.factory('RooGaussian:{name}(global_{sysname}, theta_{sysname}, 1)'.format(sysname=sysname, name=nc))
        all_constrains.add(_)
        all_globals.add(global_obs)
    ws.defineSet('constrains', all_constrains)
    ws.factory('PROD:prod_constrains(%s)' % ','.join([v.GetName() for v in utils.iter_collection(all_constrains)]))

    if sysnames:
        create_model(ws, categories, processes,
                     factory_model_cat=factory_model_cat, expression_nobs_err_cat=expression_nobs_err_cat,
                     expression_model='model_nosys')
        ws.factory('PROD:model(model_nosys, prod_constrains)')
    else:
        create_model(ws, categories, processes,
                     factory_model_cat=factory_model_cat,
                     expression_nobs_err_cat=expression_nobs_err_cat)
    ws.saveSnapshot('initial', ws.allVars())

    model_config = ROOT.RooStats.ModelConfig('ModelConfig', ws)
    model_config.SetPdf('model')
    model_config.SetObservables(ws.set('all_obs'))
    model_config.SetNuisanceParameters(all_nuisances)
    model_config.SetGlobalObservables(all_globals)
    poi = utils.get_free_variables(ws)
    poi.remove(all_nuisances)
    model_config.SetParametersOfInterest(poi)
    getattr(ws, 'import')(model_config)

    return ws
