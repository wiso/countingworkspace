import ROOT


def loop_iterator(iterator):
    object = iterator.Next()
    while object:
        yield object
        object = iterator.Next()


def iter_collection(rooAbsCollection):
    iterator = rooAbsCollection.createIterator()
    return loop_iterator(iterator)


def silence_roofit():
    ROOT.RooMsgService.instance().getStream(1).removeTopic(ROOT.RooFit.NumIntegration)
    ROOT.RooMsgService.instance().getStream(1).removeTopic(ROOT.RooFit.Fitting)
    ROOT.RooMsgService.instance().getStream(1).removeTopic(ROOT.RooFit.Minimization)
    ROOT.RooMsgService.instance().getStream(1).removeTopic(ROOT.RooFit.InputArguments)
    ROOT.RooMsgService.instance().getStream(1).removeTopic(ROOT.RooFit.Eval)
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)


def get_free_variables(ws):
    all_vars = ws.allVars()
    all_obs = ws.set('all_obs')
    all_free = ROOT.RooArgSet()
    for v in iter_collection(all_vars):
        if not v.isConstant() and not all_obs.contains(v):
            all_free.add(v)
    return all_free


def generate_asimov(ws):
    all_obs_rooargset = ws.set('all_obs')
    asimov = ROOT.RooDataSet('asimov', 'asimov', all_obs_rooargset)
    ws.loadSnapshot('initial')
    for obs, exp in zip(iter_collection(all_obs_rooargset),
                        iter_collection(ws.set('all_exp'))):
        obs.setVal(exp.getVal())
    asimov.add(all_obs_rooargset)
    return asimov


def generate_toys(ws, ntoys=100):
    all_obs_rooargset = ws.set('all_obs')
    return ws.pdf('model').generate(all_obs_rooargset, ntoys)


def generate_and_fit(ws, ntoys=100):
    ws.loadSnapshot('initial')
    toys = generate_toys(ws, ntoys)
    for itoy in range(ntoys):
        toy_data = toys.get(itoy)
        # TODO: bad trick
        c = ws.cat('index')
        if c:
            toy_data.add(c)
        toy = ROOT.RooDataSet('toy_%d' % itoy, 'toy_%d' % itoy, toy_data)
        toy.add(toy_data)
        ws.loadSnapshot('initial')
        fr = ws.pdf('model').fitTo(toy, ROOT.RooFit.Save(True), ROOT.RooFit.PrintLevel(-2), ROOT.RooFit.Hesse(0))
        yield fr


def toy_study(ws, ntoys=100, seed=0, save_errors=True):
    from array import array
    ROOT.RooRandom.randomGenerator().SetSeed(seed)

    f = ROOT.TFile.Open("result_%s.root" % ROOT.RooRandom.randomGenerator().GetSeed(), 'RECREATE')
    tree = ROOT.TTree('results', 'results')
    status = array('i', [0])
    nll = array('f', [0])
    all_values = {}
    all_values_error = {}
    all_values_error_up = {}
    all_values_error_down = {}

    tree.Branch('status', status, 'status/I')
    tree.Branch('nll', nll, 'nll/F')

    for result in generate_and_fit(ws, ntoys):
        status[0] = result.status()
        nll[0] = result.minNll()

        r = iter_collection(result.floatParsFinal())

        for rr in r:
            name = rr.GetName()
            if name not in all_values:
                all_values[name] = array('f', [0])
                tree.Branch(name, all_values[name], '%s/F' % name)
                if save_errors:
                    all_values_error[name] = array('f', [0])
                    all_values_error_up[name] = array('f', [0])
                    all_values_error_down[name] = array('f', [0])
                    tree.Branch(name + '_error', all_values_error[name], '%s_error/F' % name)
                    tree.Branch(name + '_error_up', all_values_error[name], '%s_error_up/F' % name)
                    tree.Branch(name + '_error_down', all_values_error[name], '%s_error_down/F' % name)
            all_values[name][0] = rr.getVal()
            all_values_error[name][0] = rr.getError()
            all_values_error_up[name][0] = rr.getErrorHi()
            all_values_error_down[name][0] = rr.getErrorLo()

        tree.Fill()
    tree.Write()
    f.Close()
