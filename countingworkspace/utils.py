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


def generate_toys(ws, ntoys=100):
    all_obs_rooargset = ws.set('all_obs')
    return ws.pdf('model').generate(all_obs_rooargset, ntoys)


def generate_and_fit(ws, ntoys=100):
    ws.loadSnapshot('initial')
    toys = generate_toys(ws, ntoys)
    for itoy in range(ntoys):
        toy_data = toys.get(itoy)
        toy = ROOT.RooDataSet('toy_%d' % itoy, 'toy_%d' % itoy, toy_data)
        toy.add(toy_data)
        ws.loadSnapshot('initial')
        fr = ws.pdf('model').fitTo(toy, ROOT.RooFit.Save(True), ROOT.RooFit.PrintLevel(-2), ROOT.RooFit.Hesse(0))
        yield fr


def toy_study(ws, ntoys=100, seed=0):
    from array import array
    ROOT.RooRandom.randomGenerator().SetSeed(seed)

    f = ROOT.TFile.Open("result_%s.root" % ROOT.RooRandom.randomGenerator().GetSeed(), 'CREATE')
    tree = ROOT.TTree('results', 'results')
    status = array('i', [0])
    nll = array('f', [0])
    all_r = {}

    tree.Branch('status', status, 'status/I')
    tree.Branch('nll', nll, 'nll/F')

    for result in generate_and_fit(ws, ntoys):
        status[0] = result.status()
        nll[0] = result.minNll()

        r = iter_collection(result.floatParsFinal())

        for rr in r:
            name = rr.GetName()
            if name not in all_r:
                all_r[name] = array('f', [0])
                tree.Branch(name, all_r[name], '%s/F' % name)
            all_r[name][0] = rr.getVal()

        tree.Fill()
    tree.Write()
    f.Close()
