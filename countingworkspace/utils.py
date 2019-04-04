import ROOT
from array import array


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
    obs = ws.set('all_obs')
    pdf = ws.obj('ModelConfig').GetPdf()
    return ROOT.RooStats.AsymptoticCalculator.GenerateAsimovData(pdf, obs)


def generate_toys(ws, ntoys=100):
    all_obs_rooargset = ws.set('all_obs')
    return ws.pdf('model').generate(all_obs_rooargset, ntoys)


def generate_and_fit(ws, ws_generate=None, ntoys=100, snapshot='initial', snapshot_gen=None):
    ws_generate = ws_generate or ws
    snapshot_gen = snapshot_gen or snapshot
    ws_generate.loadSnapshot(snapshot_gen)
    toys = generate_toys(ws_generate, ntoys)
    ws.loadSnapshot(snapshot)
    for itoy in range(ntoys):
        toy_data = toys.get(itoy)
        # TODO: bad trick
        c = ws.cat('index')
        if c:
            toy_data.add(c)
        toy = ROOT.RooDataSet('toy_%d' % itoy, 'toy_%d' % itoy, toy_data)
        toy.add(toy_data)
        ws.loadSnapshot(snapshot)
        fr = ws.pdf('model').fitTo(toy, ROOT.RooFit.Save(True), ROOT.RooFit.PrintLevel(-2), ROOT.RooFit.Hesse(0))
        yield fr


class ToyStudyPlugin:
    def initialize(self, tree):
        self.tree = tree

    def run(self, fit_result):
        pass


class ToyStudyError(ToyStudyPlugin):
    def __init__(self, save_asym=False):
        self.save_asym = save_asym
        self.variables = set()  # filled during first event

    def initialize(self, tree):
        super().initialize(tree)
        self.all_values_error = {}
        if self.save_asym:
            self.all_values_error_up = {}
            self.all_values_error_down = {}

    def run(self, fit_result):
        r = iter_collection(fit_result.floatParsFinal())

        for rr in r:
            name = rr.GetName()
            if name not in self.variables:
                self.all_values_error[name] = array('f', [0])
                self.tree.Branch(name + '_error', self.all_values_error[name], '%s_error/F' % name)
                if self.save_asym:
                    self.all_values_error_up[name] = array('f', [0])
                    self.all_values_error_down[name] = array('f', [0])
                    self.tree.Branch(name + '_error_up', self.all_values_error[name], '%s_error_up/F' % name)
                    self.tree.Branch(name + '_error_down', self.all_values_error[name], '%s_error_down/F' % name)
                self.variables.add(name)
            self.all_values_error[name][0] = rr.getError()
            if self.save_asym:
                self.all_values_error_up[name][0] = rr.getErrorHi()
                self.all_values_error_down[name][0] = rr.getErrorLo()


def toy_study(ws, ws_generate=None, ntoys=100, snapshot='initial', snapshot_gen=None, seed=0, plugins=None):

    plugins = plugins or list()
    ROOT.RooRandom.randomGenerator().SetSeed(seed)

    f = ROOT.TFile.Open("result_%s.root" % ROOT.RooRandom.randomGenerator().GetSeed(), 'RECREATE')
    tree = ROOT.TTree('results', 'results')

    for plugin in plugins:
        plugin.initialize(tree)

    status = array('i', [0])
    nll = array('f', [0])
    all_values = {}

    tree.Branch('status', status, 'status/I')
    tree.Branch('nll', nll, 'nll/F')

    for result in generate_and_fit(ws, ws_generate, ntoys, snapshot, snapshot_gen):
        status[0] = result.status()
        nll[0] = result.minNll()

        for plugin in plugins:
            plugin.run(result)

        r = iter_collection(result.floatParsFinal())

        for rr in r:
            name = rr.GetName()
            if name not in all_values:
                all_values[name] = array('f', [0])
                tree.Branch(name, all_values[name], '%s/F' % name)
            all_values[name][0] = rr.getVal()

        tree.Fill()
    tree.Write()
    f.Close()
