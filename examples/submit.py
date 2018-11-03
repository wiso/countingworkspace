#!/bin/env python
from countingworkspace import create_workspace
import countingworkspace.utils
from countingworkspace.examples import NCATEGORIES, NPROCESS, NTRUE, EFFICIENCIES, EXPECTED_BKG_CAT, LUMI

countingworkspace.utils.silence_roofit()
ws = create_workspace(NCATEGORIES, NPROCESS, NTRUE, EFFICIENCIES, EXPECTED_BKG_CAT)
countingworkspace.utils.toy_study(ws, 1000)
