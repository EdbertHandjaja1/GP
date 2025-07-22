from testfunctions import TestingfunctionBorehole
from testfunctions import TestingfunctionPiston
from testfunctions import TestingfunctionWingweight
from testfunctions import TestingfunctionOTLcircuit

class TestFuncCaller(object):
    def __init__(self, func):
        if func == 'borehole':
            meta = TestingfunctionBorehole.query_func_meta()
            nofailmodel = TestingfunctionBorehole.borehole_model
            true_func = TestingfunctionBorehole.borehole_true
        elif func == 'otlcircuit':
            meta = TestingfunctionOTLcircuit.query_func_meta()
            nofailmodel = TestingfunctionOTLcircuit.OTLcircuit_model
            true_func = TestingfunctionOTLcircuit.OTLcircuit_true
        elif func == 'wingweight':
            meta = TestingfunctionWingweight.query_func_meta()
            nofailmodel = TestingfunctionWingweight.Wingweight_model
            true_func = TestingfunctionWingweight.Wingweight_true
        elif func == 'piston':
            meta = TestingfunctionPiston.query_func_meta()
            nofailmodel = TestingfunctionPiston.Piston_model
            true_func = TestingfunctionPiston.Piston_true
        else:
            raise ValueError(
                'Choose between (\'borehole\', \'otlcircuit\', '
                '\'wingweight\', \'piston\')')
        
        self.info = {
            'function': meta['function'],
            'xdim': meta['xdim'],
            'thetadim': meta.get('thetadim', 0),
            'nofailmodel': nofailmodel,
            'true_func': true_func
        }