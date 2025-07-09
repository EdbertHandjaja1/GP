from surmise.emulation import emulator
import numpy as np

xvec_multi = np.array((T, H)).T.reshape(-1, 2)
emu_multi = emulator(x=xvec_multi, theta=thetavec, f=fmat,
                     method='PCGP',
                     args={'warnings': True})

