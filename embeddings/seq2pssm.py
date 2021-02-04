import numpy as np

class s2p(object):
    
    def __init__(self):
        self.t2v = {}
        self.dim = None
    def pssm_normalized(self, id, seq, length=2000):
        pssmvec=np.loadtxt('../pssm/'+id+'.pssm')
        if len(pssmvec) > length:
            pssmvec= pssmvec[:length]
        elif len(pssmvec) <length:
            pssmvec= np.concatenate((pssmvec, np.zeros((length - len(pssmvec), 20))))
        return pssmvec
        del pssmvec
