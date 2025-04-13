from numpy.fft import fft2,  fftfreq, fftshift, ifftshift, rfftfreq
from src.py.utils import *


##########################
# Correlation from field #
##########################

def get_Cq(folder, seed=None, start=900, nn=1, sub='', M=None):
    """ Get Cq from field data for sev"""
    run_folder = folder
    if not (seed is None): run_folder = folder + "{m}/".format(m = seed) + sub
    X, d, N, L, T, dt, con = get_para(run_folder)

    q1 = (fftfreq(N, L / (2 * np.pi * N)))
    q2 = (fftfreq(N, L / (2 * np.pi * N)))
    q1, q2 = np.meshgrid(q1, q2)

    Cq = np.zeros((2, 2, *np.shape(q1)), dtype=np.complex128)

    fns = ["varphi_1", "varphi_2"]
    for a, fa in enumerate(fns):
        for b, fb in enumerate(fns):
    
            ffa = get_field(run_folder, fa)
            ffb = get_field(run_folder, fb)

            if (M is None): M = len(ffa)
            n = M - start

            for i in range(start, M):
                phiqa = (fft2(ffa[i])) * (L / N)**2
                phiqb = (fft2(ffb[i])) * (L / N)**2
                Cqi = phiqa * np.conjugate(phiqb) / L**2
                Cq[a, b] += Cqi / n
            
    return (q1, q2), (Cq) , con, T, L, N


def get_data_single(seed, folder, start, sub='', M=None):
    """ Get Cq and data from field data for single seed"""
    q, Cqvec, con, T, L, N = get_Cq(folder, seed=seed,  start=start, sub=sub, M=M)
    mask = ((q[0]==0) & (q[1]==0))
    q2 = q[0]**2 + q[1]**2
    mask = mask | (Cqvec < 1e-15)
    CqvecMasked = np.ma.array(Cqvec, mask=mask)
    lim = [np.min(np.abs(CqvecMasked)), np.max(np.abs(CqvecMasked))]

    return q, Cqvec, CqvecMasked, con, T, L, N, lim

def get_data(folder, start=200, sub=''):
    """ Get Cq and data from field data for multiple seeds"""
    n = count_files(folder)

    seeds = range(1, n + 1)
    data = []
    lims = [np.inf, 0.]

    for seed in seeds:
        d = get_data_single(seed, folder, start, sub=sub)
        data.append(d)
        
        lims[0] = min(lims[0], d[7][0])
        lims[1] = max(lims[1], d[7][1])

    return data, lims, n


def average_q(Cqvec, q, N, L):
    """ Average Cqvec over solid angle 2D"""
    Cqvec = Cqvec.reshape(2, 2, N**2)
    absq = np.sqrt(q[0]**2 + q[1]**2).flatten()
    qrange = (0, np.pi * (N + 1)/ L)

    q0 = np.linspace(*qrange, N)
    q0 = rfftfreq(N, L / (2 * np.pi * N))

    # Get index of absq so that q[i-1] < absq[indx[i]] <= q[i]
    indx = np.searchsorted(q0, absq)
    
    Cq  = np.zeros((2, 2, *np.shape(q0)))
    n   = np.zeros((2, 2, *np.shape(q0)))

    for j in range(N**2):
        i = indx[j]-1

        for a in range(2):
            for b in range(2):
                if (np.abs(Cqvec[a, b, j])>1e-20): Cq[a, b, i] += Cqvec[a, b, j]; n[a, b, i] += 1

    empty = (n==0.)
    Cq[~empty] = Cq[~empty] / n[~empty]
    Cq[empty] = 0

    return Cq, q0

def get_corr(folder, seed=None, start=0, M=None):
    """Get the correlation function from field"""
    q, Cqvec, CqvecMasked, con, T, L, N, lim = get_data_single(seed, folder, start, M=M)
    Cq, q0 = average_q(Cqvec, q, N, L)
    return Cq, q0, (con, T, N, L)
 




#############################
# Correlation from saved Cq #
#############################

def get_corr_saved(folder):
    """Get the correlation function from saved Cabq"""
    data = h5py.File(folder+"Cabq")
    keys = [k for k in data.keys()][1:]
    C = np.array([data[k] for k in keys])
    return np.array(
        [[[[Ctpab[0] + Ctpab[1] * 1j for Ctpab in Ctpa] for Ctpa in Ctp] for Ctp in Ct] for Ct in C]
        )

def get_Cq_saved(folder, seed=None, sub=''):
    run_folder = folder
    if not (seed is None): run_folder = folder + "{m}/".format(m = seed) + sub
    X, d, N, L, T, dt, con = get_para(run_folder)

    qrange = (0, np.pi * (N + 1)/ L)
    q0 = rfftfreq(N, L / (2 * np.pi * N))

    Cq = get_corr_saved(run_folder)
            
    return q0, Cq, con, T, L, N
