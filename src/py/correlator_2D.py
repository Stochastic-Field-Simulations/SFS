from numpy.fft import fft2, ifft2, fftfreq, fftshift, ifftshift, rfftfreq
from SFS.src.py.utils import *


def get_Cqvec(seed, folder, start=0, nn=1, sub=''):
    run_folder = folder + "{m}/".format(m = seed) + sub
    X, d, N, L, T, dt, con = get_para(run_folder)

    fn = "varphi"
    field = get_field(run_folder, fn)
    q1 = (fftfreq(N, L / (2 * np.pi * N)))
    q2 = (fftfreq(N, L / (2 * np.pi * N)))
    q1, q2 = np.meshgrid(q1, q2)

    samples = len(field)
    M = samples - start

    avf = np.zeros_like(fft2(field[0]))
    for i in range(start, samples):
        phiq = (fft2(field[i])) * (L / N)**2
        avf += phiq / M

    Cq = np.zeros_like(q1)
    for i in range(start, samples):
        phiq = (fft2(field[i])) * (L / N)**2
        Cqi = np.abs(phiq - avf)**2 / L**2
        Cq += Cqi / M
    
    return (q1, q2), (Cq) , con, T, L, N, M 


def average_q(Cqveq, q, N, L):
    Cqveq = Cqveq.flatten()
    absq = np.sqrt(q[0]**2 + q[1]**2).flatten()
    qrange = (0, np.pi * (N + 1)/ L)
    q0 = np.linspace(*qrange, N)
    q0 = rfftfreq(N, L / (2 * np.pi * N))

    # Get index of absq so that q[i-1] < absq[indx[i]] <= q[i]
    indx = np.searchsorted(q0, absq)
    
    Cq = np.zeros(len(q0))
    n = np.zeros(len(q0))  

    for j in range(N**2):
        i = indx[j]
        if (i<len(q0)) & (Cqveq[j]>1e-20): Cq[i] += Cqveq[j]; n[i] += 1

    empty = (n==0.)
    Cq[~empty] = Cq[~empty] / n[~empty]
    Cq[empty] = 0

    return Cq, q0


def get_data_single(seed, folder, start, sub=''):
        q, Cqvec, con, T, L, N, M = get_Cqvec(seed, folder, start=start, sub=sub)
        mask = ((q[0]==0) & (q[1]==0))
        q2 = q[0]**2 + q[1]**2
        mask = mask | (Cqvec < 1e-15)
        CqvecMasked = np.ma.array(Cqvec, mask=mask)

        return (q, Cqvec, CqvecMasked, con, T, L, N, M)


def get_data(folder, start=200, sub=''):
    n = count_files(folder)

    seeds = range(1, n + 1)
    data = []
    lims = [np.inf, 0.]

    for seed in seeds:
        d = get_data_single(seed, folder, start, sub=sub)
        data.append(d)
        
        CqvecMasked = d[2]
        mm = [np.min(CqvecMasked), np.max(CqvecMasked)]
        lims[0] = min(lims[0], mm[0])
        lims[1] = max(lims[1], mm[1])

    return data, lims, n
