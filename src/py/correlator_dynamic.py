from numpy.fft import fft2, ifft2, fftfreq, fftshift, ifftshift, rfftfreq
from SFS.src.py.utils import *


def get_qw(param, mask=False):
    T, N, M, L, TIME = param
    q = (fftfreq(N, L / (2 * np.pi * N)))
    w = (fftfreq(M, TIME / (2 * np.pi * M)))
    q, w = np.meshgrid(q, w) 
    q, w = fftshift(q), fftshift(w)
    if mask:
        mask = q==0
        w = np.ma.array(w, mask=mask)
        q = np.ma.array(q, mask=mask)
    return q, w


#########################################
# Dynamic correlator and susceptibility #
#########################################

def get_CX(seed, folder, get_K, start=0):
    """
    Compute the correlation function and susceptibility in 1D.
    This function calculates the correlation function (Cqw) and susceptibility (Xqw)
    in the frequency and wavevector domain for a given field. It also applies a mask
    to filter out values below a threshold.
    Parameters:
        seed (int): Seed value used to identify the specific run folder.
        folder (str): Path to the base folder containing the data.
        get_K (callable): Function to compute the K field, which takes the field, 
                          configuration, and parameters as input.
        start (int, optional): Starting index for slicing the field and time arrays. 
                               Defaults to 0.
    Returns:
        tuple: A tuple containing:
            - (q, w) (tuple of np.ma.array): Wavevector (q) and frequency (w) arrays 
              with masked values below the threshold.
            - Cqw (np.ma.array): Correlation function in the frequency and wavevector domain.
            - Xqw (np.ma.array): Susceptibility in the frequency and wavevector domain.
            - con (dict): Configuration parameters from the run folder.
            - param (tuple): Tuple containing temperature (T), number of points (N), 
              number of samples (M), system size (L), and total time (TIME).
    """
    run_folder = folder + "{m}/".format(m = seed)
    X, d, N, L, T, dt, con = get_para(run_folder)

    fn = "varphi"
    field = get_field(run_folder, fn)
    samples = len(field)
    M = samples - start
    times = get_time(run_folder)[start:]
    field = field[start:]
    TIME = times[-1] - times[0]

    param = T, N, M, L, TIME
    q, w = get_qw(param)
    
    #? Apply Hannig window to the field
    # win = np.hanning(M) # hanning window
    # field = field * win[:, None]

    phiqw = fft2(field) * (L/N) * (TIME/M)
    phiqw = fftshift(phiqw)

    Cqw = phiqw * np.conj(phiqw) / L / TIME

    K = get_K(field, con, param)
    Kqw = fft2(K) * (L/N) * (TIME/M)
    Kqw = fftshift(Kqw) - q**2 * phiqw

    Xqw = 1 / (2*T) * (phiqw * np.conj(- 1j * w * phiqw - Kqw ) ) / L / TIME

    mask = Cqw < 1e-10

    w = np.ma.array(w, mask=mask)
    q = np.ma.array(q, mask=mask)
    Cqw = np.ma.array(Cqw, mask=mask)
    Xqw = np.ma.array(Xqw, mask=mask)
    
    return (q, w), Cqw, Xqw, con, param


def get_CX_av(num, folder):
    (q, w), C, X, con, param = get_CX(1, folder, get_K)
    count = 1

    for i in range(1, num):
        Ci, Xi = get_CX(i+1, folder, get_K)[1:3]
        C += Ci
        X += Xi
        count += 1
        
    C = C / count
    X = X / count

    if np.max(np.abs(np.imag(C))) < 1e-10: C = np.real(C)
    return (q, w), C, X, con, param


def get_CX_Nf(seed, folder, get_K, Nf=2, conserved=False):
    run_folder = folder + "{m}/".format(m = seed)
    field = np.array([get_field(run_folder, "varphi_"+str(a+1)) for a in range(Nf)])
    phiqw = np.array([fftshift(fft2(field[a])) for a in range(Nf)])
    X, d, N, L, T, dt, con = get_para(run_folder)
    M = len(phiqw[0])
    TIME = get_time(run_folder)[-1]
    param = T, N, M, L, TIME

    q, w = get_qw(param)
    mask = 0
    
    Cqw = np.zeros((Nf, Nf, *np.shape(q)), dtype=np.complex128) 
    for a in range(Nf):
        for b in range(Nf):
            Cqw[a, b] = phiqw[a] * np.conj(phiqw[b]) / L / TIME * (L/N)**2 * (TIME/M)**2
            mask = (np.abs(Cqw[a, b])<1e-15) | mask

    Xqw = np.zeros((Nf, Nf, *np.shape(q)), dtype=np.complex128)
    K = get_K(field, con)
    Kqw = np.array([fftshift(fft2(K[a])) - q**2 * phiqw[a] for a in range(Nf)])
    if conserved: Kqw = - q**2 * Kqw
    for a in range(Nf):
        for b in range(Nf):
            Xqw[a,b] = (phiqw[a] * np.conj(-1j*w*phiqw[b] - Kqw[b]))/(2*T) * L/N**2*TIME/M**2
    
    w = np.ma.array(w, mask=mask)
    q = np.ma.array(q, mask=mask)
    mask = np.array([[mask for a in range(Nf)] for b in range(Nf)])
    Cqw = np.ma.array(Cqw, mask=mask)
    Xqw = np.ma.array(Xqw, mask=mask)
    
    return (q, w), Cqw, Xqw, con, param


def get_phi_Nf(seed, folder, Nf=2):
    run_folder = folder + "{m}/".format(m = seed)
    field = np.array([get_field(run_folder, "varphi_"+str(a+1)) for a in range(Nf)])
    phiqw = np.array([fftshift(fft2(field[a])) for a in range(Nf)])
    TIME = get_time(run_folder)[-1]
    X, d, N, L, T, dt, con = get_para(run_folder)
    M = len(phiqw[0])
    return phiqw / L / TIME * (L/N)**2 * (TIME/M)**2

def get_phi_Nf_av(num, folder, Nf=2):
    phi = get_phi_Nf(1, folder, Nf=Nf)
    count = 1

    for i in range(1, num):
        phi += get_phi_Nf(i+1, folder, Nf=Nf)
        count += 1
    phi = phi / count
    
    phi2 = np.zeros((Nf, Nf, *np.shape(phi)[1:]), dtype=phi.dtype)
    for a in range(Nf):
        for b in range(Nf):
            phi2[a, b] = phi[a]*np.conj(phi[b])
    
    return phi2

# def get_CX_Nf_av(num, folder, get_K, Nf=2):
#     phi2 = get_phi_Nf_av(num, folder, Nf=Nf)

#     (q, w), C, X, con, param = get_CX_Nf(1, folder, get_K, Nf=Nf)
#     count = 1

#     for i in range(1, num):
#         Ci, Xi = get_CX_Nf(i+1, folder, get_K, Nf=Nf)[1:3] #- phi2
#         C += Ci
#         X += Xi
#         count += 1
        
#     C = C / count
#     X = X / count

#     return (q, w), C, X, con, param

def get_CX_Nf_av(num, folder, get_K, Nf=2, conserved=False):
    (q, w), C, X, con, param = get_CX_Nf(1, folder, get_K, Nf=Nf, conserved=conserved)
    count = 1

    for i in range(1, num):
        Ci, Xi = get_CX_Nf(i+1, folder, get_K, Nf=Nf, conserved=conserved)[1:3]
        C += Ci
        X += Xi
        count += 1
        
    C = C / count
    X = X / count

    return (q, w), C, X, con, param


def plot_Cqw_abs(C, q, w, param, size=6):
    """
    Plots the absolute value of the correlation function C(q, ω) on a logarithmic color scale.
    Parameters:
        C (numpy.ndarray): A 2D array representing the correlation function values.
        q (numpy.ndarray): A 1D array of momentum values corresponding to the rows of C.
        w (numpy.ndarray): A 1D array of frequency values corresponding to the columns of C.
        param (tuple): A tuple containing additional parameters (T, N, M, L, TIME).
        size (int, optional): The size of the plot. Default is 6.
    Returns:
        None: Displays the plot of |C(q, ω)| with a logarithmic color scale.
    """
    C = np.abs(C)
    lim = np.array([np.min(C[C>0]), np.max(C)])
    norm = colors.LogNorm(*lim)
    T, N, M, L, TIME = param
    fig, ax = plt.subplots(figsize=(size*1.4, size), sharex=True, sharey=True)

    p = ax.pcolor(w, q, C, norm=norm)

    ax.set_xlabel("$\\omega$")
    ax.set_ylabel("$q$")
    
    fig.colorbar(p, label="$C(q,\\omega)$", ax=ax)
    plt.show()


def plot_CXqw(C, X, q, w, param, wlim=None, qlim=None, size=12):
    T, N, M, L, TIME = param
    
    fig, ax = plt.subplots(1, 2, figsize=(size, size*0.4))

    p = ax[0].pcolor(w, q, C, norm=colors.LogNorm(np.min(C), np.max(C)))
    ax[0].set_xlabel("$\\omega$")
    ax[0].set_ylabel("$q$")
    fig.colorbar(p, label="$C(q,\\omega)$", ax=ax[0])

    ImX = np.imag(X)
    p = ax[1].pcolor(w, q, ImX)
    ax[1].set_xlabel("$\\omega$")
    ax[1].set_ylabel("$q$")
    fig.colorbar(p, label="$\\mathrm{Im} \\chi(q,\\omega)$", ax=ax[1])

    if not wlim is None:
        ax[0].set_xlim(-wlim, wlim)
        ax[1].set_xlim(-wlim, wlim)
    if not qlim is None:
        ax[0].set_ylim(-qlim, qlim)
        ax[1].set_ylim(-qlim, qlim)

    plt.tight_layout()
    return fig, ax


def plot_Cqw_Nf(C, q, w, param, cm=cm.viridis, size=6, Nf=2):
    T, N, M, L, TIME = param
    fig, ax = plt.subplots(Nf, Nf, figsize=(size*1.4, size), sharex=True, sharey=True,  gridspec_kw=dict(hspace=0.0, wspace=0.0))
    lim = np.max(np.abs(C))

    for a in range(Nf):
        for b in range(Nf):
            p = ax[a,b].pcolormesh(w.data, q.data, C[a,b], norm=colors.Normalize(-lim, lim), cmap=cm)

            ax[a,b].set_ylim(np.min(q), np.max(q))

    ax[-1, 0].set_xlabel("$\\omega$")
    ax[-1, 0].set_ylabel("$q$")
    
    fig.colorbar(p, label="$C(q,\\omega)$", ax=ax)

    plt.show()


def plot_abs_Nf(C, q, w, param, size=6, Nf=2):
    C = np.abs(C)
    lim = np.array([np.min(C[C>0]), np.max(C)])
    norm = colors.LogNorm(*lim)
    T, N, M, L, TIME = param
    fig, ax = plt.subplots(Nf, Nf, figsize=(size*1.4, size), sharex=True, sharey=True, gridspec_kw=dict(hspace=0.0, wspace=0.0))

    for a in range(Nf):
        for b in range(Nf):
            p = ax[a,b].pcolormesh(w.data, q.data, C[a,b], norm=norm)
            ax[a,b].set_ylim(np.min(q), np.max(q))
            
    ax[-1, 0].set_xlabel("$\\omega$")
    ax[-1, 0].set_ylabel("$q$")
    
    fig.colorbar(p, label="$C(q,\\omega)$", ax=ax)
    plt.show()
