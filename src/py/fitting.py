from numpy import log as ln
from scipy.optimize import curve_fit
from numpy.polynomial.polynomial import Polynomial as poly
from src_py.utils import *
from src_py.correlator_2D import *


def fitting(rs, yy, f):
    i_max = np.argmax(yy)
    
    ii_list     = np.array([], dtype=int) # indecies
    param_list  = np.array([[],]) # critical point
    err_list    = np.array([]) # error from curve fit
    resid_list  = np.array([]) # residue from linear fit
    ex_list     = np.array([]) # exponents from curve fit
    c_list      = np.array([]) # exponents from linear fit

    # which points to include?
    rng = range(i_max+1, len(rs)-10)
    for i, ii in enumerate(rng):
        x = rs[ii:]
        y = yy[ii:]

        try: p, cov = curve_fit(f, x, y, bounds=([0, rs[i_max], 0], [10, rs[ii], 10]))
        except: print("could not fit {i}".format(i=ii)); continue
        
        a, rc, ex = p
        lnx = ln(x - rc)
        lny = ln(y)

        try: lsq, report = poly.fit(lnx, lny, 1, full=True); resid_list = np.append(resid_list, [report[0][0]/len(lnx), ])
        except: print("could not lsq {i}".format(i=ii)); continue
        
        err = np.sqrt(np.sum(np.diag(cov)))
        
        ii_list = np.append(ii_list, [ii,])
        param_list = np.append(param_list, [[a, rc, ex],])
        ex_list = np.append(ex_list, [ex,])
        err_list = np.append(err_list, [err,])

        c_list = np.append(c_list, [- lsq.convert().coef[1],])

    param_list = param_list.reshape(len(ii_list), 3)
    return ii_list, param_list, err_list, resid_list, c_list 

def find_ii_best(err_list, resid_list):
    thresh = 2*np.min(resid_list)
    ii_best = -1
    for ii in np.argsort(err_list):
        if resid_list[ii] < thresh:
            ii_best = ii
            break
    return ii_best, thresh


def show_fit(rs, yy, ii_best, f, ylabel):
    ii_list, param_list, err_list, resid_list, c_list = fitting(rs, yy, f)
    
    ii = ii_list[ii_best]
    a, rc, ex = param_list[ii_best, :]

    fig, ax = plt.subplots(1, 2, figsize=(11,5), sharey=True)

    r = np.linspace(rc, np.max(rs), 200)
    label = "$\\propto |r - r_c|^{ -c }  $" + "\n" + "$r_c={rc:.4f}$\n$ c={ex:.4f}$".format(rc=rc, ex=ex)\
        + "\n" + "$j = {j}$".format(j=ii)

    ax[0].loglog(rs[ii:]-rc, yy[ii:], '.')
    ax[0].loglog(r-rc, f(r, a, rc, ex), 'k-', lw=2, label=label, zorder=0)
    ax[0].legend()
    ax[0].set_xlabel("$r - r_c$")
    ax[0].set_ylabel("$"+ylabel+"$")
    ax[0].set_xlim(np.min(rs[ii:]-rc), np.max(rs[ii:]-rc))
    ax[0].set_ylim(np.min(yy), np.max(yy))
    ax[0].set_xticklabels([], minor=True)

    ax[1].semilogy(rs-rc, yy, '.', label='$\\mathrm{Included}$')
    ax[1].semilogy(rs[:ii]-rc, yy[:ii], '.', label='$\\mathrm{Excluded}$')
    ax[1].semilogy(r-rc, f(r, a, rc, ex), 'k-',zorder=0)
    ax[1].legend()
    ax[1].set_xlabel("$r - r_c$")

    return fig


def get_crit_props(folder, sub=''):
    n = count_files(folder)

    xi = []
    xs = []
    rs = []

    for j in range(0, n):
        q, Cqveq, CqveqMasked, con, T, L, N, M = get_data_single(j+1, folder, 0, sub=sub)
        Cq, q0 = average_q(Cqveq, q, N, L)
        Cq, q0 = Cq[(Cq!=0)], q0[(Cq!=0)]
        r = con["r"]

        f = lambda q, A, B, xi : A / (1 + (q * xi)**B)
        p = curve_fit(f, q0, Cq)
        rs.append(r)
        xi.append(p[0][2])
        xs.append(Cq[0])

        param = con, T, L, N, M

    return rs, xi, xs, param

def plot_err(rs, y, f, label):

    ii_list, param_list, err_list, resid_list, c_list = fitting(rs, y, f)
    ii_best, thresh = find_ii_best(err_list, resid_list)

    fig, ax = plt.subplots()

    l1, = ax.semilogy(ii_list, err_list, '.-', label="$\\mathrm{err.}$")
    l2, = ax.plot(ii_list[ii_best], err_list[ii_best], 'rx', label='Best fit $'+label+'$')

    ax.set_ylabel("$\\mathrm{error}$")
    ax.set_xlabel("$j$")
    ax.set_title("Error best fit $"+label+"$, $j={j}$".format(j=ii_list[ii_best]))

    ax = ax.twinx()

    l3, = ax.semilogy(ii_list, resid_list, 'k.-', label="residue")
    l4, = ax.plot([ii_list[0], ii_list[-1]], 2*np.min(resid_list) * np.array([1, 1]), 'k--', label="thresh.")

    ax.set_yticklabels([], minor=True)

    ax.legend(handles=[l1, l2, l3, l4], fontsize=12)

    return fig, ii_best