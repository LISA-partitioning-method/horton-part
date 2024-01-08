# HORTON-PART: molecular density partition schemes based on HORTON package.
# Copyright (C) 2023-2024 The HORTON-PART Development Team
#
# This file is part of HORTON-PART
#
# HORTON-PART is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# HORTON-PART is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --
"""
Module of CDIIS algorithm.
"""

import logging

import numpy as np
import scipy.linalg
from scipy.linalg import solve_triangular

__all__ = ["cdiis"]

logger = logging.getLogger(__name__)


def cdiis(
    x0,
    func,
    threshold,
    maxiter=1000,
    modeQR="full",
    mode="R-CDIIS",
    diis_size=5,
    param=0.1,
    minrestart=1,
    slidehole=False,
):
    r"""
    CDIIS algorithm.

    Multiple variations of the algorithm are implemented.
    The core of the function is to implement the restarted CDIIS (R-CDIIS) and the Adaptive-Depth CDIIS (AD-CDIIS)
    compared to the Fixed-Depth CDIIS (FD-CDIIS)

    Parameters
    ----------
    x : array_like
        1D array, initial values.
    func : callable
        Function `g` which satisfies :math:`g(x)=x`
    threshold : float
        Convergence threshold for the iterative process.
    param : float, optional
        default value : 0:1
        tau parameter for the R-CDIIS algorithm
        delta parameter for the AD-CDIIS algorithm
    threshold : float, optional
       default value : 1e-08
       tolerence parameter for convergence test on residual (commutator)
    maxiter : integer, optional
       default value : 50
       maximal number of iterations allowed
    mode : string, optional
       default value : "R-CDIIS"
       four modes available : "R-CDIIS", "AD-CDIIS", "FD-CDIIS", "Roothaan"
    diis_size : integer, optional
       default value : 5
       size of the window of stored previous iterates in the FD-CDIIS algorithm
       this dimension is also used for the adaptative algorithm
    minrestart : integer, optional
       default value : 1
       number of iterates we keep when a restart occurs
    modeQR : string, optional
       default value : "full"
       mode to build the QR decomposition of the matrix of residuals differences
       - full to compute the qr decomposition with scipy.linalg.qr
       - economic to use the economic mode of scipy.linalg.qr
       - otherwise : compute the qr decomposition with scipy.linalg.qr_insert and scipy.linalg.qr_delete
    slidehole : boolean, optional
       default value : False
       if True : allows hole in the AD-CDIIS algorithm


    Returns
    -------

    """

    def residual(x):
        """The residual."""
        return func(x) - x

    if mode == "Roothaan":
        booldiis = False
    else:
        booldiis = True

    logger.debug("\n\n\n-----------------------------------------\nCDIIS like program")
    logger.debug("---- Mode: " + str(mode))

    if mode == "R-CDIIS" or mode == "AD-CDIIS":
        logger.debug("---- param value: " + str(param))

    x = x0
    npar = len(x)
    if npar < diis_size:
        diis_size = npar
        logger.debug(
            f"The DIIS size is less than the number of parameters, and it is reduced to the number of parameters {npar}"
        )
    if mode == "FD-CDIIS":
        logger.debug("---- sizediis: " + str(diis_size))

    r = residual(x)  # residual
    # lists to save the iterates
    history_x = [x]
    history_r = [r]  # iterates of the current residual
    slist = []  # difference of residual (depending on the choice of CDIIS)
    rnormlist = []  # iterates of the current residual norm
    restartIt = []  # list of the iterations k when the R-CDIIS algorithm restarts
    mklist = []  # list of mk
    cnormlist = []

    # init
    mk = 0
    nbiter = 1
    # boolean to manage the first step
    notfirst = 0
    restart = True  # boolean to manage the QR decomposition when restart occurs
    history_dr = None
    xlast = None

    # for the reader of the paper
    if mode == "R-CDIIS":
        tau = param
    elif mode == "AD-CDIIS":
        delta = param

    # while the residual is not small enough
    # TODO: why r[-1] in original implementation?
    while np.linalg.norm(history_r[-1]) > threshold and nbiter < maxiter:
        # rlistIter.append(rlist)
        rnormlist.append(np.linalg.norm(r))
        mklist.append(mk)

        logger.debug("======================")
        logger.debug("iteration: " + str(nbiter))
        logger.debug("mk value: " + str(mk))
        logger.debug("||r(k)|| = " + str(np.linalg.norm(history_r[-1])))

        if mk > 0 and booldiis:
            # if there exist previous iterates and diis mode
            logger.debug("size of Cs: " + str(np.shape(history_dr)))
            if mode == "R-CDIIS":
                if modeQR == "full":
                    if mk == 1 or restart is True:
                        # if Q,R does not exist yet
                        restart = False
                        Q, R = scipy.linalg.qr(history_dr)
                    else:
                        # update Q,R from previous one
                        Q, R = scipy.linalg.qr_insert(Q, R, history_dr[:, -1], mk - 1, "col")
                elif modeQR == "economic":
                    Q, R = scipy.linalg.qr(history_dr, mode="economic")

            elif mode == "AD-CDIIS":
                Q, R = scipy.linalg.qr(history_dr, mode="economic")

            elif mode == "FD-CDIIS":
                if modeQR == "full":
                    if mk == 1:
                        # if Q,R does not exist yet
                        Q, R = scipy.linalg.qr(history_dr)
                    elif mk < diis_size:
                        # we only add a column
                        Q, R = scipy.linalg.qr_insert(Q, R, history_dr[:, -1], mk - 1, "col")
                    else:
                        if notfirst:
                            # of not the first time we reach the size
                            Q, R = scipy.linalg.qr_delete(Q, R, 0, which="col")
                            # we remove the first column
                        Q, R = scipy.linalg.qr_insert(Q, R, history_dr[:, -1], mk - 1, "col")
                        # we add a column
                        notfirst = 1

                elif modeQR == "economic":
                    Q, R = scipy.linalg.qr(history_dr, mode="economic")

            # the orthonormal basis as the subpart of Q denoted Q1
            Q1 = Q[:, 0:mk]

            ## Solve the LS equation R1 gamma = -Q_1^T r^(k-mk) or -Q_1^T r^(k)
            ## depending on the choice of algorithm, the RHS is not the same (last or oldest element)
            if mode == "AD-CDIIS" or mode == "FD-CDIIS":
                rhs = -np.dot(Q.T, np.reshape(history_r[-1], (-1, 1)))  # last : r^{k}

            elif mode == "R-CDIIS":
                rhs = -np.dot(Q.T, np.reshape(history_r[0], (-1, 1)))  # oldest : r^{k-m_k}

            # compute gamma solution of R_1 gamma = RHS
            gamma = solve_triangular(R[0:mk, 0:mk], rhs[0:mk], lower=False)
            # Note: gamma has the same shape as rhs which is 2D array.
            gamma = gamma.flatten()
            # compute c_i coefficients
            c = np.zeros(mk + 1)
            # the function gamma to c depends on the algorithm choice
            if mode == "AD-CDIIS" or mode == "FD-CDIIS":
                logger.debug(
                    "size of c: " + str(np.shape(c)[0]) + ", size of gamma: " + str(np.shape(gamma))
                )
                # Algorithm 3 and version P
                # c_0 = -gamma_1 (c_0, ... c_mk) and (gamma_1,...,gamma_mk)
                c[0] = -gamma[0]
                for i in range(1, mk):  # 1... mk-1
                    c[i] = gamma[i - 1] - gamma[i]  # c_i=gamma_i-gamma_i+1
                c[mk] = 1.0 - np.sum(c[0:mk])

            else:
                # mode == "R-CDIIS"
                c[0] = 1.0 - np.sum(gamma)
                for i in range(1, mk + 1):
                    c[i] = gamma[i - 1]

            # x_tilde
            x_tilde = np.zeros_like(x)
            for i in range(mk + 1):
                x_tilde = x_tilde + c[i] * history_x[i]
            # TODO: why np.inf norm?
            cnormlist.append(np.linalg.norm(c, np.inf))
        else:  #  ROOTHAAN (if booldiis==False) or first iteration of cdiis
            x_tilde = x.copy()
            cnormlist.append(1.0)

        # computation of the new dm k+1 from dmtilde
        x = func(x_tilde)
        history_x.append(x)
        # residual
        r = residual(x)
        logger.debug("||r_{k+1}|| = " + str(np.linalg.norm(r)))

        # compute the s^k vector
        if mode in ["AD-CDIIS", "FD-CDIIS"]:
            #  as the difference between the r^{k+1} and the last r^{k}
            s = r - history_r[-1]
        elif mode == "R-CDIIS":
            # as the difference between the r^k and the older r^{k-mk}
            s = r - history_r[0]
        elif mode == "Roothaan":
            s = r.copy()

        history_r.append(r)
        slist.append(s)

        if mk == 0 or not booldiis:  # we build the matrix of the s vector
            history_dr = np.reshape(s, (-1, 1))
        else:
            history_dr = np.hstack((history_dr, np.reshape(s, (-1, 1))))

        if mode == "R-CDIIS":
            if mk > 0:
                logger.debug(
                    "tau*||s^(k)|| = "
                    + str(tau * np.linalg.norm(history_dr[:, -1]))
                    + "   >?  ||s^(k)-Q*Q.T*s^(k)|| = "
                    + str(
                        np.linalg.norm(
                            history_dr[:, -1] - np.dot(Q1, np.dot(Q1.T, history_dr[:, -1]))
                        )
                    )
                )

                # Cs[:, -1] - np.dot(Q1, np.dot(Q1.T, Cs[:, -1]))
                if tau * np.linalg.norm(history_dr[:, -1]) > np.linalg.norm(
                    history_dr[:, -1] - Q1 @ Q1.T @ history_dr[:, -1]
                ):
                    # logger.info("********* Restart ***********")
                    restartIt.append(nbiter)
                    mk = minrestart - 1
                    # reinitialization
                    history_dr = history_dr[:, -minrestart:]
                    # print Cs
                    slist = slist[-minrestart:]
                    history_r = history_r[-minrestart:]
                    history_x = history_x[-minrestart:]
                    restart = True
                else:
                    mk += 1
            else:
                # if mk==0
                mk += 1

        if mode == "AD-CDIIS":
            # mode Adaptive-Depth
            mk += 1
            outNbr = 0
            indexList = []

            for l in range(0, mk - 1):
                # print l,np.linalg.norm(rlist[-1]),delta*np.linalg.norm(rlist[l])
                if np.linalg.norm(history_r[-1]) < (delta * np.linalg.norm(history_r[l])):
                    outNbr += 1
                    indexList.append(l)
                else:
                    if slidehole is False:
                        break

            if indexList:
                mk -= outNbr
                logger.debug("Indexes out :" + str(indexList))
                # delete the corresponding s vectores
                history_dr = np.delete(history_dr, indexList, axis=1)
                for ll in sorted(indexList, reverse=True):
                    # delete elements of each lists
                    slist.pop(ll)
                    history_r.pop(ll)
                    history_x.pop(ll)

            # Check if mk == npar + 1. This is actually FD-CDIIS
            if mk == npar + 1:
                logger.debug(str(np.shape(history_dr)))
                history_dr = history_dr[:, 1 : mk + 1]
                logger.debug(str(np.shape(history_dr)))
                history_x.pop(0)
                slist.pop(0)
                history_r.pop(0)
                mk -= 1

        elif mode == "FD-CDIIS":
            # keep only sizediis iterates
            if mk == diis_size:
                logger.debug(str(np.shape(history_dr)))
                history_dr = history_dr[:, 1 : mk + 1]
                logger.debug(str(np.shape(history_dr)))
                history_x.pop(0)
                slist.pop(0)
                history_r.pop(0)

            if mk < diis_size:
                mk += 1

        nbiter += 1
        xlast = x

    if np.linalg.norm(history_r[-1]) > threshold and nbiter == maxiter:
        conv = False
    else:
        conv = True
    return conv, nbiter - 1, rnormlist, mklist, cnormlist, xlast
