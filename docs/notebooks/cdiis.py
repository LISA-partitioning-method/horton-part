import logging

import numpy as np
import scipy.linalg
from scipy.linalg import solve_triangular
from setup import prepare_argument_dict, prepare_grid_and_dens, print_results

from horton_part import LinearISAWPart, check_pro_atom_parameters, compute_quantities

logging.basicConfig(level=logging.ERROR, format="%(levelname)s:    %(message)s")
logger = logging.getLogger(__name__)

np.set_printoptions(precision=8, linewidth=np.inf, suppress=True)


def cdiis_algo(
    bs_funcs,
    rho,
    propars,
    points,
    weights,
    threshold,
    density_cutoff,
    # parameters for CDIIS
    # for all CDIIS
    maxiter=1000,
    modeQR="full",
    mode="R-CDIIS",
    # FD-CDIIS
    sizediis=5,
    # R-CDIIS and AD-CDIIS
    param=0.1,
    # R-CDIIS
    minrestart=1,
    # AD-CDIIS
    slidehole=False,
):
    """
    CDiis algorithm
    --------------

    CDIIS algorithm. Multiple variations of the algorithm are implemented.
    The core of the function is to implement the restarted CDIIS (R-CDIIS) and the Adaptive-Depth CDIIS (AD-CDIIS) compared to the Fixed-Depth CDIIS (FD-CDIIS)

    Parameters
    ----------
    param : float
        default value : 0:1
        tau parameter for the R-CDIIS algorithm
        delta parameter for the AD-CDIIS algorithm
    threshold : float
       default value : 1e-08
       tolerence parameter for convergence test on residual (commutator)
    maxiter : integer
       default value : 50
       maximal number of iterations allowed
    mode : string
       default value : "R-CDIIS"
       four modes available : "R-CDIIS", "AD-CDIIS", "FD-CDIIS", "Roothaan"
    sizediis : integer
       default value : 8
       size of the window of stored previous iterates in the FD-CDIIS algorithm
       this dimension is also used for the adaptative algorithm
    minrestart : integer
       default value : 1
       number of iterates we keep when a restart occurs
    modeQR : string
       default value : "full"
       mode to build the QR decomposition of the matrix of residuals differences
       - full to compute the qr decomposition with scipy.linalg.qr
       - economic to use the economic mode of scipy.linalg.qr
       - otherwise : compute the qr decomposition with scipy.linalg.qr_insert and scipy.linalg.qr_delete
    slidehole : boolean
       default value : False
       if True : allows hole in the AD-CDIIS algorithm
    name : string
        default value ""
        name of the computation (to identify the log file : diis_namevalue.log)

    Outputs
    -------
    conv : boolean
       if convergence, True, else, False
    rnormlist : numpy.array
       list of norm of r_k
    mklist : numpy.array
       list of m_k value at each step
    cnormlist : numpy.array
       list of the iterates of the norm of the c_i coefficients
    dmlast : numpy.array
       last computed density matrix
    """

    def func_g(x):
        """The objective fixed-point equation."""
        pro_shells, _, _, ratio, _ = compute_quantities(rho, x, bs_funcs, density_cutoff)
        return np.einsum("ip,p->i", pro_shells * ratio, weights)

    def error(x):
        """The residual."""
        return func_g(x) - x

    # booldiis : boolean
    # default value : True
    # if False : Roothann algorithm
    # if True : one of the CDIIS algorithm
    if mode == "Roothaan":
        booldiis = False
    else:
        booldiis = True

    logger.info("\n\n\n-----------------------------------------\nCDIIS like program")
    logger.info("---- Mode: " + str(mode))
    if mode == "R-CDIIS" or mode == "AD-CDIIS":
        logger.info("---- param value: " + str(param))

    npar = len(propars)
    if npar < sizediis:
        sizediis = npar
        logger.info(
            f"The DIIS size is less than the number of parameters, and it is reduced to the number of parameters {npar}"
        )
    if mode == "FD-CDIIS":
        logger.info("---- sizediis: " + str(sizediis))

    ## compute the initial values
    # the initial values are x0
    dm = propars

    # commutator : residual
    r = error(propars)  # residual

    # lists to save the iterates
    dmlist = [dm]
    rlist = [r]  # iterates of the current residual
    # rlistIter = []  # the residuals family we keep at iteration k
    slist = []  # difference of residual (depending on the choice of CDIIS)
    rnormlist = []  # iterates of the current residual norm
    restartIt = []  # list of the iterations k when the R-CDIIS algorithm restarts
    mklist = []  # list of mk
    cnormlist = []

    # init
    gamma = 1.0
    mk = 0
    nbiter = 1
    # boolean to manage the first step
    notfirst = 0
    restart = True  # boolean to manage the QR decomposition when restart occurs
    Cs = None

    # for the reader of the paper
    if mode == "R-CDIIS":
        tau = param
    elif mode == "AD-CDIIS":
        delta = param

    # while the residual is not small enough
    # TODO: why r[-1]?
    while np.linalg.norm(rlist[-1]) > threshold and nbiter < maxiter:
        # rlistIter.append(rlist)
        rnormlist.append(np.linalg.norm(r))
        mklist.append(mk)

        logger.info("======================")
        logger.info("iteration: " + str(nbiter))
        logger.info("mk value: " + str(mk))
        logger.info("||r(k)|| = " + str(np.linalg.norm(rlist[-1])))

        if mk > 0 and booldiis:
            # if there exist previous iterates and diis mode
            logger.info("size of Cs: " + str(np.shape(Cs)))
            if mode == "R-CDIIS":
                if modeQR == "full":
                    if mk == 1 or restart is True:
                        # if Q,R does not exist yet
                        restart = False
                        Q, R = scipy.linalg.qr(Cs)
                    else:
                        # update Q,R from previous one
                        Q, R = scipy.linalg.qr_insert(Q, R, Cs[:, -1], mk - 1, "col")
                elif modeQR == "economic":
                    Q, R = scipy.linalg.qr(Cs, mode="economic")

            elif mode == "AD-CDIIS":
                Q, R = scipy.linalg.qr(Cs, mode="economic")

            elif mode == "FD-CDIIS":
                if modeQR == "full":
                    if mk == 1:
                        # if Q,R does not exist yet
                        Q, R = scipy.linalg.qr(Cs)
                    elif mk < sizediis:
                        # we only add a column
                        Q, R = scipy.linalg.qr_insert(Q, R, Cs[:, -1], mk - 1, "col")
                    else:
                        if notfirst:
                            # of not the first time we reach the size
                            Q, R = scipy.linalg.qr_delete(Q, R, 0, which="col")
                            # we remove the first column
                        Q, R = scipy.linalg.qr_insert(Q, R, Cs[:, -1], mk - 1, "col")
                        # we add a column
                        notfirst = 1

                elif modeQR == "economic":
                    Q, R = scipy.linalg.qr(Cs, mode="economic")

            # the orthonormal basis as the subpart of Q denoted Q1
            Q1 = Q[:, 0:mk]

            ## Solve the LS equation R1 gamma = -Q_1^T r^(k-mk) or -Q_1^T r^(k)
            ## depending on the choice of algorithm, the RHS is not the same (last or oldest element)
            if mode == "AD-CDIIS" or mode == "FD-CDIIS":
                rhs = -np.dot(Q.T, np.reshape(rlist[-1], (-1, 1)))  # last : r^{k}

            elif mode == "R-CDIIS":
                rhs = -np.dot(Q.T, np.reshape(rlist[0], (-1, 1)))  # oldest : r^{k-m_k}

            # compute gamma solution of R_1 gamma = RHS
            # TODO: this is problematic if sizediis is larger than the size
            # print(R[0:mk, 0:mk])
            gamma = solve_triangular(R[0:mk, 0:mk], rhs[0:mk], lower=False)
            # Note: gamma has the same shape as rhs which is 2D array.
            gamma = gamma.flatten()
            # compute c_i coefficients
            c = np.zeros(mk + 1)
            # the function gamma to c depends on the algorithm choice
            if mode == "AD-CDIIS" or mode == "FD-CDIIS":
                logger.info(
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

            # dmtilde
            dmtilde = np.zeros_like(dm)
            for i in range(mk + 1):
                dmtilde = dmtilde + c[i] * dmlist[i]
            # TODO: why np.inf norm?
            cnormlist.append(np.linalg.norm(c, np.inf))
        else:  #  ROOTHAAN (if booldiis==False) or first iteration of cdiis
            dmtilde = dm.copy()
            cnormlist.append(1.0)

        # computation of the new dm k+1 from dmtilde
        dm = func_g(dmtilde)
        dmlist.append(dm)
        # residual
        r = error(dm)
        logger.info("||r_{k+1}|| = " + str(np.linalg.norm(r)))

        # compute the s^k vector
        if mode in ["AD-CDIIS", "FD-CDIIS"]:
            #  as the difference between the r^{k+1} and the last r^{k}
            s = r - rlist[-1]
        elif mode == "R-CDIIS":
            # as the difference between the r^k and the older r^{k-mk}
            s = r - rlist[0]
        elif mode == "Roothaan":
            s = r.copy()

        rlist.append(r)
        slist.append(s)

        if mk == 0 or not booldiis:  # we build the matrix of the s vector
            Cs = np.reshape(s, (-1, 1))
        else:
            Cs = np.hstack((Cs, np.reshape(s, (-1, 1))))

        if mode == "R-CDIIS":
            if mk > 0:
                logger.info(
                    "tau*||s^(k)|| = "
                    + str(tau * np.linalg.norm(Cs[:, -1]))
                    + "   >?  ||s^(k)-Q*Q.T*s^(k)|| = "
                    + str(np.linalg.norm(Cs[:, -1] - np.dot(Q1, np.dot(Q1.T, Cs[:, -1]))))
                )

                # Cs[:, -1] - np.dot(Q1, np.dot(Q1.T, Cs[:, -1]))
                if tau * np.linalg.norm(Cs[:, -1]) > np.linalg.norm(
                    Cs[:, -1] - Q1 @ Q1.T @ Cs[:, -1]
                ):
                    # logger.info("********* Restart ***********")
                    restartIt.append(nbiter)
                    mk = minrestart - 1
                    # reinitialization
                    Cs = Cs[:, -minrestart:]
                    # print Cs
                    slist = slist[-minrestart:]
                    rlist = rlist[-minrestart:]
                    dmlist = dmlist[-minrestart:]
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
                if np.linalg.norm(rlist[-1]) < (delta * np.linalg.norm(rlist[l])):
                    outNbr += 1
                    indexList.append(l)
                else:
                    if slidehole is False:
                        break

            if indexList:
                mk -= outNbr
                logger.info("Indexes out :" + str(indexList))
                # delete the corresponding s vectores
                Cs = np.delete(Cs, indexList, axis=1)
                for ll in sorted(indexList, reverse=True):
                    # delete elements of each lists
                    slist.pop(ll)
                    rlist.pop(ll)
                    dmlist.pop(ll)

            # Check if mk == npar + 1. This is actually FD-CDIIS
            if mk == npar + 1:
                logger.info(str(np.shape(Cs)))
                Cs = Cs[:, 1 : mk + 1]
                logger.info(str(np.shape(Cs)))
                dmlist.pop(0)
                slist.pop(0)
                rlist.pop(0)
                mk -= 1

        elif mode == "FD-CDIIS":
            # keep only sizediis iterates
            if mk == sizediis:
                logger.info(str(np.shape(Cs)))
                Cs = Cs[:, 1 : mk + 1]
                logger.info(str(np.shape(Cs)))
                dmlist.pop(0)
                slist.pop(0)
                rlist.pop(0)

            if mk < sizediis:
                mk += 1

        nbiter += 1
        dmlast = dm

    if np.linalg.norm(rlist[-1]) > threshold and nbiter == maxiter:
        conv = False
    else:
        conv = True

    # return conv, nbiter - 1, rnormlist, mklist, cnormlist, dmlast
    check_pro_atom_parameters(dmlast)
    if not conv:
        raise RuntimeError("Not converged!")
    return dmlast


if __name__ == "__main__":
    import os

    TS42 = [
        "C2H2",
        "C2H4",
        "C2H5OH",
        "C2H6",
        "C3H6",
        "C3H7OH",
        "C3H8",
        "C4H10O",
        "C4H10",
        "C4H8",
        "C5H12",
        "C6H14",
        "C6H6",
        "C7H16",
        "C8H18",
        "CCl4",
        "CH3CH2OCH2CH3",
        "CH3CH3CH3N",
        "CH3CHO",
        "CH3COCH3",
        "CH3NH2",
        "CH3NHCH3",
        "CH3OCH3",
        "CH3OH",
        "CH4",
        "CO2",
        "COS",
        "CO",
        "CS2",
        "Cl2",
        "H2CO",
        "H2O",
        "H2S",
        "H2",
        "HBr",
        "HCl",
        "HF",
        "N2O",
        "N2",
        "NH3",
        "SO2",
        "SiH4",
    ]
    ISA_TESTS = [
        "H2O",
        "HF",
        "HCN",
        "CO",
        "C6H6",
        "CH4",
        "H3O+",
        "N3-",
    ]

    fchk_path = "/Users/yingxing/Projects/ISA-benchmark/latest-draft/dataset/fchk"

    solver_customized_options = {
        "R-CDIIS": {"param": 1e-2, "minrestart": 1},  # 1e-2
        "FD-CDIIS": {"sizediis": 5},  # 3, 4, 5
        "AD-CDIIS": {"param": 1e-4},  # 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6
    }

    # mode = "FD-CDIIS"
    mode = "AD-CDIIS"
    # mode = "R-CDIIS"

    # for name in TS42 + ISA_TESTS:
    for name in ["COS"]:
        # for name in ["COS"]:
        print(name)
        mol, grid, rho = prepare_grid_and_dens(os.path.join(fchk_path, f"{name}_LDA.fchk"))
        kwargs = prepare_argument_dict(mol, grid, rho)
        kwargs["solver"] = cdiis_algo
        # four modes available : "R-CDIIS", "AD-CDIIS", "FD-CDIIS", "Roothaan"
        kwargs["solver_options"] = {
            "mode": mode,
            "maxiter": 1000,
        }
        kwargs["solver_options"].update(solver_customized_options[mode])
        kwargs["inner_threshold"] = 1e-8
        part = LinearISAWPart(**kwargs)
        part.do_all()
        print_results(part)
        print()
