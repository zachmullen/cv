import numpy as np


def denoise(im, U_init, tolerance=0.1, tau=0.125, tv_weight=100):
    m, n = im.shape
    U = U_init
    Px = Py = im
    error = 1

    while error > tolerance:
        Uold = U

        GradUx = np.roll(U, -1, axis=1) - U
        GradUy = np.roll(U, -1, axis=0) - U

        PxNew = Px + (tau/tv_weight)*GradUx
        PyNew = Py + (tau/tv_weight)*GradUy
        NormNew = np.maximum(1, np.sqrt(PxNew**2 + PyNew**2))

        Px = PxNew/NormNew
        Py = PyNew/NormNew

        RxPx = np.roll(Px, 1, axis=1)
        RyPy = np.roll(Py, 1, axis=0)

        DivP = (Px - RxPx) + (Py - RyPy)

        U = im + tv_weight*DivP

        error = np.linalg.norm(U-Uold)/np.sqrt(n*m)

    return U, im - U
