# -*- coding: utf-8 -*-
"""
Python 3
05 / 07 / 2024
@author: z_tjona

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth
"""

# ----------------------------- logging --------------------------
import logging
from sys import stdout
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import time

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    stream=stdout,
    datefmt="%m-%d %H:%M:%S",
)
logging.info(datetime.now())

# ----------------------------- contadores --------------------------
operaciones_sumrest = 0
operaciones_muldiv = 0

# ####################################################################
def eliminacion_gaussiana(A: np.ndarray) -> np.ndarray:
    """Resuelve un sistema de ecuaciones lineales mediante el método de eliminación gaussiana."""
    global operaciones_sumrest, operaciones_muldiv
    operaciones_sumrest = 0
    operaciones_muldiv = 0

    if not isinstance(A, np.ndarray):
        A = np.array(A)
    assert A.shape[0] == A.shape[1] - 1, "La matriz A debe ser de tamaño n-by-(n+1)."
    n = A.shape[0]

    for i in range(0, n - 1):
        p = None
        for pi in range(i, n):
            if A[pi, i] == 0:
                continue
            if p is None or abs(A[pi, i]) < abs(A[p, i]):
                p = pi

        if p is None:
            raise ValueError("No existe solución única.")

        if p != i:
            # logging.info(f"\nIntercambiando filas {i} y {p}")
            A[[i, p], :] = A[[p, i], :]
            # logging.info(f"\n{A}")

        for j in range(i + 1, n):
            m = A[j, i] / A[i, i]
            operaciones_muldiv += 1  # división
            tam = A.shape[1] - i
            A[j, i:] = A[j, i:] - m * A[i, i:]
            operaciones_muldiv += tam  # multiplicaciones
            operaciones_sumrest += tam  # restas

        # logging.info(f"\n{A}")

    if A[n - 1, n - 1] == 0:
        raise ValueError("No existe solución única.")

    solucion = np.zeros(n)
    solucion[n - 1] = A[n - 1, n] / A[n - 1, n - 1]
    operaciones_muldiv += 1

    for i in range(n - 2, -1, -1):
        suma = 0
        for j in range(i + 1, n):
            suma += A[i, j] * solucion[j]
            operaciones_muldiv += 1
            operaciones_sumrest += 1
        solucion[i] = (A[i, n] - suma) / A[i, i]
        operaciones_sumrest += 1
        operaciones_muldiv += 1

    # logging.info(f"Operaciones suma/resta: {operaciones_sumrest}")
    # logging.info(f"Operaciones multiplicación/división: {operaciones_muldiv}")
    return solucion

# ####################################################################
def descomposicion_LU(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Realiza la descomposición LU de una matriz cuadrada A."""
    global operaciones_sumrest, operaciones_muldiv
    operaciones_sumrest = 0
    operaciones_muldiv = 0

    A = np.array(A, dtype=float)
    n = A.shape[0]
    L = np.zeros((n, n), dtype=float)

    for i in range(0, n):
        if A[i, i] == 0:
            raise ValueError("No existe solución única.")
        L[i, i] = 1
        for j in range(i + 1, n):
            m = A[j, i] / A[i, i]
            operaciones_muldiv += 1
            tam = n - i
            A[j, i:] = A[j, i:] - m * A[i, i:]
            operaciones_muldiv += tam
            operaciones_sumrest += tam
            L[j, i] = m
        logging.info(f"\n{A}")

    logging.info(f"Operaciones suma/resta: {operaciones_sumrest}")
    logging.info(f"Operaciones multiplicación/división: {operaciones_muldiv}")
    return L, A

# ####################################################################
def resolver_LU(L: np.ndarray, U: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Resuelve un sistema de ecuaciones lineales mediante la descomposición LU."""
    global operaciones_sumrest, operaciones_muldiv

    n = L.shape[0]
    y = np.zeros((n, 1))
    y[0] = b[0] / L[0, 0]
    operaciones_muldiv += 1

    for i in range(1, n):
        suma = 0
        for j in range(i):
            suma += L[i, j] * y[j]
            operaciones_muldiv += 1
            operaciones_sumrest += 1
        y[i] = (b[i] - suma) / L[i, i]
        operaciones_sumrest += 1
        operaciones_muldiv += 1

    sol = np.zeros((n, 1))
    sol[-1] = y[-1] / U[-1, -1]
    operaciones_muldiv += 1

    for i in range(n - 2, -1, -1):
        suma = 0
        for j in range(i + 1, n):
            suma += U[i, j] * sol[j]
            operaciones_muldiv += 1
            operaciones_sumrest += 1
        sol[i] = (y[i] - suma) / U[i, i]
        operaciones_sumrest += 1
        operaciones_muldiv += 1

    logging.info(f"Operaciones suma/resta: {operaciones_sumrest}")
    logging.info(f"Operaciones multiplicación/división: {operaciones_muldiv}")
    return sol

# ####################################################################
def gauss_jordan(Ab: np.ndarray) -> np.ndarray:
    """Resuelve un sistema de ecuaciones lineales mediante el método de Gauss-Jordan."""
    global operaciones_sumrest, operaciones_muldiv
    operaciones_sumrest = 0
    operaciones_muldiv = 0

    if not isinstance(Ab, np.ndarray):
        Ab = np.array(Ab)
    n = Ab.shape[0]

    for i in range(0, n):
        p = None
        for pi in range(i, n):
            if Ab[pi, i] == 0:
                continue
            if p is None or abs(Ab[pi, i]) < abs(Ab[p, i]):
                p = pi

        if p is None:
            raise ValueError("No existe solución única.")

        if p != i:
            Ab[[i, p], :] = Ab[[p, i], :]
            # logging.info(f"\nIntercambiando filas {i} y {p}\n{Ab}")

        Ab[i, :] = Ab[i, :] / Ab[i, i]
        operaciones_muldiv += Ab.shape[1]

        for j in range(n):
            if i == j:
                continue
            m = Ab[j, i]
            Ab[j, :] = Ab[j, :] - m * Ab[i, :]
            operaciones_muldiv += Ab.shape[1]
            operaciones_sumrest += Ab.shape[1]

        # logging.info(f"\n{Ab}")

    solucion = Ab[:, -1]
    # logging.info(f"Operaciones suma/resta: {operaciones_sumrest}")
    # logging.info(f"Operaciones multiplicación/división: {operaciones_muldiv}")
    return solucion

# ####################################################################
def matriz_aumentada(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not isinstance(A, np.ndarray):
        A = np.array(A, dtype=float)
    if not isinstance(b, np.ndarray):
        b = np.array(b, dtype=float)
    return np.hstack((A, b.reshape(-1, 1)))

# ####################################################################
def separar_m_aumentada(Ab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(Ab, np.ndarray):
        Ab = np.array(Ab, dtype=float)
    return Ab[:, :-1], Ab[:, -1].reshape(-1, 1)

# Para graficar los tiempos
def comparar_tiempos(max_n=100):
    tiempos_gauss = []
    tiempos_jordan = []
    ns = list(range(1, max_n + 1))

    for n in ns:
        # Crear matriz aleatoria y vector b aleatorio
        A = np.random.rand(n, n) * 10
        b = np.random.rand(n) * 10
        Ab = matriz_aumentada(A, b)

        # Eliminar Gaussiana
        Ab_g = matriz_aumentada(A, b)
        start = time.time()
        try:
            eliminacion_gaussiana(Ab_g)
        except:
            pass
        end = time.time()
        tiempos_gauss.append(end - start)

        # Gauss-Jordan
        Ab_j = matriz_aumentada(A, b)
        start = time.time()
        try:
            gauss_jordan(Ab_j)
        except:
            pass
        end = time.time()
        tiempos_jordan.append(end - start)

    # Graficar
    plt.figure(figsize=(10,6))
    plt.plot(ns, tiempos_gauss, label="Eliminación Gaussiana", marker="o", markersize=3, linewidth=1)
    plt.plot(ns, tiempos_jordan, label="Gauss-Jordan", marker="x", markersize=3, linewidth=1)
    plt.xlabel("Número de incógnitas (n)")
    plt.ylabel("Tiempo de ejecución (segundos)")
    plt.title("Comparación de complejidad: Gaussiana vs Gauss-Jordan")
    plt.legend()
    plt.grid(True)
    plt.show()


