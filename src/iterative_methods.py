import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import logging
from sys import stdout

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    stream=stdout,
    datefmt="%m-%d %H:%M:%S",
)


# ------------------ Métodos existentes ------------------

def gauss_jacobi(*, A: np.array, b: np.array, x0: np.array, tol: float, max_iter: int):
    n = A.shape[0]
    x = x0.copy()
    tray = [x.copy()]
    for _ in range(1, max_iter):
        x_new = np.zeros((n, 1))
        for i in range(n):
            suma = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - suma) / A[i, i]

        tray.append(x_new.copy())
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new.copy()
    return x, tray


def gauss_seidel(*, A: np.array, b: np.array, x0: np.array, tol: float, max_iter: int):
    n = A.shape[0]
    x = x0.copy()
    tray = [x.copy()]
    for _ in range(1, max_iter):
        x_new = x.copy()
        for i in range(n):
            suma = sum(A[i, j] * x_new[j] if j != i else 0 for j in range(n))
            x_new[i] = (b[i] - suma) / A[i, i]

        tray.append(x_new.copy())
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new.copy()
    return x, tray


# ------------------ Animación de trayectoria ------------------

def animar_trayectoria(tray, metodo_nombre="Metodo", filename="trayectoria.gif"):
    xs = [vec[0, 0] for vec in tray]
    ys = [vec[1, 0] for vec in tray]

    fig, ax = plt.subplots()
    ax.set_xlim(min(xs) - 1, max(xs) + 1)
    ax.set_ylim(min(ys) - 1, max(ys) + 1)
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_title(f"Trayectoria de {metodo_nombre}")
    linea, = ax.plot([], [], 'o-', color='blue')

    def actualizar(frame):
        linea.set_data(xs[:frame+1], ys[:frame+1])
        return linea,

    anim = FuncAnimation(fig, actualizar, frames=len(xs), interval=800, blit=True)
    anim.save(filename, writer=PillowWriter(fps=1))
    plt.close()
    print(f"✅ GIF guardado como '{filename}'")

# ------------------ Ejemplo de uso ------------------

