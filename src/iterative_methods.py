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
            return x_new, tray, True
        x = x_new.copy()
    return x, tray, False


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
            return x_new, tray, True
        x = x_new.copy()
    return x, tray, False


# ------------------ Animación mejorada ------------------

def animar_trayectoria(tray, metodo_nombre="Método", filename="trayectoria.gif", convergio=True, solucion=None):
    xs = [vec[0, 0] for vec in tray]
    ys = [vec[1, 0] for vec in tray]

    fig, ax = plt.subplots()
    linea, = ax.plot([], [], 'o-', color='blue', label='Iteraciones')
    inicio, = ax.plot([], [], 'ro', label='Punto inicial')
    texto_final = ax.text(0.5, 1.05, '', transform=ax.transAxes, ha='center', fontsize=10, color='green')
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_title(f"Trayectoria de {metodo_nombre}")

    def actualizar(frame):
        if frame == 0:
            inicio.set_data([xs[0]], [ys[0]])
        linea.set_data(xs[:frame + 1], ys[:frame + 1])

        # Auto-ajuste de límites
        margen = 1
        ax.set_xlim(min(xs[:frame + 1]) - margen, max(xs[:frame + 1]) + margen)
        ax.set_ylim(min(ys[:frame + 1]) - margen, max(ys[:frame + 1]) + margen)

        # Mostrar mensaje final si es el último frame
        if frame == len(xs) - 1:
            if convergio:
                texto_final.set_text(f"✅ Convergió a {solucion.flatten()}")
                texto_final.set_color('green')
            else:
                texto_final.set_text("❌ Divergió")
                texto_final.set_color('red')

        return linea, inicio, texto_final

    anim = FuncAnimation(fig, actualizar, frames=len(xs), interval=800, blit=True)
    anim.save(filename, writer=PillowWriter(fps=1))
    plt.close()
    print(f"🎞️ GIF guardado como '{filename}'")

