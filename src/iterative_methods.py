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

def animar_trayectoria(
    tray, 
    metodo_nombre="Método", 
    filename="trayectoria.gif", 
    convergio=True, 
    solucion=None,
    repeticiones_final=5,
    A=None, b=None  # Para graficar las líneas del sistema
):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patheffects import withStroke
    import numpy as np

    xs = [vec[0, 0] for vec in tray]
    ys = [vec[1, 0] for vec in tray]

    total_frames = len(xs) + repeticiones_final - 1
    fig, ax = plt.subplots()
    linea, = ax.plot([], [], 'o-', color='blue', label='Iteraciones')
    inicio, = ax.plot([], [], 'ro', label='Punto inicial')

    # Texto final en esquina inferior derecha del gráfico
    texto_final = ax.text(
        0.98, 0.02, '', 
        transform=ax.transAxes, 
        ha='right', va='bottom', 
        fontsize=11,
        weight='bold',
        path_effects=[withStroke(linewidth=3, foreground='white')]
    )

    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_title(f"Trayectoria de {metodo_nombre}")
    ax.grid(True)

    # Dibujo de las rectas del sistema Ax = b
    if A is not None and b is not None:
        x_vals = np.linspace(min(xs) - 5, max(xs) + 5, 400)
        for i in range(A.shape[0]):
            if A[i, 1] != 0:
                y_vals = (b[i, 0] - A[i, 0] * x_vals) / A[i, 1]
                ax.plot(x_vals, y_vals, linestyle='--', label=f"Ec {i+1}")
            else:
                x_const = b[i, 0] / A[i, 0]
                ax.axvline(x_const, linestyle='--', label=f"Ec {i+1}")

    def actualizar(frame):
        idx = min(frame, len(xs) - 1)

        if idx == 0:
            inicio.set_data([xs[0]], [ys[0]])

        linea.set_data(xs[:idx + 1], ys[:idx + 1])

        # Zoom dinámico
        margen = 1
        ax.set_xlim(min(xs[:idx + 1]) - margen, max(xs[:idx + 1]) + margen)
        ax.set_ylim(min(ys[:idx + 1]) - margen, max(ys[:idx + 1]) + margen)

        # Mostrar texto final si aplica
        if frame >= len(xs) - 1:
            if convergio:
                texto_final.set_text(f"✅ Convergió a {np.round(solucion.flatten(), 4)}")
                texto_final.set_color('green')
            else:
                texto_final.set_text("❌ Divergió")
                texto_final.set_color('red')
        else:
            texto_final.set_text("")

        return linea, inicio, texto_final

    anim = FuncAnimation(fig, actualizar, frames=total_frames, interval=200, blit=True)
    anim.save(filename, writer=PillowWriter(fps=1))
    plt.close()
    print(f"🎞️ GIF guardado como '{filename}'")

def animar_varias_trayectorias(
    trays_info,  # lista de tuplas (trayectoria, convergio, solucion, nombre_aprox)
    metodo_nombre="Método",
    filename="varias_trayectorias.gif",
    A=None, b=None,
    repeticiones_final=5
):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patheffects import withStroke
    import numpy as np

    colores = ['blue', 'green', 'purple']
    marcadores = ['o-', 's-', '^-']
    inicio_colors = ['red', 'orange', 'brown']

    num_approx = len(trays_info)
    max_len = max(len(t[0]) for t in trays_info)
    total_frames = max_len + repeticiones_final - 1

    fig, ax = plt.subplots()
    lineas = []
    inicios = []
    for i in range(num_approx):
        l, = ax.plot([], [], marcadores[i], color=colores[i], label=trays_info[i][3])
        lineas.append(l)
        ini, = ax.plot([], [], 'o', color=inicio_colors[i])
        inicios.append(ini)

    # Texto final
    texto_final = ax.text(
        0.98, 0.02, '', 
        transform=ax.transAxes, 
        ha='right', va='bottom',
        fontsize=11,
        weight='bold',
        path_effects=[withStroke(linewidth=3, foreground='white')]
    )

    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_title(f"Trayectorias ({metodo_nombre})")
    ax.grid(True)

    # Dibujo del sistema Ax = b
    if A is not None and b is not None:
        x_vals = np.linspace(-10, 10, 400)
        for i in range(A.shape[0]):
            if A[i, 1] != 0:
                y_vals = (b[i, 0] - A[i, 0] * x_vals) / A[i, 1]
                ax.plot(x_vals, y_vals, linestyle='--', label=f"Ec {i+1}")
            else:
                x_const = b[i, 0] / A[i, 0]
                ax.axvline(x_const, linestyle='--', label=f"Ec {i+1}")

    def actualizar(frame):
        min_xs, max_xs, min_ys, max_ys = [], [], [], []

        for i, (tray, _, _, _) in enumerate(trays_info):
            idx = min(frame, len(tray) - 1)
            xs = [v[0, 0] for v in tray[:idx + 1]]
            ys = [v[1, 0] for v in tray[:idx + 1]]

            lineas[i].set_data(xs, ys)
            inicios[i].set_data([tray[0][0, 0]], [tray[0][1, 0]])

            min_xs += xs
            max_xs += xs
            min_ys += ys
            max_ys += ys

        # Zoom dinámico
        margen = 1
        ax.set_xlim(min(min_xs) - margen, max(max_xs) + margen)
        ax.set_ylim(min(min_ys) - margen, max(max_ys) + margen)

        # Mensaje final
        if frame >= max_len - 1:
            textos = []
            for i, (tray, convergio, sol, nombre) in enumerate(trays_info):
                if convergio:
                    textos.append(f"✅ {nombre}: {np.round(sol.flatten(), 4)}")
                else:
                    textos.append(f"❌ {nombre}: Divergió")
            texto_final.set_text('\n'.join(textos))
        else:
            texto_final.set_text("")

        return lineas + inicios + [texto_final]

    ax.legend()
    anim = FuncAnimation(fig, actualizar, frames=total_frames, interval=800, blit=True)
    anim.save(filename, writer=PillowWriter(fps=5))
    plt.close()
    print(f"🎞️ GIF con varias trayectorias guardado como '{filename}'")
