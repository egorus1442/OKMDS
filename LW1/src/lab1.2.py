import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.animation import FuncAnimation

# Задаем начальные параметры
T = np.linspace(0, 20, 5001)  # Время
l = 1.5
m = 0.2

# Условия
def r(t):
    return 2 + np.cos(6 * t)

def phi(t):
    return 7 * t + 1.2 * np.cos(6 * t)

# Вычисляем координаты в декартовой системе
def rx(t):
    return r(t) * np.cos(phi(t))

def ry(t):
    return r(t) * np.sin(phi(t))

# Скорости
def vpx(t):
    return np.gradient(rx(t), t)

def vpy(t):
    return np.gradient(ry(t), t)

def vp(t):
    return np.sqrt(vpx(t)**2 + vpy(t)**2)

# Ускорения
def wpx(t):
    return np.gradient(vpx(t), t)

def wpy(t):
    return np.gradient(vpy(t), t)

def W(t):
    return np.sqrt(wpx(t)**2 + wpy(t)**2)

def Wt(t):
    return np.gradient(vp(t), t)

def Wn(t):
    return np.sqrt(W(t)**2 - Wt(t)**2)

# Массивы для хранения координат, скоростей и ускорений
xn = np.zeros_like(T)
yn = np.zeros_like(T)
vx = np.zeros_like(T)
vy = np.zeros_like(T)
v = np.zeros_like(T)
wt = np.zeros_like(T)
wn = np.zeros_like(T)

# Заполняем массивы значениями
for i, t_val in enumerate(T):
    xn[i] = rx(t_val)
    yn[i] = ry(t_val)
    vx[i] = vpx(T)[i]
    vy[i] = vpy(T)[i]
    v[i] = vp(T)[i]
    wt[i] = Wt(T)[i]
    wn[i] = Wn(T)[i]

# Настройка графика
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.axis('equal')
ax.plot(xn, yn)

P, = ax.plot(xn[0], yn[0], marker='o')  # Точка, которая будет двигаться

# Угловая скорость
Phi = math.atan2(vy[0], vx[0])

VLine = ax.plot([xn[0], xn[0] + vx[0]], [yn[0], yn[0] + vy[0]], 'black')[0]

# Функция для поворота вектора (x, y) на угол a
def rotate(x, y, a):
    x_rotated = x * np.cos(a) - y * np.sin(a)
    y_rotated = x * np.sin(a) + y * np.cos(a)
    return x_rotated, y_rotated

# Стрелки для отображения скоростей
V_arrow_x = np.array([-v[0] * 0.1, 0.0, -v[0] * 0.1], dtype=float)
V_arrow_y = np.array([v[0] * 0.05, 0.0, -v[0] * 0.05], dtype=float)
V_arrow_rotx, V_arrow_roty = rotate(V_arrow_x, V_arrow_y, Phi)
V_arrow, = ax.plot(xn[0] + vx[0] + V_arrow_rotx, yn[0] + vy[0] + V_arrow_roty, color="black")

# Стрелка для углового ускорения
WTLine = ax.plot([xn[0], xn[0] + wt[0] * math.cos(Phi)], [yn[0], yn[0] + wt[0] * math.sin(Phi)], 'red')[0]

WT_arrow_x = np.array([-wt[0] * 0.1, 0.0, -wt[0] * 0.1], dtype=float)
WT_arrow_y = np.array([wt[0] * 0.05, 0.0, -wt[0] * 0.05], dtype=float)
WT_arrow_rotx, WT_arrow_roty = rotate(WT_arrow_x, WT_arrow_y, Phi)
WT_arrow, = ax.plot(xn[0] + wt[0] * math.cos(Phi) + WT_arrow_rotx, yn[0] + wt[0] * math.sin(Phi) + WT_arrow_roty, color="red")

# Стрелка для нормального ускорения
WNLine = ax.plot([xn[0], xn[0] + wn[0] * math.cos(Phi - math.pi / 2)], [yn[0], yn[0] + wn[0] * math.sin(Phi - math.pi / 2)], 'blue')[0]

WN_arrow_x = np.array([-wn[0] * 0.1, 0.0, -wn[0] * 0.1], dtype=float)
WN_arrow_y = np.array([wn[0] * 0.05, 0.0, -wn[0] * 0.05], dtype=float)
WN_arrow_rotx, WN_arrow_roty = rotate(WN_arrow_x, WN_arrow_y, Phi - math.pi / 2)
WN_arrow, = ax.plot(xn[0] + wn[0] * math.cos(Phi - math.pi / 2) + WN_arrow_rotx, yn[0] + wn[0] * math.sin(Phi - math.pi / 2) + WN_arrow_roty, color="blue")

# Добавляем радиус-вектор
RLine = ax.plot([0, xn[0]], [0, yn[0]], 'green')[0]

R_arrow_x = np.array([-0.1, 0.0, -0.1], dtype=float)
R_arrow_y = np.array([0.05, 0.0, -0.05], dtype=float)
R_arrow_rotx, R_arrow_roty = rotate(R_arrow_x, R_arrow_y, math.atan2(yn[0], xn[0]))
R_arrow, = ax.plot(xn[0] + R_arrow_rotx, yn[0] + R_arrow_roty, color="green")

arrow_scale = 0.1

# Функция обновления для анимации
def cha(i):
    # Обновление положения точки (передаем как массивы с одним элементом)
    P.set_data([xn[i]], [yn[i]])

    # Пересчитываем угловую скорость для текущего времени
    Phi = math.atan2(vy[i], vx[i])

    # Обновление линии скорости
    VLine.set_data([xn[i], xn[i] + vx[i] * arrow_scale], [yn[i], yn[i] + vy[i] * arrow_scale])

    # Обновление стрелки скорости
    V_arrow_x = np.array([-v[i] * 0.1 * arrow_scale, 0.0, -v[i] * 0.1 * arrow_scale], dtype=float)
    V_arrow_y = np.array([v[i] * 0.05 * arrow_scale, 0.0, -v[i] * 0.05 * arrow_scale], dtype=float)
    V_arrow_rotx, V_arrow_roty = rotate(V_arrow_x, V_arrow_y, Phi)
    V_arrow.set_data(xn[i] + vx[i] * arrow_scale + V_arrow_rotx, yn[i] + vy[i] * arrow_scale + V_arrow_roty)

    # Обновление линии углового ускорения
    WTLine.set_data([xn[i], xn[i] + wt[i] * math.cos(Phi) * arrow_scale], [yn[i], yn[i] + wt[i] * math.sin(Phi) * arrow_scale])

    # Обновление стрелки углового ускорения
    WT_arrow_x = np.array([-wt[i] * 0.1 * arrow_scale, 0.0, -wt[i] * 0.1 * arrow_scale], dtype=float)
    WT_arrow_y = np.array([wt[i] * 0.05 * arrow_scale, 0.0, -wt[i] * 0.05 * arrow_scale], dtype=float)
    WT_arrow_rotx, WT_arrow_roty = rotate(WT_arrow_x, WT_arrow_y, Phi)
    WT_arrow.set_data(xn[i] + wt[i] * math.cos(Phi) * arrow_scale + WT_arrow_rotx, yn[i] + wt[i] * math.sin(Phi) * arrow_scale + WT_arrow_roty)

    # Обновление линии нормального ускорения
    WNLine.set_data([xn[i], xn[i] + wn[i] * math.cos(Phi - math.pi / 2) * arrow_scale], [yn[i], yn[i] + wn[i] * math.sin(Phi - math.pi / 2) * arrow_scale])

    # Обновление стрелки нормального ускорения
    WN_arrow_x = np.array([-wn[i] * 0.1 * arrow_scale, 0.0, -wn[i] * 0.1 * arrow_scale], dtype=float)
    WN_arrow_y = np.array([wn[i] * 0.05 * arrow_scale, 0.0, -wn[i] * 0.05 * arrow_scale], dtype=float)
    WN_arrow_rotx, WN_arrow_roty = rotate(WN_arrow_x, WN_arrow_y, Phi - math.pi / 2)
    WN_arrow.set_data(xn[i] + wn[i] * math.cos(Phi - math.pi / 2) * arrow_scale + WN_arrow_rotx, yn[i] + wn[i] * math.sin(Phi - math.pi / 2) * arrow_scale + WN_arrow_roty)

    # Обновление линии радиус-вектора
    RLine.set_data([0, xn[i]], [0, yn[i]])

    # Обновление стрелки радиус-вектора
    R_arrow_x = np.array([-0.1, 0.0, -0.1], dtype=float)
    R_arrow_y = np.array([0.05, 0.0, -0.05], dtype=float)
    R_arrow_rotx, R_arrow_roty = rotate(R_arrow_x, R_arrow_y, math.atan2(yn[i], xn[i]))
    R_arrow.set_data(xn[i] + R_arrow_rotx, yn[i] + R_arrow_roty)

    # Приближение графика
    ax.set_xlim(xn[i]-6, xn[i]+6)
    ax.set_ylim(yn[i]-6, yn[i]+6)

    return [P, VLine, V_arrow, WTLine, WT_arrow, WNLine, WN_arrow, RLine, R_arrow]

# Анимация
a = FuncAnimation(fig, cha, frames=len(T), interval=50)
plt.show()
