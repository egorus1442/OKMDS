import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp
import math

# Ввод переменной t и радиусов необходимых окружностей + ввод угла поворота шариков
t = sp.Symbol('t')
R = 2

# Построение графика и подграфика с выравниванием осей
fig = plt.figure(figsize=(17, 8)) # создание окна
ax1 = fig.add_subplot(1, 2, 1) # место для построения графиков
ax1.axis('equal') # задает равный масштаб по осям

phi = np.linspace(0, 2 * math.pi, 500)
psi = np.linspace(-math.pi/2, 0, 500)

conline, = ax1.plot([sp.sin(2*psi[0]) * R * (sp.cos(phi[0])), 0], [-R, R], 'black') # пружина
P, = ax1.plot(sp.sin(2*psi[0]) * R * (sp.cos(phi[0])), sp.cos(2*psi[0]) * R, marker='o', color='black') # Точка
Circ, = ax1.plot(R * (sp.cos(phi[0])) * np.cos(phi), R * np.sin(phi), 'black') # Построение окружности с радиусом  R и центром в начале 
# координат, используя параметрическое уравнение окружности x=Rcos(ϕ) и y=Rsin(ϕ).

#Доп графики
ax2 = fig.add_subplot(4, 2, 2)
T = np.linspace(0, 2 * math.pi, 1000)

# параметрическое уравнение для координат
x = sp.sin(t)+2
y = sp.cos(t+math.pi)+2

# производная координаты по t, что дает скорость по оси
Vx = sp.diff(x, t)
Vy = sp.diff(y,t)

T = np.linspace(0, 10, 1000)

# Инициализация массивов для хранения значений скоростей
VX = np.zeros_like(T)
VY = np.zeros_like(T)

# Вычисление значений скоростей для каждого значения T
for i in np.arange(len(T)):
    VX[i] = sp.Subs(Vx, t, T[i])
    VY[i] = sp.Subs(Vy, t, T[i])

# Построение графика скорости по оси X
ax2.plot(T, VX)
ax2.set_xlabel('T')
ax2.set_ylabel('VX')

# Построение графика скорости по оси Y
ax3 = fig.add_subplot(4, 2, 4)
ax3.plot(T, VY)
ax3.set_xlabel('T')
ax3.set_ylabel('VY')

def anima(i):
    P.set_data(sp.sin(2*psi[i]) * R * (sp.cos(phi[i])), sp.cos(2*psi[i]) * R)
    conline.set_data([sp.sin(2*psi[i]) * R * (sp.cos(phi[i])), 0], [sp.cos(2*psi[i]) * R, R])
    Circ.set_data(R * (sp.cos(phi[i])) * np.cos(phi), R * np.sin(phi))
    return Circ, P, conline

anim = FuncAnimation(fig, anima, frames=500, interval=1)
plt.show()