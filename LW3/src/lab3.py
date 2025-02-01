import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint

def SystDiffEq(y, t, m1, m2, m3, r, R, g, M1, M2, c):
    # y = [ phi, theta, phi', theta'] -> dy = [ phi', theta', phi'', theta'']
    dy = np.zeros_like(y)
    dy[0] = y[2]
    dy[1] = y[3]
    # a11 * phi'' + a12 * theta'' = b1
    # a21 * phi'' + a22 * theta'' = b2
    # коэффициенты первого уравнения
    a11 = (3*m2 + (2/3)*m3) * (R-r)**2
    a12 = (R-r) * m2 * R
    b1 = (2*m2 + m3) * g * (R-r) * np.sin(y[0]) - 2*c*y[0] + 2*M2
    # коэффициенты второго уравнения
    a21 = m2 * R * (R-r)
    a22 = (m1 + m2) * R**2
    b2 = 2 * M1

    # решение правилом Крамера
    detA = a11 * a22 - a12 * a21
    detA1 = b1 * a22 - a12 * b2
    detA2 = a11 * b2 - b1 * a21

    dy[2] = detA1 / detA
    dy[3] = detA2 / detA

    return dy

# задается интервал
step = 1000
t = np.linspace(0, 10, step)

# начальные значения
O = 0  # начало координат
r2 = 0.2  # внешний радиус пружины
r1 = 0.05  # внутренний радиус пружины
m1 = 2
m2 = 1
m3 = 1
c = 2
M1 = 0.5
M2 = 0.2
g = 9.8

r = 0.2  # r маленького колеса
R = 1  # R большого колеса

# задаем начальные значения
y0 = [np.pi/6, np.pi/2, 0, 2]
# заполняем массив производными
Y = odeint(SystDiffEq, y0, t, (m1, m2, m3, r, R, g, M1, M2, c))

# заполняем функции
phi = Y[:, 0]
theta = Y[:, 1]
phit = Y[:, 2]
thetat = Y[:, 3]

phitt = np.zeros_like(t)
thetatt = np.zeros_like(t)
NAx = np.zeros_like(t)
NAy = np.zeros_like(t)
for i in range(len(t)):
    phitt[i] = SystDiffEq(Y[i], t[i], m1, m2, m3, r, R, g, M1, M2, c)[2]
    thetatt[i] = SystDiffEq(Y[i], t[i], m1, m2, m3, r, R, g, M1, M2, c)[3]
    NAx[i] = m2 * (R-r) * (phitt[i] * np.cos(phi[i]) - phit[i] * np.sin(phi[i]))
    NAy[i] = m2 * (-(R-r) * (phitt[i] * np.sin(phi[i]) + phit[i] * np.cos(phi[i]) + g))

# выводим графики зависимости
fgrt = plt.figure()

phiplt = fgrt.add_subplot(2, 2, 1)
phiplt.plot(t, phi, color='red')
phiplt.set_title('Phi(t)')

thetaplt = fgrt.add_subplot(2, 2, 2)
thetaplt.plot(t, theta, color='red')
thetaplt.set_title('Theta(t)')

NAxplt = fgrt.add_subplot(2, 2, 3)
NAxplt.plot(t, NAx, color='orange')
NAxplt.set_title('NAx(t)')

NAyplt = fgrt.add_subplot(2, 2, 4)
NAyplt.plot(t, NAy, color='orange')
NAyplt.set_title('NAy(t)')

fgrt.show()

# координаты точки А
Xa = (R-r) * np.cos(phi)
Ya = (R-r) * np.sin(phi)
# координаты точки Б
Xb = R * np.cos(np.pi/2 + theta)
Yb = R * np.sin(np.pi/2 + theta)

# окно и график
fgr = plt.figure()
grf = fgr.add_subplot(1, 1, 1)
grf.axis('equal')
grf.set(xlim=[-3, 3], ylim=[-3, 3])
grf.set_aspect(1)

# треугольник
grf.plot([0, 0.1, -0.1, 0], [0, -0.2, -0.2, 0], color='black')

# точки на графике
p1 = grf.plot(O, O, marker='o', color='black')[0]
pA = grf.plot(Xa[0], Ya[0], marker='o', color='black')[0]
pB = grf.plot(Xb[0], Yb[0], marker='o', color='black')[0]

# прямая ОА
O1 = grf.plot([Xa[0], O], [Ya[0], O], color='black')[0]

# большая окружность
circle = plt.Circle((O, O), R, fill=False)
grf.add_artist(circle)

# маленькая окружность
circleA = plt.Circle((Xa[0], Ya[0]), r, fill=False)
grf.add_artist(circleA)

# спиральная пружина
Ns = 2

numpnts = np.linspace(0, 1, 50*Ns+1)
Betas = numpnts * (Ns * 2 * np.pi + phi[0])
Xs = ((r2-r1) * numpnts) * np.cos(Betas + np.pi/2)
Ys = ((r2-r1) * numpnts) * np.sin(Betas + np.pi/2)

SpPruzh = grf.plot(Xs, Ys, color='black')[0]

# анимация
def run(i):
    pA.set_data(Xa[i], Ya[i])
    pB.set_data(Xb[i], Yb[i])
    O1.set_data([Xa[i], O], [Ya[i], O])
    circleA.center = (Xa[i], Ya[i])

    Betas = numpnts * (Ns * 2 * np.pi - phi[i])
    Xs = -((r2-r1) * numpnts) * np.cos(Betas + np.pi)
    Ys = ((r2-r1) * numpnts) * np.sin(Betas + np.pi)
    SpPruzh.set_data(Xs, Ys)
    return pA, pB, O1, circleA, SpPruzh

anim = FuncAnimation(fgr, run, interval=1, frames=step)

plt.show()
