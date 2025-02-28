import numpy as np
import sympy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sympy.utilities.lambdify import implemented_function


def r(t):
    # Функция радиус-вектора материальной точки от времени
    return 2 + sympy.cos(6 * t)


def phi(t):
    # Функция угла материальной точки от времени
    return 7 * t + 1.2 * sympy.cos(6 * t)

def Rot2D(X, Y, phi):
    # Поворот двумерной ДСК с помощью матрицы поворота
    X_r = X * np.cos(phi) - Y * np.sin(phi)
    Y_r = X * np.sin(phi) + Y * np.cos(phi)
    return X_r, Y_r


def main():
    # Переменная времени t имеет символьный sympy тип
    t = sympy.Symbol("t")

    # Переход от ДСК к ПСК
    x = r(t) * sympy.cos(phi(t))
    y = r(t) * sympy.sin(phi(t))

    # Определение скорости и ускорения как производных координаты
    Vx = sympy.diff(x, t)
    Vy = sympy.diff(y, t)
    Wx = sympy.diff(Vx, t)
    Wy = sympy.diff(Vy, t)

    # Создание функций физических величин, зависящих от времени
    # Выражение в числовую функцию, которую можно использовать для вычислений с числовыми значениями. 
    F_x = sympy.lambdify(t, x, "numpy")
    F_y = sympy.lambdify(t, y, "numpy")
    F_Vx = sympy.lambdify(t, Vx, "numpy")
    F_Vy = sympy.lambdify(t, Vy, "numpy")
    F_Wx = sympy.lambdify(t, Wx, "numpy")
    F_Wy = sympy.lambdify(t, Wy, "numpy")

    # Указание временного начала и конца моделирования
    time_steps_amount = 1000 
    T = np.linspace(0, 4 * np.pi, time_steps_amount)

    # Указание значения физических величин в каждый момент времени  
    X = F_x(T) # Координата по оси OX
    Y = F_y(T) # Координата по оси OY
    VX = F_Vx(T) # Скорость по оси OX
    VY = F_Vy(T) # Скорость по оси OY
    V_phi = np.arctan2(VY, VX) # Направление (угол) вектора скорости
    WX = F_Wx(T) # Ускорение по оси OX
    WY = F_Wy(T) # Ускорение по оси OY
    W_phi = np.arctan2(WY, WX) # Направление (угол) вектора ускорения
    
    # Вектор нормального ускорения перпендикулярен вектору линейной скорости и лежит на радиусе кривизны
    P_phi = V_phi + np.pi/2
    
    # Масштаб вектора V (если он нерепрезентативно мал или велик)
    V_scale = 1/10

    # Масштаб вектора W (если он нерепрезентативно мал или велик)
    W_scale = 1/10
    
    # Масштаб отрезка P (если он нерепрезентативно мал или велик)
    P_scale = 1/10

    # Создание окна для графика
    fig = plt.figure()
    # Создание одной ячейки для отрисовки графика
    ax1 = fig.add_subplot(1, 1, 1)
    # Указываем единый масштаб по осям
    ax1.axis("equal")
    # фиксируем пределы окна по осям
    ax1.set(
        xlim=[-3, 3],
        ylim=[-3, 3],
        title=f"\
            Радиyс-вектор точки - серый вектор (масштаб 1.0)\n\
            Скорость (V) - зеленый вектор (масштаб {V_scale})\n\
            Ускорение (W) - красный вектор (масштаб {W_scale})",
    )
    ax1.plot(X, Y)

    # Добавление материальной точки
    (point,) = ax1.plot(X[0], Y[0], marker="o")
    
    # Добавление радиyс-вектора материальной точки
    R_color = [0.5, 0.5, 0.5] # Серый цвет
    (R_line,) = ax1.plot(
        [0, X[0]], 
        [0, Y[0]], 
        color=R_color,
    )

    # Добавление линии вектора скорости V
    V_color = [0, 0.7, 0] # Зеленый цвет
    (V_line,) = ax1.plot(
        [X[0], X[0] + VX[0] * V_scale],
        [Y[0], Y[0] + VY[0] * V_scale],
        color=V_color,
    )

    # Добавление линии вектора yскорения W
    W_color = [0.7, 0, 0] # Красный цвет
    (W_line,) = ax1.plot(
        [X[0], X[0] + WX[0] * W_scale],
        [Y[0], Y[0] + WY[0] * W_scale],
        color=W_color,
    )
    
    # Добавление стрелки вектора V
    X_arr_V = np.multiply(V_scale, np.array([-0.7, 0, -0.7])) # X координаты трех точек стрелки вектора V
    Y_arr_V = np.multiply(V_scale, np.array([0.2, 0, -0.2])) # Y координаты трех точек стрелки вектора V
    RX_V, RY_V = Rot2D(X_arr_V, Y_arr_V, V_phi[0]) # Задание начального направления вектора V
    (V_arrow,) = ax1.plot(RX_V, RY_V, color=V_color)

    # Добавление стрелки вектора W
    X_arr_W = np.multiply(W_scale, np.array([-0.7, 0, -0.7])) # X координаты трех точек стрелки вектора W
    Y_arr_W = np.multiply(W_scale, np.array([0.2, 0, -0.2])) # Y координаты трех точек стрелки вектора W
    RX_W, RY_W = Rot2D(X_arr_W, Y_arr_W, W_phi[0]) # Задание начального направления вектора W
    (W_arrow,) = ax1.plot(RX_W, RY_W, color=W_color)

    def animate(i):
        # Анимация материальной точки
        point.set_data(X[i], Y[i]) # Смена положения материальной точки
        
        # Анимация радиyс-вектора материальной точки
        R_line.set_data([0, X[i]], [0, Y[i]])

        # Анимация вектора V
        V_line.set_data([X[i], X[i] + VX[i] * V_scale], [Y[i], Y[i] + VY[i] * V_scale]) # Смена положения линии вектора V
        RX_V, RY_V = Rot2D(X_arr_V, Y_arr_V, V_phi[i])  # Смена направления вектора V
        V_arrow.set_data(X[i] + VX[i] * V_scale + RX_V, Y[i] + VY[i] * V_scale + RY_V) # Смена положения стрелки вектора V

        # Анимация вектора W
        W_line.set_data([X[i], X[i] + WX[i] * W_scale], [Y[i], Y[i] + WY[i] * W_scale]) # Смена положения линии вектора W
        RX_W, RY_W = Rot2D(X_arr_W, Y_arr_W, W_phi[i])  # Смена направления вектора W
        W_arrow.set_data(X[i] + WX[i] * W_scale + RX_W, Y[i] + WY[i] * W_scale + RY_W) # Смена положения стрелки вектора W
        
        return point, R_line, V_line, V_arrow, W_line, W_arrow

    animation = FuncAnimation(fig, animate, frames=time_steps_amount, interval=100)

    plt.show()


if __name__ == "__main__":
    main()

