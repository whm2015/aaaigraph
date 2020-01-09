import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0.0, 1.0 + 0.01, 0.01)
s = np.cos(2 * 2*np.pi * t)
t[41:60] = np.nan

plt.subplot(2, 1, 1)
plt.plot(t, s, '-', lw=2)

plt.xlabel('time (s)')
plt.ylabel('voltage (mV)')
plt.title('A sine wave with a gap of NaNs between 0.4 and 0.6')
plt.grid(True)

plt.subplot(2, 1, 2)
t[0] = np.nan
t[-1] = np.nan
plt.plot(t, s, '-', lw=2)
plt.title('Also with NaN in first and last point')

plt.xlabel('time (s)')
plt.ylabel('more nans')
plt.grid(True)

plt.subplots_adjust(left=0, bottom=0, right=0.5, top=0.5, wspace=0, hspace=0)
plt.show()

x = np.arange(3200, 35000, 3200)
num1 = [0.472, 0.469, 0.447, 0.433, 0.418, 0.418, 0.418, 0.418, 0.418, 0.418]
num2 = [0.337, 0.327, 0.325, 0.316, 0.312, 0.311, 0.308, 0.305, 0.295, 0.290]
y1 = np.array(num1)
y2 = np.array(num2)

# 用3次多项式拟合
f1 = np.polyfit(x, y1, 3)
p1 = np.poly1d(f1)
print(p1)  # 打印出拟合函数
yvals1 = p1(x)  # 拟合y值

# 绘图
plot1 = plt.plot(x, y1, 's', label='original values')
plot2 = plt.plot(x, yvals1, 'r', label='polyfit values')

plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)
plt.title('polyfitting')
plt.savefig('nihe1.png')
plt.show()