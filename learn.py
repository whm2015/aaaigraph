import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import transforms
from matplotlib.patches import Ellipse

import numpy as np
import matplotlib.pyplot as plt

ax = plt.subplot()

data = np.random.random((4, 4))
print(data)
plt.imshow(data)
# ax.imshow(data)
# # plt.setp(ax1.get_xticklabels(), visible=False)
# # plt.setp(ax1.get_yticklabels(), visible=False)
# ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
# ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
# ax.tick_params(axis='minor', which='minor', length=0)
#
#
plt.xticks(range(4), range(1, 4 + 1))
ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
ax.tick_params(axis='x', which='minor', length=0)
ax.set_xlabel(r'$\beta^{(i)^{\prime}}$')
plt.show()