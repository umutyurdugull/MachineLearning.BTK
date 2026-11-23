import numpy as np
from scipy import stats
x = np.array([182, 175, 188, 190, 178, 180, 185, 179, 183, 177, 174]) # work duration
y = np.array([77,  69,  83,  85,  73,  75,  78,  74,  76,  70,  72])  # label count
xmean = np.mean(x)
ymean = np.mean(y)


numerator = np.sum((x - xmean) * (y - ymean))
denominator = np.sum((x - xmean) **2)

theta_1 = numerator / denominator
theta_0 = ymean  - (theta_1*xmean)

print(theta_1)
print(theta_0)


# Y_HAT = theata0 + theta1*x

xNew = 195

y_pred = theta_0 + theta_1*xNew
print(y_pred)

#SCIPY 

result = stats.linregress(x,y)
#        theta0           + theta1*xNew 
y_pred = result.intercept + result.slope*195
print(y_pred) # same result.