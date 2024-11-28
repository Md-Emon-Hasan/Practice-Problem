# import dependency
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris
import pickle

# load iris datasets
df = load_iris()

# target the data
x = df.data
y = df.target

# train the data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

# set algorithm
model = LinearRegression()

# fit the model
model.fit(x_train,y_train)

# save the train model into the file
pickle.dump(model,open('model.pkl','wb'))