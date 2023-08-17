import cv2
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
import codecs

weights = np.zeros(1024) # w1 = 0, w2 = 1
bias = 3                   
s = 0.15

class Neuron:
  def __init__(self, weights, bias):
    self.weights = weights
    self.bias = bias

  def feedforward(self, inputs):
    # Умножаем входы на веса, прибавляем порог, затем используем функцию активации
    return np.dot(self.weights, inputs) > self.bias

names = ['cry', 'cat', 'smile', 'cringe', 'dead', 'love', 'message', 'unhappy', 'suspicious', 'upvote']

cry = Neuron(weights, bias)
cry.weights = np.random.normal(0, 0.1, size=1024)
cat = Neuron(weights, bias)
cat.weights = np.random.normal(0, 0.1, size=1024)
smile = Neuron(weights, bias)
smile.weights = np.random.normal(0, 0.1, size=1024)
cringe = Neuron(weights, bias)
cringe.weights = np.random.normal(0, 0.1, size=1024)
dead = Neuron(weights, bias)
dead.weights = np.random.normal(0, 0.1, size=1024)
love = Neuron(weights, bias)
love.weights = np.random.normal(0, 0.1, size=1024)
message = Neuron(weights, bias)
message.weights = np.random.normal(0, 0.1, size=1024)
unhappy = Neuron(weights, bias)
unhappy.weights = np.random.normal(0, 0.1, size=1024)
suspicious = Neuron(weights, bias)
suspicious.weights = np.random.normal(0, 0.1, size=1024)
upvote = Neuron(weights, bias)
upvote.weights = np.random.normal(0, 0.1, size=1024)
neurons = []
neurons.append(cry)
neurons.append(cat)
neurons.append(smile)
neurons.append(cringe)
neurons.append(dead)
neurons.append(love)
neurons.append(message)
neurons.append(unhappy)
neurons.append(suspicious)
neurons.append(upvote)
print(neurons)
for i in range(10):
	np.set_printoptions(threshold=np.inf)
	print(neurons[i].weights)


for epoch in tqdm(range(70)):
	df = pd.read_csv("ds.csv", sep=";")
	df = df.sample(frac=1).reset_index(drop=True)

	for i in range(200): #The Loop
		a = np.fromstring(df.at[i, 'value'][1:-1],sep=', ').astype(int)
		a[a == 255] = 1
		tmp = np.zeros(10)

		for j in range(10):
			if(neurons[j].feedforward(a) == 1):
				if(names[j] != df.at[i, 'name']):
					neurons[j].weights - s
			elif(names[j] == df.at[i, 'name']):
				neurons[j].weights + s

df = pd.read_csv("result.csv", sep=",")

for i in range(10):
	df.at[i, 'value'] = str(neurons[i].weights)

df.to_csv('result.csv')
f = codecs.open('result.csv',encoding='utf-8')
contents = f.read()
f.close()
newcontents = contents.replace('\n ',' ')
newcontents = contents.replace('  ', ' ')
with open('result.scv', 'wt', encoding='utf-8') as file:	
	file.write(newcontents)

'''
for i in range(10):
	np.set_printoptions(threshold=np.inf)
	with open('result', 'a') as file:	
		file.write(str(neurons[i].weights)+'\n')
'''
#[1:-1],sep=' ').astype(float) 
'''
	if(cry.feedforward(a)):
		tmp[0] = 1
	if(neurons[0].feedforward(a)):
		tmp[1] = 1
	if(smile.feedforward(a)):
		tmp[2] = 1
	if(cringe.feedforward(a)):
		tmp[3] = 1
	if(dead.feedforward(a)):
		tmp[4] = 1
	if(love.feedforward(a)):
		tmp[5] = 1
	if(message.feedforward(a)):
		tmp[6] = 1
	if(unhappy.feedforward(a)):
		tmp[7] = 1
	if(suspicious.feedforward(a)):
		tmp[8] = 1
	if(upvote.feedforward(a)):
		tmp[9] = 1
'''

'''
df = pd.read_csv("ds.csv", sep=";")
a = np.fromstring(df.at[0, 'value'][1:-1],sep=', ').astype(int)
a = np.resize(a,(32,32))
cv2.imwrite('img.png', a)
im_gray = cv2.imread('img.png', cv2.IMREAD_GRAYSCALE)
thresh = 200
img_binary = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
cv2.imwrite('cry.png', img_binary)
print(img_binary)

a = np.resize(img_binary,1024)
np.set_printoptions(threshold=np.inf)
df.loc[0, 'value'] = a 
df.to_csv('ds_new.csv')
'''
#df = pd.read_csv("ds_new.csv", sep=";")
#a = np.fromstring(df.at[0, 'value'][1:-1],sep=' ').astype(int)
#a = np.resize(a,(32,32))
#cv2.imwrite('img.png', a)





#a = np.array([255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 254, 254, 254, 254, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 254, 254, 255, 253, 251, 252, 254, 254, 254, 254, 253, 251, 252, 254, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 254, 254, 255, 252, 252, 255, 254, 254, 255, 248, 246, 253, 255, 254, 255, 254, 251, 254, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 254, 255, 254, 252, 255, 255, 222, 170, 138, 130, 132, 133, 136, 133, 154, 196, 247, 255, 253, 252, 255, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255, 254, 255, 253, 253, 255, 197, 127, 142, 186, 224, 243, 253, 255, 249, 234, 203, 157, 125, 160, 244, 255, 252, 254, 255, 254, 255, 255, 255, 255, 255, 255, 254, 255, 252, 255, 237, 129, 141, 231, 255, 254, 254, 255, 254, 254, 254, 255, 254, 255, 250, 180, 112, 195, 255, 251, 254, 255, 254, 255, 255, 255, 255, 254, 255, 252, 255, 214, 107, 205, 255, 253, 252, 252, 254, 254, 254, 254, 254, 254, 252, 250, 253, 255, 244, 128, 157, 255, 252, 255, 254, 255, 255, 255, 254, 255, 253, 255, 216, 108, 233, 255, 252, 253, 255, 254, 255, 255, 255, 255, 255, 255, 254, 255, 254, 252, 252, 255, 146, 151, 255, 252, 255, 254, 255, 255, 254, 254, 255, 241, 106, 228, 255, 240, 234, 255, 254, 254, 255, 255, 255, 255, 253, 255, 214, 71, 244, 255, 253, 251, 255, 131, 177, 255, 252, 254, 255, 255, 254, 252, 255, 136, 186, 255, 255, 171, 57, 255, 254, 254, 255, 255, 255, 255, 254, 255, 232, 97, 250, 255, 254, 254, 254, 247, 111, 234, 255, 254, 254, 254, 253, 255, 215, 127, 255, 251, 255, 238, 122, 255, 254, 254, 255, 255, 255, 255, 254, 254, 255, 133, 246, 255, 254, 255, 252, 255, 186, 145, 255, 252, 254, 254, 251, 255, 144, 197, 255, 252, 255, 246, 132, 255, 254, 254, 255, 255, 255, 255, 254, 253, 255, 136, 237, 255, 254, 254, 254, 254, 254, 121, 240, 255, 254, 254, 255, 249, 126, 249, 255, 254, 255, 252, 133, 251, 255, 254, 255, 255, 255, 255, 254, 252, 255, 138, 233, 255, 254, 255, 254, 252, 255, 162, 190, 255, 252, 253, 255, 218, 141, 255, 252, 254, 254, 255, 132, 246, 255, 254, 255, 255, 255, 255, 255, 252, 255, 146, 221, 255, 253, 255, 254, 253, 255, 202, 156, 255, 251, 252, 255, 200, 161, 255, 251, 254, 253, 255, 137, 237, 255, 254, 255, 255, 255, 255, 255, 251, 255, 148, 214, 255, 253, 255, 255, 254, 255, 224, 141, 255, 252, 252, 255, 186, 170, 255, 251, 254, 252, 255, 138, 233, 255, 254, 255, 255, 255, 255, 255, 252, 255, 172, 211, 255, 253, 255, 255, 254, 255, 233, 136, 255, 252, 252, 255, 196, 161, 255, 251, 254, 252, 255, 144, 220, 255, 253, 254, 253, 252, 251, 251, 252, 253, 253, 254, 255, 254, 255, 255, 254, 255, 227, 142, 255, 252, 253, 255, 216, 143, 255, 252, 254, 254, 255, 224, 244, 255, 251, 254, 255, 254, 255, 255, 254, 255, 254, 251, 253, 254, 255, 254, 253, 255, 205, 154, 255, 251, 254, 255, 246, 128, 252, 255, 254, 255, 253, 252, 254, 254, 255, 251, 220, 188, 170, 170, 188, 218, 250, 255, 255, 253, 251, 254, 252, 255, 167, 186, 255, 252, 254, 252, 255, 141, 205, 255, 253, 254, 254, 255, 251, 203, 151, 129, 142, 170, 189, 191, 173, 144, 130, 154, 214, 255, 254, 254, 253, 255, 125, 235, 255, 254, 254, 253, 255, 207, 133, 255, 251, 254, 217, 155, 121, 141, 195, 240, 255, 255, 252, 248, 250, 248, 227, 179, 116, 87, 151, 226, 255, 195, 135, 255, 252, 254, 255, 254, 252, 255, 129, 196, 255, 164, 55, 73, 128, 149, 144, 133, 129, 128, 131, 134, 133, 133, 138, 148, 157, 132, 107, 158, 255, 115, 225, 255, 253, 254, 255, 254, 254, 255, 230, 107, 241, 253, 252, 254, 252, 253, 254, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 254, 252, 255, 146, 164, 255, 252, 254, 255, 255, 254, 255, 252, 255, 201, 113, 246, 255, 251, 254, 254, 254, 254, 254, 254, 253, 252, 252, 252, 252, 253, 249, 249, 255, 159, 138, 255, 252, 255, 254, 255, 255, 255, 254, 255, 252, 255, 200, 108, 221, 255, 253, 251, 252, 254, 254, 254, 254, 254, 254, 253, 251, 252, 255, 253, 143, 142, 255, 253, 254, 254, 255, 255, 255, 255, 255, 254, 255, 252, 255, 224, 118, 157, 243, 255, 254, 255, 254, 253, 252, 254, 255, 254, 254, 255, 198, 115, 177, 255, 252, 254, 255, 254, 255, 255, 255, 255, 255, 255, 254, 255, 252, 254, 253, 177, 121, 149, 202, 239, 254, 255, 254, 255, 248, 220, 171, 124, 142, 232, 255, 251, 254, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 254, 255, 253, 252, 255, 250, 199, 152, 134, 133, 137, 136, 136, 131, 143, 176, 235, 255, 253, 252, 255, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 254, 255, 254, 252, 253, 255, 254, 255, 248, 235, 232, 242, 253, 255, 254, 254, 252, 253, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 254, 255, 254, 252, 251, 253, 255, 254, 254, 255, 254, 252, 252, 254, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 254, 254, 254, 254, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255])
#a = np.resize(a,(32,32))
#cv2.imwrite('img.png', a)
#im_gray = cv2.imread('img.png', cv2.IMREAD_GRAYSCALE)
#thresh = 200
#img_binary = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
#cv2.imwrite('cry.png', img_binary)
