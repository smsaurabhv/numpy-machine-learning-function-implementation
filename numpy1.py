#build a basic sigmoid functions


import math 
import numpy as np


def basic_sigmoid(value_to_entered):
	#math.exp for getting sigmoid values
	if(np.isscalar(value_to_entered)):
		print ("this is value :"+str(value_to_entered))
		print("exponent value of this"+str(math.exp(-1*value_to_entered)))
		getexponent =math.exp(-1*value_to_entered)
		print ("exponent value"+str(getexponent))
		sigmoid =(1/(1+getexponent))
		return sigmoid
	else:
		getarray =[]
		for i in value_to_entered:
			print("exponent value of this"+str(math.exp(-1*i)))
			getexponent =math.exp(-1*i)
			print ("exponent value"+str(getexponent))
			sigmoid =(1/(1+getexponent))
			getarray.append(sigmoid)
		return getarray	

def gradient_sigmoid(value_to_entered):
	#math.exp for getting sigmoid values
	if(np.isscalar(value_to_entered)):
		print ("this is value :"+str(value_to_entered))
		print("exponent value of this"+str(math.exp(-1*value_to_entered)))
		getexponent =math.exp(-1*value_to_entered)
		print ("exponent value"+str(getexponent))
		sigmoid =(1/(1+getexponent))
		gradsigmoid = sigmoid*(1-sigmoid)
		return gradsigmoid
	else:
		getarray =[]
		for i in value_to_entered:
			print("exponent value of this"+str(math.exp(-1*i)))
			getexponent =math.exp(-1*i)
			print ("exponent value"+str(getexponent))
			sigmoid =(1/(1+getexponent))
			gradsigmoid = sigmoid*(1-sigmoid)
			getarray.append(gradsigmoid)
		return getarray	

def image2vector(imagesornumpyarray):
	print("dimension of images or numpy array "+str(imagesornumpyarray.shape))
	height = imagesornumpyarray.shape[0]
	width =imagesornumpyarray.shape[1]
	depth=imagesornumpyarray.shape[2]
	vectorhape =height*width*depth
	imagesornumpyarray=imagesornumpyarray.reshape(vectorhape,1)
	return imagesornumpyarray

def normalizingrows(imagesornumpyarray):
	print(imagesornumpyarray)
	x_norm = np.linalg.norm(imagesornumpyarray,axis=1,keepdims=True)
	print("matrix normalization norm"+str(x_norm))
	imagesornumpyarray = imagesornumpyarray/x_norm
	return imagesornumpyarray


def softmax(inputarray):
	print(inputarray)
	inputarray = np.exp(inputarray)
	print("this is exponent of all array")
	print(inputarray)
	getsumofallexponent = np.sum(inputarray,axis=1,keepdims=True)
	inputarray = inputarray/getsumofallexponent
	return inputarray 
def L1_loss(predictedoutput,actualoutput):
	print("predicted output is"+str(predictedoutput))
	print("actual output is"+str(actualoutput))
	geterror = np.absolute(actualoutput-predictedoutput)
	totalloss = np.sum(geterror)
	return totalloss
def L2_loss(predictedoutput,actualoutput):
	print("predicted output is"+str(predictedoutput))
	print("actual output is"+str(actualoutput))
	geterror = np.absolute(actualoutput-predictedoutput)
	geterror = geterror**2
	totalloss = np.sum(geterror)
	return totalloss


print(basic_sigmoid(3))	


print("now calculation sigmoid fucntion by np array ")


randarray = np.array([1,2,3,4,5,6,7,8,9,10])

print("array is")
print(randarray)

print("exponent value of array is")
print(np.exp(randarray))

print("sigmoid value of array is")
print(basic_sigmoid(randarray))


print("now sigmoid gradient ")
print(gradient_sigmoid(randarray))


print("now images reshaping")
image = np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],

       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],

       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])
print("new shape of images or numpy array")
print(image2vector(image))
print("now normalizing rows ")
print("""
Another common technique we use in Machine Learning
 and Deep Learning is to normalize our data. It often leads to a
  better performance because gradient descent converges faster after normalization. 
  Here, by normalization we mean changing x to  x (dividing each row vector of x by its norm).
	""")
randarray=np.array([
    [0, 3, 4],
    [1, 6, 4]])
print("Array is")
print(randarray)
print("After normalization")
print(normalizingrows(randarray))
print("now softmax function")

randarray = [
    [9, 2, 5, 0, 0],
    [7, 5, 0, 0 ,0]]

print(softmax(randarray))    
print("now L1 loss")
yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print(L1_loss(yhat,y))
print("now L2 loss")
yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print(L2_loss(yhat,y))
