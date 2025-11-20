import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from sklearn.model_selection import train_test_split




def load_letters(filename):
    with open(filename, 'r') as file:
        letters_raw = json.load(file)
    letters = {k: np.array(v).flatten() for k, v in letters_raw.items()}
    return letters

def load_labels(filename):
    with open(filename, 'r') as file:
        labels_raw = json.load(file)
    labels = {k: np.array(v) for k, v in labels_raw.items()}
    return labels
 



def turn90(matrix,n=1):
    rotated_matrix = np.rot90(matrix,n)
    return rotated_matrix


def turn_matrix(matrix,angle=90):
    rotated_matrix = rotate(matrix,angle, reshape=True)
    center_x, center_y = rotated_matrix.shape[0] // 2, rotated_matrix.shape[1] // 2
    start_x = center_x - 4
    start_y = center_y - 4
    final_matrix = rotated_matrix[start_x:start_x+8, start_y:start_y+8]
    return final_matrix



def rout_90(x,dim=8):
    x=np.array(X[0]).reshape(8,8)
    x=np.rot90(x)
    x=x.reshape(8**2)
    return x

def normalize_data(data):
    return (data + 1) / 2  

def normalize_data_list(data):
    normal_list=np.zeros(len(data))
    nom=[]
    for i in range(len(data)):
        nom.append(np.array((data[i] + 1) / 2 ))
    return nom


def modify_random_elements(arr, percentage):
    arr = arr.copy()  # جلوگیری از تغییر مستقیم در داده‌های ورودی
    total_elements = arr.size
    num_elements_to_modify = int(total_elements * percentage / 100)
    indices = np.random.choice(total_elements, num_elements_to_modify, replace=False)
    for index in indices:
        row, col = divmod(index, arr.shape[1])
        arr[row, col] =0
    return arr


def nose_train_list(arr, percentage):
    arr = arr.copy()  # جلوگیری از تغییر مستقیم در داده‌های ورودی
    total_elements = len(arr)
    num_elements_to_modify = int(total_elements * percentage / 100)
    for i in range(num_elements_to_modify):
        indx=np.random.randint(0,total_elements)
        arr[indx]=rout_90(np.array([arr[indx]]),10)
    return arr



def f_score_perectorn(y,y_pred):
    f_count=0
    _len=len(y_pred)
    for idx in range(len(y)):
        if int(y[idx])==int(y_pred[idx]):
            f_count+=1
    return f_count/_len

def f_score_deltaRule(y,yperd):
    n_count=0
    for i in range(len(y)):
        if list(y[i])==list(yperd[i]):
            n_count+=1   
    return n_count/len(y)
    
    
    

# 2. تابع فعال‌سازی برای پرسپترون (Step Function)
def step_function(x):
    return np.where(x >= 0, 1, -1)

# 3. تابع فعال‌سازی برای قانون دلتا (Sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)


letters = load_letters("D:\pythonprojects\letters.json")
labels = load_labels("D:\pythonprojects\labels.json")
 
# 4. کلاس پرسپترون
class Perceptron:
    def __init__(self, input_size, lr=0.01, epochs=10000):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.lr = lr
        self.epochs = epochs
        self.eror=None
    def train(self, X, y):
        self.eror=[]
        for _ in range(self.epochs):
            total_eror=0
            for xi, target in zip(X, y):
                output = np.dot(xi, self.weights)+ self.bias
                error = target - output
                self.weights += self.lr * error * xi
                self.bias += self.lr * error
                total_eror += np.abs(error).sum()
            self.eror.append(total_eror)
    
    def predict(self, X):
        pred_matrix=np.zeros(len(X))
        for idex in  range(len(X)):
            pred_matrix[idex]=np.round(np.dot(X[idex], self.weights) + self.bias)
        return pred_matrix


class DeltaRule:
    def __init__(self, input_size, output_size, lr=0.01, epochs=1000):
        # self.weights = np.random.randn(input_size, output_size)
        self.weights=np.array([np.zeros(output_size) for i in range(input_size)])
        self.bias = np.random.randn()
        self.lr = lr
        self.epochs = epochs
        self.erors=None
    def train(self, X, y): 
        self.erors=[]
        for _ in range(self.epochs):
            total_error = 0
            for xi, target in zip(X, y):
                output = sigmoid(np.dot(xi, self.weights) + self.bias) 
                error = target - output 
                self.weights += self.lr * np.outer(xi, error * sigmoid_derivative(output)) 
                self.bias += self.lr * error * sigmoid_derivative(output)
                total_error += np.abs(error).sum()
            self.erors.append(total_error)

 
    def predict(self,X):
        pred_list=[]
        for i in range(len(X)):
            x=np.round(sigmoid(np.dot(X[i], self.weights) + self.bias))
            pred_list.append(list(x))  
        return pred_list
 # 6. اجرای آموزش مدل‌ها




X = np.array(list(letters.values()))
y = np.array(list(labels.values()))
 
perceptron = Perceptron(input_size=64)
deltarule=DeltaRule(64,12)

perceptron.train(X, y.argmax(axis=1))  # دسته‌بندی پرسپترون فقط برچسب‌های عددی می‌پذیرد
deltarule.train(X, y)  # قانون دلتا می‌تواند چندکلاسه را یاد بگیرد

print("Training finished!")


plt.plot(deltarule.erors, 'r-x')
plt.xlabel("Number of Iteration")
plt.ylabel("Total error")
plt.title("Delta Rule Training Error")
plt.show()

X_clean=np.array(list(letters.values()))
X_noisy = np.array([modify_random_elements(x.reshape(8, 8), percentage=5).flatten() for x in X])
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#comment
X_noisy_30 = np.array([modify_random_elements(x.reshape(8, 8), percentage=30).flatten() for x in X])
X_noisy_50 = np.array([modify_random_elements(x.reshape(8, 8), percentage=50).flatten() for x in X])
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

print(f"Perceptron test (10% nose): {perceptron.predict(modify_random_elements(X_noisy,10))}")
print(f"Perceptron test (clean): {perceptron.predict(X_clean)}")
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #comment
print(f"Perceptron test (30% nose): {perceptron.predict(modify_random_elements(X_noisy_30,30))}")
print(f"Perceptron test (50% nose): {perceptron.predict(modify_random_elements(X_noisy_50,50))}")
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 
y_deta_pred=deltarule.predict(X)
y_perceptron_pred=perceptron.predict(np.array(X))
y_deta_pred_noisy=deltarule.predict(nose_train_list(X,10))
y_perceptron_pred_noisy=perceptron.predict(nose_train_list(X,10))
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#comment
y_deta_pred_noisy_30=deltarule.predict(nose_train_list(X,30))
y_perceptron_pred_noisy_30=perceptron.predict(nose_train_list(X,30))
y_deta_pred_noisy_50=deltarule.predict(nose_train_list(X,50))
y_perceptron_pred_noisy_50=perceptron.predict(nose_train_list(X,50))
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 
delta_acc_clean =f_score_deltaRule(y,y_deta_pred)
perceptron_acc_clean = f_score_perectorn(y.argmax(axis=1),y_perceptron_pred)
delta_acc_noisy= f_score_deltaRule(y,y_deta_pred_noisy)
perceptron_acc_noisy = f_score_perectorn(y.argmax(axis=1),y_perceptron_pred_noisy)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#comment
delta_acc_noisy_30=f_score_deltaRule(y,y_deta_pred_noisy_30)
perceptron_acc_noisy_30 = f_score_perectorn(y.argmax(axis=1),y_perceptron_pred_noisy_30)
delta_acc_noisy_50=f_score_deltaRule(y,y_deta_pred_noisy_50)
perceptron_acc_noisy_50 = f_score_perectorn(y.argmax(axis=1),y_perceptron_pred_noisy_50)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

 




print(f"Perceptron Training Accuracy (clean): {perceptron_acc_clean*100:.2f}%")
print(f"Delta Rule Training Accuracy (clean): {delta_acc_clean*100:.2f}%")
print(f"Perceptron Training Accuracy (noisy, 10%): {perceptron_acc_noisy*100:.2f}%")
print(f"Delta Rule Training Accuracy (noisy, 10%): {delta_acc_noisy*100:.2f}%")
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#comment
print(f"Perceptron Training Accuracy (noisy, 30%): {perceptron_acc_noisy_30*100:.2f}%")
print(f"Delta Rule Training Accuracy (noisy, 30%): {delta_acc_noisy_30*100:.2f}%")
print(f"Perceptron Training Accuracy (noisy, 50%): {perceptron_acc_noisy_50*100:.2f}%")
print(f"Delta Rule Training Accuracy (noisy, 50%): {delta_acc_noisy_50*100:.2f}%")
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 
 


X_=list(X)
Y_=list(y)
for epoc in range(4):
    for i in range(len(X)):
                X_.append(turn_matrix(np.array(X[i]).reshape(8,8),np.random.randint(0,90)))
                Y_.append(y[i])
for percent in range(10,40,10):
        for j in range(len(X)):
            X_.append(modify_random_elements(X[j].reshape(8,8),percent))
            Y_.append(y[j])
for k in range(len(X)):
        X_.append(list(turn90(X[k].reshape(8,8)).reshape(1,64)[0]))
        Y_.append(y[k])
  
 

  
x_trian,x_test,y_trian,y_test=train_test_split(X_,Y_,test_size=0.1,random_state=123)


perceptron_trian =Perceptron(input_size=64)
deltarule_trian=DeltaRule(64,12)

perceptron_trian.train([np.array(x).reshape(1,64)[0] for x in x_trian], np.array(y_trian).argmax(axis=1))  # دسته‌بندی پرسپترون فقط برچسب‌های عددی می‌پذیرد
y_perceptron_pred_train=perceptron_trian.predict([np.array(x).reshape(1,64)[0] for x in x_trian])
y_perceptron_pred_test=perceptron_trian.predict([np.array(x).reshape(1,64)[0] for x in x_test])
perceptron_acc_train= f_score_perectorn(np.array(y_trian).argmax(axis=1),np.array(y_perceptron_pred_train))
perceptron_acc_test= f_score_perectorn(np.array(y_test).argmax(axis=1),np.array(y_perceptron_pred_test))


deltarule_trian.train([np.array(x).reshape(1,64)[0] for x in x_trian], y_trian ) # قانون دلتا می‌تواند چندکلاسه را یاد بگیرد
y_deltarule_pred_train=deltarule_trian.predict(np.array(list([np.array(x).reshape(1,64)[0] for x in x_trian])))
y_deltarule_pred_test=deltarule_trian.predict(np.array(list([np.array(x).reshape(1,64)[0] for x in x_test])))
deltarule_acc_train= f_score_deltaRule(y_trian,y_deltarule_pred_train)
deltarule_acc_test= f_score_deltaRule(y_test,y_deltarule_pred_test)





print(f"deltarule_acc_train  ===> {deltarule_acc_train*100}%")
print(f"deltarule_acc_test   ===> {deltarule_acc_test*100}%")
print(f"perceptron_acc_train ===> {perceptron_acc_train*100}%")
print(f"perceptron_acc_test  ===> {perceptron_acc_test*100}%")









# import numpy as np
# import json
# import matplotlib.pyplot as plt
# from scipy.ndimage import rotate
# from sklearn.model_selection import train_test_split




# def load_letters(filename):
#     with open(filename, 'r') as file:
#         letters_raw = json.load(file)
#     letters = {k: np.array(v).flatten() for k, v in letters_raw.items()}
#     return letters

# def load_labels(filename):
#     with open(filename, 'r') as file:
#         labels_raw = json.load(file)
#     labels = {k: np.array(v) for k, v in labels_raw.items()}
#     return labels
 



# def turn90(matrix,n=1):
#     rotated_matrix = np.rot90(matrix,n)
#     return rotated_matrix


# def turn_matrix(matrix,angle=90):
#     rotated_matrix = rotate(matrix,angle, reshape=True)
#     center_x, center_y = rotated_matrix.shape[0] // 2, rotated_matrix.shape[1] // 2
#     start_x = center_x - 4
#     start_y = center_y - 4
#     final_matrix = rotated_matrix[start_x:start_x+8, start_y:start_y+8]
#     return final_matrix



# def rout_90(x,dim=8):
#     x=np.array(X[0]).reshape(8,8)
#     x=np.rot90(x)
#     x=x.reshape(8**2)
#     return x

# def normalize_data(data):
#     return (data + 1) / 2  

# def normalize_data_list(data):
#     normal_list=np.zeros(len(data))
#     nom=[]
#     for i in range(len(data)):
#         nom.append(np.array((data[i] + 1) / 2 ))
#     return nom

# # def f_score(y,y_pred):
# #     count=0
# #     for i in range(len(y)):
# #         if np.array(y[i]).all()==np.array(y_pred[i]).all():
# #             count+=1
# #     return count/len(y)

# def modify_random_elements(arr, percentage):
#     arr = arr.copy()  # جلوگیری از تغییر مستقیم در داده‌های ورودی
#     total_elements = arr.size
#     num_elements_to_modify = int(total_elements * percentage / 100)
#     indices = np.random.choice(total_elements, num_elements_to_modify, replace=False)
#     for index in indices:
#         row, col = divmod(index, arr.shape[1])
#         arr[row, col] =0
#     return arr


# def nose_train_list(arr, percentage):
#     arr = arr.copy()  # جلوگیری از تغییر مستقیم در داده‌های ورودی
#     total_elements = len(arr)
#     num_elements_to_modify = int(total_elements * percentage / 100)
#     for i in range(num_elements_to_modify):
#         indx=np.random.randint(0,total_elements)
#         arr[indx]=rout_90(np.array([arr[indx]]),10)
#     return arr



# def f_score_perectorn(y,y_pred):
#     f_count=0
#     _len=len(y_pred)
#     for idx in range(len(y)):
#         if int(y[idx])==int(y_pred[idx]):
#             f_count+=1
#     return f_count/_len

# def f_score_deltaRule(y,yperd):
#     n_count=0
#     for i in range(len(y)):
#         if list(y[i])==list(yperd[i]):
#             n_count+=1   
#     return n_count/len(y)
    
    
    

# # 2. تابع فعال‌سازی برای پرسپترون (Step Function)
# def step_function(x):
#     return np.where(x >= 0, 1, -1)

# # 3. تابع فعال‌سازی برای قانون دلتا (Sigmoid)
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def sigmoid_derivative(x):
#     return x * (1 - x)


# letters = load_letters("D:\pythonprojects\letters.json")
# labels = load_labels("D:\pythonprojects\labels.json")


# # 4. کلاس پرسپترون
# class Perceptron:
#     def __init__(self, input_size, lr=0.01, epochs=10000):
#         self.weights = np.random.randn(input_size)
#         self.bias = np.random.randn()
#         self.lr = lr
#         self.epochs = epochs
#         self.eror=None
#     def train(self, X, y):
#         self.eror=[]
#         for _ in range(self.epochs):
#             total_eror=0
#             for xi, target in zip(X, y):
#                 output = np.dot(xi, self.weights)+ self.bias
#                 error = target - output
#                 self.weights += self.lr * error * xi
#                 self.bias += self.lr * error
#                 total_eror += np.abs(error).sum()
#             self.eror.append(total_eror)
    
#     def predict(self, X):
#         pred_matrix=np.zeros(len(X))
#         for idex in  range(len(X)):
#             pred_matrix[idex]=np.round(np.dot(X[idex], self.weights) + self.bias)
#         return pred_matrix


# class DeltaRule:
#     def __init__(self, input_size, output_size, lr=0.01, epochs=1000):
#         # self.weights = np.random.randn(input_size, output_size)
#         self.weights=np.array([np.zeros(output_size) for i in range(input_size)])
#         self.bias = np.random.randn()
#         self.lr = lr
#         self.epochs = epochs
#         self.erors=None
#     def train(self, X, y): 
#         self.erors=[]
#         for _ in range(self.epochs):
#             total_error = 0
#             for xi, target in zip(X, y):
#                 output = sigmoid(np.dot(xi, self.weights) + self.bias) 
#                 error = target - output 
#                 self.weights += self.lr * np.outer(xi, error * sigmoid_derivative(output)) 
#                 self.bias += self.lr * error * sigmoid_derivative(output)
#                 total_error += np.abs(error).sum()
#             self.erors.append(total_error)

 
#     def predict(self,X):
#         pred_list=[]
#         for i in range(len(X)):
#             x=np.round(sigmoid(np.dot(X[i], self.weights) + self.bias))
#             pred_list.append(list(x))  
#         return pred_list
#  # 6. اجرای آموزش مدل‌ها




# X = np.array(list(letters.values()))
# y = np.array(list(labels.values()))
 
# perceptron = Perceptron(input_size=64)
# deltarule=DeltaRule(64,12)

# perceptron.train(X, y.argmax(axis=1))  # دسته‌بندی پرسپترون فقط برچسب‌های عددی می‌پذیرد
# deltarule.train(X, y)  # قانون دلتا می‌تواند چندکلاسه را یاد بگیرد

# print("Training finished!")


# plt.plot(deltarule.erors, 'r-x')
# plt.xlabel("Number of Iteration")
# plt.ylabel("Total error")
# plt.title("Delta Rule Training Error")
# plt.show()

# X_clean=np.array(list(letters.values()))
# X_noisy = np.array([modify_random_elements(x.reshape(8, 8), percentage=5).flatten() for x in X])
# #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #comment
# X_noisy_30 = np.array([modify_random_elements(x.reshape(8, 8), percentage=30).flatten() for x in X])
# X_noisy_50 = np.array([modify_random_elements(x.reshape(8, 8), percentage=50).flatten() for x in X])
# #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# print(f"Perceptron test (10% nose): {perceptron.predict(modify_random_elements(X_noisy,10))}")
# print(f"Perceptron test (clean): {perceptron.predict(X_clean)}")
# #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# # #comment
# print(f"Perceptron test (30% nose): {perceptron.predict(modify_random_elements(X_noisy_30,30))}")
# print(f"Perceptron test (50% nose): {perceptron.predict(modify_random_elements(X_noisy_50,50))}")
# #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 
# y_deta_pred=deltarule.predict(X)
# y_perceptron_pred=perceptron.predict(np.array(X))
# y_deta_pred_noisy=deltarule.predict(nose_train_list(X,10))
# y_perceptron_pred_noisy=perceptron.predict(nose_train_list(X,10))
# #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #comment
# y_deta_pred_noisy_30=deltarule.predict(nose_train_list(X,30))
# y_perceptron_pred_noisy_30=perceptron.predict(nose_train_list(X,30))
# y_deta_pred_noisy_50=deltarule.predict(nose_train_list(X,50))
# y_perceptron_pred_noisy_50=perceptron.predict(nose_train_list(X,50))
# #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 
# delta_acc_clean =f_score_deltaRule(y,y_deta_pred)
# perceptron_acc_clean = f_score_perectorn(y.argmax(axis=1),y_perceptron_pred)
# delta_acc_noisy= f_score_deltaRule(y,y_deta_pred_noisy)
# perceptron_acc_noisy = f_score_perectorn(y.argmax(axis=1),y_perceptron_pred_noisy)
# #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #comment
# delta_acc_noisy_30=f_score_deltaRule(y,y_deta_pred_noisy_30)
# perceptron_acc_noisy_30 = f_score_perectorn(y.argmax(axis=1),y_perceptron_pred_noisy_30)
# delta_acc_noisy_50=f_score_deltaRule(y,y_deta_pred_noisy_50)
# perceptron_acc_noisy_50 = f_score_perectorn(y.argmax(axis=1),y_perceptron_pred_noisy_50)
# #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

 




# print(f"Perceptron Training Accuracy (clean): {perceptron_acc_clean*100:.2f}%")
# print(f"Delta Rule Training Accuracy (clean): {delta_acc_clean*100:.2f}%")
# print(f"Perceptron Training Accuracy (noisy, 10%): {perceptron_acc_noisy*100:.2f}%")
# print(f"Delta Rule Training Accuracy (noisy, 10%): {delta_acc_noisy*100:.2f}%")
# #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #comment
# print(f"Perceptron Training Accuracy (noisy, 30%): {perceptron_acc_noisy_30*100:.2f}%")
# print(f"Delta Rule Training Accuracy (noisy, 30%): {delta_acc_noisy_30*100:.2f}%")
# print(f"Perceptron Training Accuracy (noisy, 50%): {perceptron_acc_noisy_50*100:.2f}%")
# print(f"Delta Rule Training Accuracy (noisy, 50%): {delta_acc_noisy_50*100:.2f}%")
# #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 
 


# X_=list(X)
# Y_=list(y)
# for epoc in range(4):
#     for i in range(len(X)):
#                 X_.append(turn_matrix(np.array(X[i]).reshape(8,8),np.random.randint(0,90)))
#                 Y_.append(y[i])
# for percent in range(10,40,10):
#         for j in range(len(X)):
#             X_.append(modify_random_elements(X[j].reshape(8,8),percent))
#             Y_.append(y[j])
# for k in range(len(X)):
#         X_.append(list(turn90(X[k].reshape(8,8)).reshape(1,64)[0]))
#         Y_.append(y[k])
  
 

  
# x_trian,x_test,y_trian,y_test=train_test_split(X_,Y_,test_size=0.1,random_state=123)


# perceptron_trian =Perceptron(input_size=64)
# deltarule_trian=DeltaRule(64,12)

# perceptron_trian.train([np.array(x).reshape(1,64)[0] for x in x_trian], np.array(y_trian).argmax(axis=1))  # دسته‌بندی پرسپترون فقط برچسب‌های عددی می‌پذیرد
# y_perceptron_pred_train=perceptron_trian.predict([np.array(x).reshape(1,64)[0] for x in x_trian])
# y_perceptron_pred_test=perceptron_trian.predict([np.array(x).reshape(1,64)[0] for x in x_test])
# perceptron_acc_train= f_score_perectorn(np.array(y_trian).argmax(axis=1),np.array(y_perceptron_pred_train))
# perceptron_acc_test= f_score_perectorn(np.array(y_test).argmax(axis=1),np.array(y_perceptron_pred_test))


# deltarule_trian.train([np.array(x).reshape(1,64)[0] for x in x_trian], y_trian ) # قانون دلتا می‌تواند چندکلاسه را یاد بگیرد
# y_deltarule_pred_train=deltarule_trian.predict(np.array(list([np.array(x).reshape(1,64)[0] for x in x_trian])))
# y_deltarule_pred_test=deltarule_trian.predict(np.array(list([np.array(x).reshape(1,64)[0] for x in x_test])))
# deltarule_acc_train= f_score_deltaRule(y_trian,y_deltarule_pred_train)
# deltarule_acc_test= f_score_deltaRule(y_test,y_deltarule_pred_test)






# X = np.array(list(letters.values()))
# y = np.array(list(labels.values()))
 
# perceptron = Perceptron(input_size=64)
# deltarule=DeltaRule(64,12)

# perceptron.train(X, y.argmax(axis=1))  # دسته‌بندی پرسپترون فقط برچسب‌های عددی می‌پذیرد
# deltarule.train(X, y)  # قانون دلتا می‌تواند چندکلاسه را یاد بگیرد

# print("Training finished!")


# plt.plot(deltarule.erors, 'r-x')
# plt.xlabel("Number of Iteration")
# plt.ylabel("Total error")
# plt.title("Delta Rule Training Error")
# plt.show()

# X_clean=np.array(list(letters.values()))
# X_noisy = np.array([modify_random_elements(x.reshape(8, 8), percentage=5).flatten() for x in X])
# #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #comment
# X_noisy_30 = np.array([modify_random_elements(x.reshape(8, 8), percentage=30).flatten() for x in X])
# X_noisy_50 = np.array([modify_random_elements(x.reshape(8, 8), percentage=50).flatten() for x in X])
# #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# print(f"Perceptron test (10% nose): {perceptron.predict(modify_random_elements(X_noisy,10))}")
# print(f"Perceptron test (clean): {perceptron.predict(X_clean)}")
# #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# # #comment
# print(f"Perceptron test (30% nose): {perceptron.predict(modify_random_elements(X_noisy_30,30))}")
# print(f"Perceptron test (50% nose): {perceptron.predict(modify_random_elements(X_noisy_50,50))}")
# #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 
# y_deta_pred=deltarule.predict(list([X]))
# y_perceptron_pred=perceptron.predict(np.array(X))
# y_deta_pred_noisy=deltarule.predict(nose_train_list(X,10))
# y_perceptron_pred_noisy=perceptron.predict(nose_train_list(X,10))
# #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #comment
# y_deta_pred_noisy_30=deltarule.predict(nose_train_list(X,30))
# y_perceptron_pred_noisy_30=perceptron.predict(nose_train_list(X,30))
# y_deta_pred_noisy_50=deltarule.predict(nose_train_list(X,50))
# y_perceptron_pred_noisy_50=perceptron.predict(nose_train_list(X,50))
# #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 
# delta_acc_clean = f_score_deltaRule(y,y_deta_pred)
# perceptron_acc_clean = f_score_perectorn(y.argmax(axis=1),y_perceptron_pred)
# delta_acc_noisy= f_score_deltaRule(y,y_deta_pred_noisy)
# perceptron_acc_noisy = f_score_perectorn(y.argmax(axis=1),y_perceptron_pred_noisy)
# #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #comment
# delta_acc_noisy_30= f_score_deltaRule(y,y_deta_pred_noisy_30)
# perceptron_acc_noisy_30 = f_score_perectorn(y.argmax(axis=1),y_perceptron_pred_noisy_30)
# delta_acc_noisy_50= f_score_deltaRule(y,y_deta_pred_noisy_50)
# perceptron_acc_noisy_50 = f_score_perectorn(y.argmax(axis=1),y_perceptron_pred_noisy_50)
# #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# print("$$$$$$$$$$$$$$$$$$$$$$$")
# print(y)
# print(y_perceptron_pred)
# print(y_perceptron_pred_noisy_30)
# print(y_perceptron_pred_noisy_50)
# y1=np.array(y).argmax(axis=1)
# f_p_count_30=0
# f_p_count_50=0

# print(y_perceptron_pred_noisy_30)


# for idx in range(len(y1)):
#     if int(y1[idx])==int(y_perceptron_pred_noisy_30[idx]):
#         f_p_count_30+=1
#     if int(y1[idx])==int(y_perceptron_pred_noisy_50[idx]):
#         f_p_count_50+=1

# n=len(y)

# print(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@f30%         {f_p_count_30/n}")
# print(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@f50%         {f_p_count_50/n}")
    
    




# print(f"Perceptron Training Accuracy (clean): {perceptron_acc_clean*100:.2f}%")
# print(f"Delta Rule Training Accuracy (clean): {delta_acc_clean*100:.2f}%")
# print(f"Perceptron Training Accuracy (noisy, 10%): {perceptron_acc_noisy*100:.2f}%")
# print(f"Delta Rule Training Accuracy (noisy, 10%): {delta_acc_noisy*100:.2f}%")
# #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #comment
# print(f"Perceptron Training Accuracy (noisy, 30%): {perceptron_acc_noisy_30*100:.2f}%")
# print(f"Delta Rule Training Accuracy (noisy, 30%): {delta_acc_noisy_30*100:.2f}%")
# print(f"Perceptron Training Accuracy (noisy, 50%): {perceptron_acc_noisy_50*100:.2f}%")
# print(f"Delta Rule Training Accuracy (noisy, 50%): {delta_acc_noisy_50*100:.2f}%")
# #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 
 


# X_=list(X)
# Y_=list(y)
# for epoc in range(4):
#     for i in range(len(X)):
#                 X_.append(turn_matrix(np.array(X[i]).reshape(8,8),np.random.randint(0,90)))
#                 Y_.append(y[i])
# for percent in range(10,40,10):
#         for j in range(len(X)):
#             X_.append(modify_random_elements(X[j].reshape(8,8),percent))
#             Y_.append(y[j])
# for k in range(len(X)):
#         X_.append(list(turn90(X[k].reshape(8,8)).reshape(1,64)[0]))
#         Y_.append(y[k])
  
 

  
# x_trian,x_test,y_trian,y_test=train_test_split(X_,Y_,test_size=0.1,random_state=123)


# perceptron_trian =Perceptron(input_size=64)
# deltarule_trian=DeltaRule(64,12)

# perceptron_trian.train([np.array(x).reshape(1,64)[0] for x in x_trian], np.array(y_trian).argmax(axis=1))  # دسته‌بندی پرسپترون فقط برچسب‌های عددی می‌پذیرد
# y_perceptron_pred_train=perceptron_trian.predict([np.array(x).reshape(1,64)[0] for x in x_trian])
# y_perceptron_pred_test=perceptron_trian.predict([np.array(x).reshape(1,64)[0] for x in x_test])
# perceptron_acc_train= f_score_perectorn(np.array(y_trian).argmax(axis=1),np.array(y_perceptron_pred_train))
# perceptron_acc_test= f_score_perectorn(np.array(y_test).argmax(axis=1),np.array(y_perceptron_pred_test))


# deltarule_trian.train([np.array(x).reshape(1,64)[0] for x in x_trian], y_trian ) # قانون دلتا می‌تواند چندکلاسه را یاد بگیرد
# y_deltarule_pred_train=deltarule_trian.predict1(np.array(list([np.array(x).reshape(1,64)[0] for x in x_trian])))
# y_deltarule_pred_test=deltarule_trian.predict1(np.array(list([np.array(x).reshape(1,64)[0] for x in x_test])))
# deltarule_acc_train= f_score_deltaRule(np.array(y_trian).argmax(axis=1),np.array(y_deltarule_pred_train).argmax(axis=1))
# deltarule_acc_test= f_score_deltaRule(np.array(y_test).argmax(axis=1),np.array(y_deltarule_pred_test).argmax(axis=1))


# print(f"Perceptron X_Train Accuracy : {perceptron_acc_train*100:.2f}%")
# print(f"Delta Rule X_Train Accuracy : {deltarule_acc_train*100:.2f}%")
# print(f"Perceptron X_Test  Accuracy : {perceptron_acc_test*100:.2f}%")
# print(f"Delta Rule X_Test  Accuracy : {deltarule_acc_test*100:.2f}%")
	



