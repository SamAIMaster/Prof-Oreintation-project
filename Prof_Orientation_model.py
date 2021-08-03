

import tkinter as tk
from tkinter import *
import tkinter as ttk
import sklearn.neighbors._typedefs
import numpy as np
import pandas as pd
import pickle
df = pd.read_csv("career_pred.csv")

df = df.iloc[: , 7:]

dataset = df

for i in dataset.columns:
    Types = []
    
    for j in dataset[i]:
        #print(j)
        if not(j in Types):
            Types.append(j)


# Business Technical Manager Design

Business = ['Business Systems Analyst',
            'Business Intelligence Analyst',
            'Information Security Analyst',
            'CRM Business Analyst',
            'E-Commerce Analyst',
            'Systems Analyst',
            'Information Technology Auditor',
            'Quality Assurance Associate'
            ]
Technical = ['Software Systems Engineer', 
             'Network Engineer',
              'Software Engineer',
             'Technical Engineer',
             'Network Security Engineer',
             'Database Developer',
             'CRM Technical Developer',
             'Mobile Applications Developer',
             'Applications Developer',
             'Web Developer',
             'Software Developer', 
             'Technical Services/Help Desk/Tech Support',
             'Technical Support',
             'Software Quality Assurance (QA) / Testing',
             'Systems Security Administrator', 
             'Portal Administrator',
             'Network Security Administrator',
             'Database Administrator',
             'Solutions Architect',
            'Data Architect',
             'Programmer Analyst'
             ]
Manager = [
            'Project Manager',
            'Information Technology Manager', 
            'Database Manager'
]

Design = [
          'UX Designer',  
          'Design & UX'
]

soft = [2,1]
hard = [3,0 ]

# Data
data = df.iloc[:,:-1].values
label = df.iloc[:,-1]

#Label Encoding: COnverting To Numeric values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()

for i in range(6,31):
    data[:,i] = labelencoder.fit_transform(data[:,i])

#Normalizing the data
from sklearn.preprocessing import Normalizer
data1=data[:,:6]
normalized_data = Normalizer().fit_transform(data1)

data2=data[:,6:]
df1 = np.append(normalized_data,data2,axis=1)

#Combining into a dataset
df2=df.iloc[:,:-1]
dataset = pd.DataFrame(df1,columns=df2.columns)
#dataset

X=dataset.copy()
Y = df["Suggested Job Role"]

# Business Technical Manager Design
for i in range(len(Y)):
    if Y[i] in Business:
        Y[i] = 'Business'
        
    elif Y[i] in Technical:
        Y[i] = 'Technical'
    elif Y[i] in Design:
        Y[i] = 'Design'
    elif Y[i] in Manager:
        Y[i] = 'Manager'

# For label
label = df.iloc[:,-1]
original=label.unique() 
label=label.values
label2 = labelencoder.fit_transform(label)
y=pd.DataFrame(label2,columns=["Suggested Job Role"])
numeric=y["Suggested Job Role"].unique() 
Y = pd.DataFrame({'Suggested Job Role':original, 'Associated Number':numeric})

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

j_l = []
for i in Y:
    if not (i in j_l):
        j_l.append(i)

Y_train = []
#print(y_train)
for i in y_train['Suggested Job Role']:
  
  if i in soft:
    
    Y_train.append(0)
  elif i in hard:
    Y_train.append(1)

Y_test = []
for i in y_test['Suggested Job Role']:
  if i in soft:
    Y_test.append(0)
  elif i in hard:
    Y_test.append(1)


#from sklearn.ensemble import RandomForestClassifier
#rf_model=RandomForestClassifier(n_estimators=1000,max_features=2,oob_score=True)

#rf_model.fit(X_train,y_train)

#from sklearn.neighbors import KNeighborsClassifier
#knn = KNeighborsClassifier(n_neighbors=9)

#knn.fit(X_train,y_train)
filename = "KNN_Model.sav"
knn = pickle.load(open(filename,'rb'))


#pickle.dump(knn,open(filename,'wb'))
#print("Acc: ",knn.score(X_test,y_test))


window = tk.Tk()
window.title('Questionare P1')
window.geometry("400x550")



'''
Button
def run_model():
    print("Model")
button = tk.Button(window, text='Stop', width=25, command = run_model)
button.pack()
window.mainloop()

'''

'''
Put text - LAbel

w = Label(window, text='GeeksForGeeks.org!')
w.pack()
'''

'''
List
top = Tk()
Lb = Listbox(top)
Lb.insert(1, 'Python')
Lb.insert(2, 'Java')
Lb.insert(3, 'C++')
Lb.insert(4, 'Any other')
Lb.pack()
'''

'''
Check Box
master = Tk()
var1 = IntVar()
Checkbutton(master, text='male', variable=var1).grid(row=0, sticky=W)
var2 = IntVar()
Checkbutton(master, text='female', variable=var2).grid(row=1, sticky=W)
mainloop()
'''

'''
Text box

master = Tk()
Label(master, text='First Name').grid(row=0)
Label(master, text='Last Name').grid(row=1)
e1 = Entry(master)
e2 = Entry(master)
e1.grid(row=0, column=1)
e2.grid(row=1, column=1)
mainloop()
'''

'''
List box

top = Tk()
Lb = Listbox(top)
Lb.insert(1, 'Python')
Lb.insert(2, 'Java')
Lb.insert(3, 'C++')
Lb.insert(4, 'Any other')
Lb.pack()
top.mainloop()
'''

'''
check list - Radio button

from tkinter import *
root = Tk()
v = IntVar()
Radiobutton(root, text='GfG', variable=v, value=1).pack(anchor=W)
Radiobutton(root, text='MIT', variable=v, value=2).pack(anchor=W)
mainloop()
'''

'''
scale

from tkinter import *
master = Tk()
w = Scale(master, from_=0, to=42)
w.pack()
w = Scale(master, from_=0, to=200, orient=HORIZONTAL)
w.pack()
mainloop()
'''
q1 = IntVar()
#Q 1
l1 = Label(window, text='Уровень знания математики в % (1-100)')
l1.pack()

q1_w = Scale(window, from_=1, to=100,orient=HORIZONTAL, variable = q1)
q1_w.pack()

q2 = IntVar()
#Q 2
l2 = Label(window, text='Навыки коммуникации в %  (1-100)' )
l2.pack()

q2_w = Scale(window, from_=1, to=100,orient=HORIZONTAL, variable = q2)
q2_w.pack()

q3 = IntVar()
#Q 3
l3 = Label(window, text='Сколько часов в день вы работаете?  (1-12)' )
l3.pack()

q3_w = Scale(window, from_=1, to=12,orient=HORIZONTAL, variable = q3)
q3_w.pack()

q4 = IntVar()
#Q 4
l4 = Label(window, text='Оцените ваш уровень логического мышления (1-8)', )
l4.pack()


q4_w = Scale(window, from_=1, to=8,orient=HORIZONTAL, variable = q4)
q4_w.pack()

q5 = IntVar()
# Q 5
l5 = Label(window, text='Сколько раз вы участвовали в Хакатонах? (0-6)', )
l5.pack()

q5_w = Scale(window, from_=0, to=6,orient=HORIZONTAL, variable = q5)
q5_w.pack()

q6 = IntVar()
# Q 6
l6 = Label(window, text='Оцените свой уровень кодинга (1-9)', )
l6.pack()

q6_w = Scale(window, from_=1, to=9,orient=HORIZONTAL, variable = q6)
q6_w.pack()

q7 = IntVar()
# Q 7
l7 = Label(window, text='Оцените ваш Навык публичных выступлений (1-9)', )
l7.pack()

q7_w = Scale(window, from_=1, to=9,orient=HORIZONTAL, variable = q7)
q7_w.pack()


def show1():
    print(q1.get()," ",q2.get()," ",q3.get()," ",q4.get()," ",q5.get()," ",q6.get()," ",q7.get())

#b_show1 = tk.Button(window, text='Show', width=25, command = show1)
#b_show1.pack()

b_next1 = tk.Button(window, text='Next', fg = "green",width=25, command = window.destroy)
b_next1.pack()

#the end of first page

mainloop()

show1()
#New Window
window = tk.Tk()
window.title('Questionare P2')
window.geometry("400x550")


# Q 8
q8 = IntVar()

l8 = Label(window, text='Работали ли вы над проектом до этого времени?', )
l8.pack()

Radiobutton(window, text='Yes', variable=q8, value=1).pack(anchor=W)
Radiobutton(window, text='No', variable=q8, value=0).pack(anchor=W)

# Q 9
q9 = IntVar()

l9 = Label(window, text='Способны ли вы к самообучению?', )
l9.pack()

Radiobutton(window, text='Yes', variable=q9, value=1).pack(anchor=W)
Radiobutton(window, text='No', variable=q9, value=0).pack(anchor=W)

# Q 10
q10 = IntVar()

l10 = Label(window, text='Ходите ли вы на дополнительные занятия?', )
l10.pack()

Radiobutton(window, text='Yes', variable=q10, value=1).pack(anchor=W)
Radiobutton(window, text='No', variable=q10, value=0).pack(anchor=W)

# Q 11
q11 = IntVar()

l11 = Label(window, text='Какие у Вас сертификаты есть?', )
l11.pack()

Radiobutton(window, text='Программирование оболочки', variable=q11, value=8).pack(anchor=W)
Radiobutton(window, text='Машинное обучение', variable=q11, value=5).pack(anchor=W)
Radiobutton(window, text='Разработка приложений', variable=q11, value=0).pack(anchor=W)
Radiobutton(window, text='Python', variable=q11, value=6).pack(anchor=W)
Radiobutton(window, text='R программирование', variable=q11, value=7).pack(anchor=W)
Radiobutton(window, text='hadoop', variable=q11, value=4).pack(anchor=W)
Radiobutton(window, text='Информационная безопасность', variable=q11, value=3).pack(anchor=W)
Radiobutton(window, text='Создание дистрибутива', variable=q11, value=1).pack(anchor=W)
Radiobutton(window, text='Фул стек', variable=q11, value=2).pack(anchor=W)






def show2():
    print(q8.get()," ",q9.get()," ",q10.get()," ",q11.get())


#b_show2 = tk.Button(window, text='Show', width=25, command = show2)
#b_show2.pack()

b_next2 = tk.Button(window, text='Next', fg = "green",width=25, command = window.destroy)
b_next2.pack()



mainloop()
show2()
#New page 3
window = tk.Tk()
window.title('Questionare P3')
window.geometry("400x550")

# Q 12
q12 = IntVar()

l12 = Label(window, text='Какие семинары вы проходили?', )
l12.pack()

Radiobutton(window, text='Облачные вычисления', variable=q12, value=0).pack(anchor=W)
Radiobutton(window, text='Безопасность баз данных', variable=q12, value=2).pack(anchor=W)
Radiobutton(window, text='Веб-технологии', variable=q12, value=7).pack(anchor=W)
Radiobutton(window, text='Наука о данных', variable=q12, value=1).pack(anchor=W)
Radiobutton(window, text='Тестирование', variable=q12, value=6).pack(anchor=W)
Radiobutton(window, text='Взлом', variable=q12, value=4).pack(anchor=W)
Radiobutton(window, text='Разработка игр', variable=q12, value=3).pack(anchor=W)
Radiobutton(window, text='Системное проектирование', variable=q12, value=5).pack(anchor=W)

# Q 13
q13 = IntVar()

l13 = Label(window, text='Взяли ли вы тест на выявления своего таланта?', )
l13.pack()

Radiobutton(window, text='Да', variable=q13, value=1).pack(anchor=W)
Radiobutton(window, text='Нет', variable=q13, value=0).pack(anchor=W)

# Q 14
q14 = IntVar()

l14 = Label(window, text='Участвовали  вы на олимпиадах?', )
l14.pack()

Radiobutton(window, text='Да', variable=q14, value=1).pack(anchor=W)
Radiobutton(window, text='Нет', variable=q14, value=0).pack(anchor=W)

# Q 15
q15 = IntVar()

l15 = Label(window, text='Какой у вас уровень навыков чтения и письма?', )
l15.pack()

Radiobutton(window, text='Отлично', variable=q15, value=0).pack(anchor=W)
Radiobutton(window, text='Среднее', variable=q15, value=1).pack(anchor=W)
Radiobutton(window, text='Плохо', variable=q15, value=2).pack(anchor=W)



def show3():
    print(q12.get()," ",q13.get()," ",q14.get()," ",q15.get())


#b_show3 = tk.Button(window, text='Show', width=25, command = show2)
#b_show3.pack()

b_next3 = tk.Button(window, text='Next', fg = "green",width=25, command = window.destroy)
b_next3.pack()

mainloop()
show3()

# new page 4
window = tk.Tk()
window.title('Questionare P4')
window.geometry("400x650")

# Q 16
q16 = IntVar()

l16 = Label(window, text='Какая у вас память?' )
l16.pack()

Radiobutton(window, text='Отличная', variable=q16, value=0).pack(anchor=W)
Radiobutton(window, text='Нормальная', variable=q16, value=1).pack(anchor=W)
Radiobutton(window, text='Плохая', variable=q16, value=2).pack(anchor=W)

# Q 17
q17 = IntVar()

l17 = Label(window, text='Интересующиеся вас предметы?' )
l17.pack()

Radiobutton(window, text='Облачные вычисления', variable=q17, value=4).pack(anchor=W)
Radiobutton(window, text='Сети', variable=q17, value=7).pack(anchor=W)
Radiobutton(window, text='Взлом', variable=q17, value=6).pack(anchor=W)
Radiobutton(window, text='Компьютерная архитектура', variable=q17, value=0).pack(anchor=W)
Radiobutton(window, text='Программирование', variable=q17, value=9).pack(anchor=W)
Radiobutton(window, text='Параллельные вычисления', variable=q17, value=8).pack(anchor=W)
Radiobutton(window, text='Интернет вещей', variable=q17, value=1).pack(anchor=W)
Radiobutton(window, text='Инженерия данных', variable=q17, value=5).pack(anchor=W)
Radiobutton(window, text='Программная инженерия', variable=q17, value=3).pack(anchor=W)
Radiobutton(window, text='Менеджмент', variable=q17, value=2).pack(anchor=W)

# Q 18
q18 = IntVar()

l18 = Label(window, text='Заинтересованная область карьеры? ' )
l18.pack()

Radiobutton(window, text='Разработчик системы', variable=q18, value=4).pack(anchor=W)
Radiobutton(window, text='Аналитик бизнес-процессов', variable=q18, value=0).pack(anchor=W)
Radiobutton(window, text='Разработчик', variable=q18, value=2).pack(anchor=W)
Radiobutton(window, text='Тестирование', variable=q18, value=5).pack(anchor=W)
Radiobutton(window, text='Безопасность', variable=q18, value=3).pack(anchor=W)
Radiobutton(window, text='Облачные вычисления', variable=q18, value=1).pack(anchor=W)

def show4():
    print(q16.get()," ",q17.get()," ",q18.get())




#b_show4 = tk.Button(window, text='Show', width=25, command = show4)
#b_show4.pack()

b_next4 = tk.Button(window, text='Next', fg = "green",width=25, command = window.destroy)
b_next4.pack()

mainloop()
show4()


# new page 5
window = tk.Tk()
window.title('Questionare P5')
window.geometry("400x550")

# Q 19
q19= IntVar()

l19 = Label(window, text='Что для вас важнее иметь высшее образование или опыт работы?', )
l19.pack()

Radiobutton(window, text='Высшее образование', variable=q19, value=0).pack(anchor=W)
Radiobutton(window, text='Работа', variable=q19, value=1).pack(anchor=W)

# Q 20
q20= IntVar()

l20 = Label(window, text='Тип кампании в которой вы бы хотели работать?', )
l20.pack()

Radiobutton(window, text='Веб-услуги', variable=q20, value=8).pack(anchor=W)
Radiobutton(window, text='Услуги SaaS', variable=q20, value=4).pack(anchor=W)
Radiobutton(window, text='Продажи и маркетинг', variable=q20, value=5).pack(anchor=W)
Radiobutton(window, text='Услуги по тестированию и обслуживанию', variable=q20, value=7).pack(anchor=W)
Radiobutton(window, text='Разработка продукта', variable=q20, value=9).pack(anchor=W)
Radiobutton(window, text='BPA', variable=q20, value=0).pack(anchor=W)
Radiobutton(window, text='Сервис', variable=q20, value=6).pack(anchor=W)
Radiobutton(window, text='Продукт', variable=q20, value=3).pack(anchor=W)
Radiobutton(window, text='Облачные услуги', variable=q20, value=1).pack(anchor=W)
Radiobutton(window, text='Финансы', variable=q20, value=2).pack(anchor=W)

# Q 21
q21= IntVar()

l21 = Label(window, text='На вашу профессию влияют взгляды старших людей?', )
l21.pack()

Radiobutton(window, text='Да', variable=q21, value=1).pack(anchor=W)
Radiobutton(window, text='Нет', variable=q21, value=0).pack(anchor=W)

# Q 22
q22= IntVar()

l22 = Label(window, text='Увлекаетесь видеоиграми?', )
l22.pack()

Radiobutton(window, text='Да', variable=q22, value=1).pack(anchor=W)
Radiobutton(window, text='Нет', variable=q22, value=0).pack(anchor=W)

#
def show5():
    print(q19.get()," ",q20.get()," ",q21.get()," ",q22.get())

#b_show5 = tk.Button(window, text='Show', width=25, command = show5)
#b_show5.pack()

b_next5= tk.Button(window, text='Next', fg = "green",width=25, command = window.destroy)
b_next5.pack()

mainloop()

show5()


# new page 6
window = tk.Tk()
window.title('Questionare P6')
window.geometry("400x900")

# Q 23
q23= IntVar()

l23 = Label(window, text='Интересующийся жанры книг?', )
l23.pack()

Radiobutton(window, text='Молитвенные книги', variable=q23, value=21).pack(anchor=W)
Radiobutton(window, text='Детские', variable=q23, value=5).pack(anchor=W)
Radiobutton(window, text='О путешествиях', variable=q23, value=29).pack(anchor=W)
Radiobutton(window, text='Романтические', variable=q23, value=23).pack(anchor=W)
Radiobutton(window, text='Кулинарные книги', variable=q23, value=7).pack(anchor=W)
Radiobutton(window, text='Саморазвитие', variable=q23, value=27).pack(anchor=W)
Radiobutton(window, text='Драма', variable=q23, value=10).pack(anchor=W)
Radiobutton(window, text='Математика', variable=q23, value=18).pack(anchor=W)
Radiobutton(window, text='Религия-духовность', variable=q23, value=22).pack(anchor=W)
Radiobutton(window, text='Антология', variable=q23, value=1).pack(anchor=W)
Radiobutton(window, text='Трилогия', variable=q23, value=30).pack(anchor=W)
Radiobutton(window, text='Автобиографии', variable=q23, value=3).pack(anchor=W)
Radiobutton(window, text='Мистерия', variable=q23, value=19).pack(anchor=W)
Radiobutton(window, text='Дневники', variable=q23, value=8).pack(anchor=W)
Radiobutton(window, text='Журналы', variable=q23, value=17).pack(anchor=W)
Radiobutton(window, text='История', variable=q23, value=15).pack(anchor=W)
Radiobutton(window, text='Искусство', variable=q23, value=2).pack(anchor=W)
Radiobutton(window, text='Словари', variable=q23, value=9).pack(anchor=W)
Radiobutton(window, text='Ужас', variable=q23, value=16).pack(anchor=W)
Radiobutton(window, text='Энциклопедии', variable=q23, value=11).pack(anchor=W)
Radiobutton(window, text='Экшн и приключения', variable=q23, value=0).pack(anchor=W)
Radiobutton(window, text='Фэнтези', variable=q23, value=12).pack(anchor=W)
Radiobutton(window, text='Комиксы', variable=q23, value=6).pack(anchor=W)
Radiobutton(window, text='Научная фантастика', variable=q23, value=26).pack(anchor=W)
Radiobutton(window, text='Серия', variable=q23, value=28).pack(anchor=W)
Radiobutton(window, text='Руководство', variable=q23, value=13).pack(anchor=W)
Radiobutton(window, text='Биографии', variable=q23, value=4).pack(anchor=W)
Radiobutton(window, text='Здоровье', variable=q23, value=14).pack(anchor=W)
Radiobutton(window, text='Сатира', variable=q23, value=24).pack(anchor=W)
Radiobutton(window, text='Наука', variable=q23, value=25).pack(anchor=W)
Radiobutton(window, text='Поэзия', variable=q23, value=20).pack(anchor=W)

b_next6= tk.Button(window, text='Next', fg = "green",width=25, command = window.destroy)
b_next6.pack()

mainloop()
print(q23)
# new page 7
window = tk.Tk()
window.title('Questionare P7')
window.geometry("400x650")

# Q 24
q24= IntVar()

l24 = Label(window, text='Ожидаемая цель, зарплата или работа?', )
l24.pack()

Radiobutton(window, text='Зарплата', variable=q24, value=1).pack(anchor=W)
Radiobutton(window, text='Работа', variable=q24, value=0).pack(anchor=W)

# Q 25
q25= IntVar()

l25 = Label(window, text='Вы в отношениях?', )
l25.pack()

Radiobutton(window, text='Да', variable=q25, value=1).pack(anchor=W)
Radiobutton(window, text='Нет', variable=q25, value=0).pack(anchor=W)

# Q 26
q26= IntVar()

l26 = Label(window, text='Какое у вас поведения?', )
l26.pack()

Radiobutton(window, text='Упрямый/упертый', variable=q26, value=1).pack(anchor=W)
Radiobutton(window, text='Нежный/мягкий', variable=q26, value=0).pack(anchor=W)

# Q 27
q27= IntVar()

l27 = Label(window, text='Вы хотите работать в управляющим отделе или в техническом?', )
l27.pack()

Radiobutton(window, text='Управляющий', variable=q27, value=0).pack(anchor=W)
Radiobutton(window, text='Технический', variable=q27, value=1).pack(anchor=W)

# Q 28
q28= IntVar()

l28 = Label(window, text='Вас мотивирует работа или зарплата?', )
l28.pack()

Radiobutton(window, text='Зарплата', variable=q28, value=0).pack(anchor=W)
Radiobutton(window, text='Работа', variable=q28, value=1).pack(anchor=W)

# Q 29
q29= IntVar()

l29 = Label(window, text='Вы работаете усердно или с умом?', )
l29.pack()

Radiobutton(window, text='Усердно', variable=q29, value=0).pack(anchor=W)
Radiobutton(window, text='С умом', variable=q29, value=1).pack(anchor=W)

# Q 30
q30= IntVar()

l30 = Label(window, text='Вы когда нибудь работали в команде?', )
l30.pack()

Radiobutton(window, text='Да', variable=q30, value=1).pack(anchor=W)
Radiobutton(window, text='Нет', variable=q30, value=0).pack(anchor=W)

# Q 31
q31= IntVar()

l31 = Label(window, text='Вы интроверт?', )
l31.pack()

Radiobutton(window, text='Да', variable=q31, value=1).pack(anchor=W)
Radiobutton(window, text='Нет', variable=q31, value=0).pack(anchor=W)



b_next7= tk.Button(window, text='Get Results', fg = "green",width=25, command = window.destroy)
b_next7.pack()

mainloop()


New_human = [q1.get(),
             q2.get(),
             q3.get(),q4.get(),q5.get(),q6.get(),q7.get(),
             q8.get(),q9.get(),q10.get(),q11.get(),q12.get(),q13.get(),q14.get(),
             q15.get(),q16.get(),q17.get(),q18.get(),q19.get(),q20.get(),q21.get(),
             q22.get(),q23.get(),q24.get(),q25.get(),q26.get(),q27.get(),q28.get(),
             q29.get(),q30.get(),q31.get()]
print("Human: ",New_human)

df = pd.read_csv("career_pred.csv")

df = df.iloc[: , 7:]
data = df.iloc[:,:-1].values
NN = Normalizer().fit(data[:,:6])
                      
new_6 =NN.transform([[q1.get(),q2.get(),q3.get(),q4.get(),q5.get(),q6.get()]])
New_human[0] = new_6[0][0]
New_human[1] = new_6[0][1]
New_human[2] = new_6[0][2]
New_human[3] = new_6[0][3]
New_human[4] = new_6[0][4]
New_human[5] = new_6[0][5]

print("New_human: ",New_human)
pr = knn.predict([New_human])
text1 = ''
if pr==1:
    text1 ="Design"
if pr==2:
    text1 ="Manager"
if pr==0:
    text1 ="Business"
if pr==3:
    text1 ="Technical"

# new page 7
window = tk.Tk()
window.title('Answer')
window.geometry("150x100")

An = Label(window, text=text1, )
An.pack()
    
