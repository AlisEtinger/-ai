#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tkinter as tk
from PIL import Image,ImageTk
from time import sleep


# In[2]:


flag = False


# In[3]:


def pauce():
    global flag 
    flag = not flag 
    while flag:
        doung()


# In[4]:


def doing():
    global flag
    while flag:
        for i in range(12):
            if not flag:break
                box = f_out.crop((i*67,0,i*67+67,164))
                img = ImageTk.PhotoImage(image=box)
                gif = cv.create_image(180,135,image=img)
                cv.update()
                sleep(0.2)


# In[5]:


root = tk.Tk()
root.geometry('400x320')
cv = tk.Canvas(root, width=350, height=260, bg='lightray')
cv.pack()
f_in = 'd:\\img.png'
f_out = Image.open(f_in)


# In[6]:


box = f_out.crop((0,0,67,164))
img = ImageTk.Photoimage(image=box)
gif = cv.create_image(180,135,image=img)


# In[7]:


tk.Button(root,command=pause,text='1/2').place(x=170,y=275)


# In[ ]:


root.mainloop()


# In[ ]:




