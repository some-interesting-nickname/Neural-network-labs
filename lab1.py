import PySimpleGUI as sg
import threading
import time
import numpy as np

def get_x_and_y(event):
    global lasx, lasy
    lasx, lasy = event.x, event.y

def draw_smth(event):
    global lasx, lasy
    canvas.create_line((lasx, lasy, event.x, event.y), 
                      fill='black', 
                      width=2)
   
    lasx, lasy = event.x, event.y

def paint(event):
	color='black'
	x1,y1=(event.x-brush),(event.y-brush)
	x2,y2=(event.x+brush),(event.y+brush)
	c.create_oval(x1,y1,x2,y2, fill=color, outline=color)

#Создание GUI, используя модуль PySimpleGUI
sg.theme('Dark')

layout = [
	[sg.Canvas(size=(32, 32), background_color='white', key= '-canvas-')],
	[sg.Button('Check'), sg.Button('Failure'), sg.Button('Clear')],
	[sg.Submit(), sg.Cancel(), sg.Input(key='-SAVEAS-FILENAME-', visible=False, enable_events=True), sg.FileSaveAs()]
]

window = sg.Window('Lab 1', layout)

window['-canvas-'].bind("<B1-Motion>",paint)
while True:
	event, values = window.read()
	if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
		break
	#print('You entered ', values[0])
	#if event == 'Submit':
	if event[:2] == ('-canvas-', '+CLICKED+'):
		paint(event)
window.close()
