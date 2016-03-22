import pyautogui

#pyautogui.PAUSE = 1
#pyautogui.FAILSAFE = True



class helloworld:
	def __init__(self):
		self.width, self.height = pyautogui.size()
	
	def printWH(self):
		print(self.width, self.height)
		

x = helloworld()
x.printWH()

