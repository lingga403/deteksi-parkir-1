import cv2
import numpy as np
import imutils as im

#Webcam
cap	 = cv2.VideoCapture('Parkir.mp4')

#count line position

blue1 = 20
blue2 = 139
red1 = 28
red2 = 178
#Initil Substractor
algo = cv2.createBackgroundSubtractorMOG2()

#min width and etc
min_width_react=80
min_heigth_react=80

detec = []
offset= 4
car = 0
status =[]



def center_handle(x,y,w,h):
	x1= int(w/2)
	y1= int(h/2)
	cx= x+x1
	cy= y+y1
	return cx,cy



while True:
	ret, frame1 = cap.read()
	roi = frame1[1077: 1917,0: 1917]
	roi_z = im.resize(roi, width = 320)
	grey = cv2.cvtColor(roi_z, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(grey,(5,5),5)
	frame1 = im.resize(frame1, width = 320)



	# applying on each 	
	img_sub = algo.apply(blur)
	dilat = cv2.dilate(img_sub,np.ones((15,15)))
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
	dilatda = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
	dilatda = cv2.morphologyEx(dilatda, cv2.MORPH_CLOSE, kernel)
	counterSahpe,h = cv2.findContours(dilatda, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


	cv2.line(roi_z, (0,blue1), (317,blue2), (255,0,0),1)
	cv2.line(roi_z, (0,red1), (317,red2), (0,0,255),1)

	for (i,c) in enumerate(counterSahpe):
		(x,y,w,h) = cv2.boundingRect(c)
		validar_counter = (w>min_width_react) and (h>min_heigth_react)
		if not validar_counter:
			continue 

 
 
		cv2.rectangle(roi_z, (x,y), (x+w, y+h), (0,255,0),2)
		center= center_handle (x,y,w,h)
		detec.append(center)
		cv2.circle(roi_z, center,4 , (0,255,255),2)

#tes garis deteksi		
	for (x,y) in detec:
		if y< (blue1 and blue2 + offset) and y > (blue1 and blue2 - offset):
			cv2.line(roi_z, (0,20), (317,139), (255,255,255),1) #
			status.append(1)
			detec.remove((x,y)) 


		elif y< (red1 and red2 + offset) and y>(red1 and red2 - offset):
			cv2.line(roi_z, (0,28), (317,178), (255,255,255),1)
			status.append(2)
			detec.remove((x,y))

#algoritmma sekuensial
	if status ==[1,2]:
		car+=1 
		print("jumlah kendaraan saat ini : "+str(car))
		status.clear()

	elif status ==[2,1]:
		car-=1
		print("jumlah kendaraan saat ini : "+str(car))
		status.clear()

	cv2.imshow('blur', blur)
	cv2.imshow('grey', grey)
	cv2.imshow('deteksi', dilatda)
	cv2.imshow('Ori', frame1)
	cv2.imshow('roi' ,roi_z)

	if cv2.waitKey(30) & 0xFF == ord('q'):
            break
#end
cap.release()
cv2.destroyAllWindows()
    
