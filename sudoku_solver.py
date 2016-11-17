import cv2
import numpy as np 
from PIL import Image
import pytesser as pytesser
import sudoku_math as solve
import time

def rectify(h):
      h = h.reshape((4,2))
      hnew = np.zeros((4,2),dtype = np.float32)

      add = h.sum(1)
      hnew[0] = h[np.argmin(add)]
      hnew[2] = h[np.argmax(add)]
       
      diff = np.diff(h,axis = 1)
      hnew[1] = h[np.argmin(diff)]
      hnew[3] = h[np.argmax(diff)]

      return hnew
def sudokuSolve(img):
  ticks=time.time()
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

  gray = cv2.GaussianBlur(gray,(5,5),0)
  thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)


  hierarchy,contours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  biggest = None
  max_area = 0
  for i in contours:
          area = cv2.contourArea(i)
          if area > 100:
                  peri = cv2.arcLength(i,True)
                  approx = cv2.approxPolyDP(i,0.02*peri,True)
                  if area > max_area and len(approx)==4:
                          biggest = approx
                          max_area = area
  cv2.drawContours(img, biggest, -1, (0,255,0), 8)

  biggest=rectify(biggest)

  thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)


  h = np.array([ [0,0],[449,0],[449,449],[0,449] ],np.float32)
  retval = cv2.getPerspectiveTransform(biggest,h)
  warp = cv2.warpPerspective(thresh,retval,(450,450))
  cv2.imwrite('box.jpg',warp)
  # mask=cv2.imread('mask.jpg')
  # mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
  # ret, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
  # i was trying something else! 
  for y in range (0,9):
    for x in range (0,9) :

      morph=warp[(50*y):((50*y)+50),(50*x):((50*x)+50)]

      morph=morph[5:45,5:45]    
      morph=255-morph
      cv2.imwrite('sudokuDigits/cell'+str(y)+str(x)+'.jpg',morph)

         
                  
  y=0
  x=0  
  text=''      	
  fullResult=''
  print "| Recognizing Numbers. . . . "

  keys = [i for i in range(48,58)]
  for y in range (0,9):
      for x in range (0,9) :
          im = Image.open('sudokuDigits/cell'+str(y)+str(x)+'.jpg')
          text =pytesser.image_to_string(im)
          if text=='\n' or text=='' or text==' ' or  ord(text[0]) not in keys:
              fullResult=fullResult+'0'
          else:
              fullResult=fullResult+str(text[0])
          

  print "Detected Game!"
  for i in range(0,81):
      print fullResult[i],
      if (i+1)%9 == 0:
          print ""

  print ""

  print "| Solving . . . ."
  solve.r(fullResult)

  timetaken=time.time()-ticks
  print "SolvedIn>> "+str(timetaken)+"secs"

  
if __name__ == "__main__":
  img=cv2.imread('sudoku.jpg')
  sudokuSolve(img)
  imgBoard=cv2.imread('box.jpg')
  cv2.imshow('sudokuGame',imgBoard)
  cv2.waitKey(0)