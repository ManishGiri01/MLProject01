import numpy as np
import cv2
import pickle


width=640
height=480
thresold=0.65

cap=cv2.VideoCapture(0)
cap.set(3,width)
cap.set(4,height)

pickle_in=open("model_trained.p","rb")
model=pickle.load(pickle_in)

def preProcessing(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)
    img=img/255
    return img

while True:
    success,imgOriginal=cap.read()
    img=np.asarray(imgOriginal)
    img=cv2.resize(img,(32,32))
    img=preProcessing(img)
    #cv2.imshow("processed Image",img)
    img=img.reshape(1,32,32,1)

    res=model.predict([img])
    x=np.argmax(res,axis=1)
    classIndex=int(x)
    #print(classIndex)

    predictions=model.predict(img)
    #print(predictions)
    probVal=np.amax(predictions)
    print(classIndex,probVal)


    if probVal>thresold:
        cv2.putText(imgOriginal,str(classIndex)+"  "+str(probVal),
                    (50,50),cv2.FONT_HERSHEY_COMPLEX,
                    1,(0,0,255),1)


    cv2.imshow("original Image",imgOriginal)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
