import csv
import cv2
import random
import numpy as np
import pytesseract

import os

from flask import Flask, render_template, redirect, url_for, request

app = Flask(__name__)

scale = 0.5
circles = []
counter = 0
counter2 = 0
point1 = []
point2 = []
myPoints = []
myColor = []

@app.route("/")
def login():
    return render_template('Login.html')

@app.route("/FeeForm")
def feeForm():
    return render_template('FeeForm.html')

@app.route("/AdmissionForm")
def admissionForm():
    return render_template('AdmissionForm.html')

@app.route("/CustomizeForm")
def customizeForm():
    return render_template('CustomizeForm.html')


@app.route("/Data")
def data():
    return render_template('Data.html')

@app.route("/home")
def home():
    return render_template('home.html')


@app.route("/", methods=['POST'])
def passwd():
    email = request.form.get('email')
    password = request.form.get('password')
    if ((email == "admin@gmail.com") and (password == "admin")):
        return redirect(url_for('home'))
    else:
        x = "Wrong ID Password"
        return render_template('Login.html',err=x)


@app.route("/FeeForm", methods=['POST'])
def form1():

    per = 25
    pixelThreshold = 300

    roi = [[(398, 280), (1040, 318), 'text', 'bank'],
       [(398, 342), (1038, 380), 'text', 'erp'],
       [(400, 404), (1074, 440), 'text', 'erpID'],
       [(400, 466), (1076, 504), 'text', 'program'],
       [(400, 526), (1072, 562), 'text', 'name'],
       [(398, 590), (1074, 626), 'text', 'email'],
       [(402, 650), (1076, 688), 'text', 'amount'],
       [(400, 714), (1040, 746), 'text', 'semester'],
       [(402, 774), (1000, 810), 'text', 'feestype'],
       [(402, 832), (1042, 868), 'text', 'date']]


    pytesseract.pytesseract.tesseract_cmd = 'D:\\New Volume\\OpenCV\\Tesseract-OCR\\tesseract.exe'

    imgQ = cv2.imread('./static/QueryImage/Query1.png')
    h, w, c = imgQ.shape
    # imgQ = cv2.resize(imgQ,(w//3,h//3))

    orb = cv2.ORB_create(1000)
    kp1, des1 = orb.detectAndCompute(imgQ, None)
    imgkp1 = cv2.drawKeypoints(imgQ,kp1,None)
    cv2.imshow("keypoints", imgkp1)

    path = './static/UserForm1'
    mypicList = os.listdir(path)
    #print(mypicList)

    for j, y in enumerate(mypicList):
        img = cv2.imread(path + "/" + y)
        # cv2.imshow(y, img) #all the sample forms
        kp2, des2 = orb.detectAndCompute(img, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.match(des2, des1)
        matches.sort(key=lambda x: x.distance)
        good = matches[:int(len(matches) * (per / 100))]
        imgMatch = cv2.drawMatches(img, kp2, imgQ, kp1, good[:100], None, flags=2)
        # cv2.imshow(y, imgMatch) #shows features on the form

        srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
        imgScan = cv2.warpPerspective(img, M, (w, h))  # straightning images however our forms are straight for now
        # cv2.imshow(y,imgScan)

        imgShow = imgScan.copy()
        imgMask = np.zeros_like(imgShow)

        myData = []
        #print(f'################## Extracting Data from Form {j}  ##################')

        for x, r in enumerate(roi):
            cv2.rectangle(imgMask, (r[0][0], r[0][1]), (r[1][0], r[1][1]), (0, 255, 0), cv2.FILLED)
            imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1,
                                      0)  # created a mask and highlighted the region on information we needed.

            imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
            # cv2.imshow(str(x), imgCrop)  #cropping out the selected portion to be passed to pytesseract

            if r[2] == 'text':
                # print('{} :{}'.format(r[3], pytesseract.image_to_string(imgCrop)))
                # print(f'{r[3]} :{pytesseract.image_to_string(imgCrop)}')
                myData.append(pytesseract.image_to_string(imgCrop))
            if r[2] == 'box':
                imgGray = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)
                imgThresh = cv2.threshold(imgGray, 170, 255, cv2.THRESH_BINARY_INV)[1]
                totalPixels = cv2.countNonZero(imgThresh)
                # print(totalPixels)
                if totalPixels > pixelThreshold:
                    totalPixels = 1;
                else:
                    totalPixels = 0
                # print(f'{r[3]} :{totalPixels}')
                myData.append(totalPixels)
            cv2.putText(imgShow, str(myData[x]), (r[0][0], r[0][1]),
                        cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 0, 255), 4)

        with open('./static/Form1Output.csv', 'a+') as f:
            for data in myData:
                f.write((str(data) + ','))
            f.write('\n')

        imgShow = cv2.resize(imgShow, (w // 2, h // 2))
        cv2.imshow(y + "2", imgShow)

    # cv2.imshow("output", imgQ)
    cv2.waitKey(0)
    op1 = "Processing Completed go to See Data For Output"
    return render_template('FeeForm.html',frm1d=op1)


@app.route("/AdmissionForm", methods=['POST'])
def form2():

    per = 25
    pixelThreshold = 300

    roi = [
        [(408, 304), (752, 342), 'text', 'name'],
        [(406, 394), (758, 428), 'text', 'lastName'],
        [(406, 484), (752, 520), 'text', 'Mobnumber'],
        [(408, 596), (754, 632), 'text', 'emailADD '],
        [(408, 706), (752, 742), 'text', 'CNIC'],
        [(408, 756), (714, 790), 'text', 'testCenter']
    ]


    pytesseract.pytesseract.tesseract_cmd = 'D:\\New Volume\\OpenCV\\Tesseract-OCR\\tesseract.exe'

    imgQ = cv2.imread('./static/QueryImage/Query2.png')
    h, w, c = imgQ.shape
    # imgQ = cv2.resize(imgQ,(w//3,h//3))

    orb = cv2.ORB_create(1000)
    kp1, des1 = orb.detectAndCompute(imgQ, None)
    # imgkp1 = cv2.drawKeypoints(imgQ,kp1,None)
    # cv2.imshow("keypoints", imgkp1)

    path = './static/UserForm2'
    mypicList = os.listdir(path)
    #print(mypicList)

    for j, y in enumerate(mypicList):
        img = cv2.imread(path + "/" + y)
        # cv2.imshow(y, img) #all the sample forms
        kp2, des2 = orb.detectAndCompute(img, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.match(des2, des1)
        matches.sort(key=lambda x: x.distance)
        good = matches[:int(len(matches) * (per / 100))]
        imgMatch = cv2.drawMatches(img, kp2, imgQ, kp1, good[:100], None, flags=2)
        # cv2.imshow(y, imgMatch) #shows features on the form

        srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
        imgScan = cv2.warpPerspective(img, M, (w, h))  # straightning images however our forms are straight for now
        # cv2.imshow(y,imgScan)

        imgShow = imgScan.copy()
        imgMask = np.zeros_like(imgShow)

        myData = []
        #print(f'################## Extracting Data from Form {j}  ##################')

        for x, r in enumerate(roi):
            cv2.rectangle(imgMask, (r[0][0], r[0][1]), (r[1][0], r[1][1]), (0, 255, 0), cv2.FILLED)
            imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1,
                                      0)  # created a mask and highlighted the region on information we needed.

            imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
            # cv2.imshow(str(x), imgCrop)  #cropping out the selected portion to be passed to pytesseract

            if r[2] == 'text':
                # print('{} :{}'.format(r[3], pytesseract.image_to_string(imgCrop)))
                # print(f'{r[3]} :{pytesseract.image_to_string(imgCrop)}')
                myData.append(pytesseract.image_to_string(imgCrop))
            if r[2] == 'box':
                imgGray = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)
                imgThresh = cv2.threshold(imgGray, 170, 255, cv2.THRESH_BINARY_INV)[1]
                totalPixels = cv2.countNonZero(imgThresh)
                # print(totalPixels)
                if totalPixels > pixelThreshold:
                    totalPixels = 1;
                else:
                    totalPixels = 0
                # print(f'{r[3]} :{totalPixels}')
                myData.append(totalPixels)
            cv2.putText(imgShow, str(myData[x]), (r[0][0], r[0][1]),
                        cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 0, 255), 4)

        with open('./static/Form2Output.csv', 'a+') as f:
            for data in myData:
                f.write((str(data) + ','))
            f.write('\n')

        imgShow = cv2.resize(imgShow, (w // 2, h // 2))
        cv2.imshow(y + "2", imgShow)

    # cv2.imshow("output", imgQ)
    cv2.waitKey(0)
    op1 = "Processing Completed go to See Data For Output"
    return render_template('AdmissionForm.html',frm1d=op1)


@app.route('/CustomizeForm', methods=['POST'])
def form3():

    path = './static/QueryImageCustom/Query.jpeg'

    def mousePoints(event, x, y, flags, params):
        global counter, point1, point2, counter2, circles, myColor
        if event == cv2.EVENT_LBUTTONDOWN:
            if counter == 0:
                point1 = int(x // scale), int(y // scale);
                counter += 1
                myColor = (random.randint(0, 2) * 200, random.randint(0, 2) * 200, random.randint(0, 2) * 200)
            elif counter == 1:
                point2 = int(x // scale), int(y // scale)

                type = input('Enter Type')
                name = input('Enter Name ')
                myPoints.append([point1, point2, type, name])
                counter = 0
            circles.append([x, y, myColor])
            counter2 += 1

    img = cv2.imread(path)
    img = cv2.resize(img, (0, 0), None, scale, scale)

    while True:
        # To Display points
        for x, y, color in circles:
            cv2.circle(img, (x, y), 3, color, cv2.FILLED)
        cv2.imshow("Original Image ", img)
        cv2.setMouseCallback("Original Image ", mousePoints)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            #print(myPoints)
            break

    per = 25
    pixelThreshold = 300

    roi = myPoints

    pytesseract.pytesseract.tesseract_cmd = 'D:\\New Volume\\OpenCV\\Tesseract-OCR\\tesseract.exe'

    imgQ = cv2.imread('./static/QueryImageCustom/Query.jpeg')
    h, w, c = imgQ.shape
    # imgQ = cv2.resize(imgQ,(w//3,h//3))

    orb = cv2.ORB_create(1000)
    kp1, des1 = orb.detectAndCompute(imgQ, None)
    imgkp1 = cv2.drawKeypoints(imgQ,kp1,None)
    cv2.imshow("keypoints", imgkp1)

    path = './static/UserFormCustom'
    mypicList = os.listdir(path)
    #print(mypicList)

    for j, y in enumerate(mypicList):
        img = cv2.imread(path + "/" + y)
        # cv2.imshow(y, img) #all the sample forms
        kp2, des2 = orb.detectAndCompute(img, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.match(des2, des1)
        matches.sort(key=lambda x: x.distance)
        good = matches[:int(len(matches) * (per / 100))]
        imgMatch = cv2.drawMatches(img, kp2, imgQ, kp1, good[:100], None, flags=2)
        #cv2.imshow(y, imgMatch) #shows features on the form

        srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
        imgScan = cv2.warpPerspective(img, M, (w, h))  # straightning images however our forms are straight for now
        # cv2.imshow(y,imgScan)

        imgShow = imgScan.copy()
        imgMask = np.zeros_like(imgShow)

        myData = []
        #print(f'################## Extracting Data from Form {j}  ##################')

        for x, r in enumerate(roi):
            cv2.rectangle(imgMask, (r[0][0], r[0][1]), (r[1][0], r[1][1]), (0, 255, 0), cv2.FILLED)
            imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1,
                                      0)  # created a mask and highlighted the region on information we needed.

            imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
            # cv2.imshow(str(x), imgCrop)  #cropping out the selected portion to be passed to pytesseract

            if r[2] == 'text':
                #print('{} :{}'.format(r[3], pytesseract.image_to_string(imgCrop)))
                # print(f'{r[3]} :{pytesseract.image_to_string(imgCrop)}')
                myData.append(pytesseract.image_to_string(imgCrop))
            if r[2] == 'box':
                imgGray = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)
                imgThresh = cv2.threshold(imgGray, 170, 255, cv2.THRESH_BINARY_INV)[1]
                totalPixels = cv2.countNonZero(imgThresh)
                # print(totalPixels)
                if totalPixels > pixelThreshold:
                    totalPixels = 1;
                else:
                    totalPixels = 0
                #print(f'{r[3]} :{totalPixels}')
                myData.append(totalPixels)
            cv2.putText(imgShow, str(myData[x]), (r[0][0], r[0][1]),
                        cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 0, 255), 4)

        with open('./static/DataOutputCustom.csv', 'a+') as f:
            for data in myData:
                #print(data)
                f.write((str(data) + ','))
            f.write('\n')

        imgShow = cv2.resize(imgShow, (w // 2, h // 2))
        cv2.imshow(y + "2", imgShow)

    # cv2.imshow("output", imgQ)
    cv2.waitKey(0)

    op1 = "Processing Completed go to See Data For Output"
    return render_template('CustomizeForm.html', frm1d=op1)

app.run(debug=True)