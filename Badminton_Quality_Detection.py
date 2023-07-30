import cv2
import numpy as np
import matplotlib.pyplot as plt

def main(fileName):
    inputBGRImg = cv2.imread(fileName)
    inputRGBImg= cv2.cvtColor(inputBGRImg, cv2.COLOR_BGR2RGB)
    inputGrayImg = cv2.cvtColor(inputBGRImg, cv2.COLOR_BGR2GRAY)
    inputGrayArr = inputGrayImg.astype(np.float64)

    def segThre(img, T):
        imgArr = img.astype(np.float64)
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                if imgArr[i][j] <= T:
                    imgArr[i][j] = 0
                if imgArr[i][j] > T:
                    imgArr[i][j] = 255
        img = imgArr.astype(np.uint8)
        return img
    def centerCor(img):
        #get the image only shows the yellow part of the image
        def detectYellow(image):
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            lower_yellow = np.array([20, 150, 100])
            upper_yellow = np.array([30, 255, 255])

            yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

            yellow_detected = cv2.bitwise_and(image, image, mask=yellow_mask)
            yellowImg = cv2.cvtColor(yellow_detected, cv2.COLOR_BGR2RGB)
            return yellowImg
        yellowOp = detectYellow(img) #yellow output with black and yellow
        yellowIntOp = cv2.cvtColor(yellowOp, cv2.COLOR_BGR2GRAY) #yellow intensity output wih gray and yellow
        yellowInt2BOp = segThre(yellowIntOp, 20) #pure white and black image showing yellow intensity

        dilatekernel = np.ones((70, 70), np.uint8)

        dilateOp = cv2.dilate(yellowInt2BOp, dilatekernel, 1)
        erodeOp = cv2.erode(dilateOp, dilatekernel, 1)

        erodeOpArr = erodeOp.astype(np.float64)
        xAverage = 0 #restore sum first, then divide count
        yAverage = 0
        count = 0
        
        for i in range(erodeOpArr.shape[0]):
            for j in range(erodeOpArr.shape[1]):
                if erodeOpArr[i][j] == 255:
                    xAverage += i
                    yAverage += j
                    count += 1
        
        xAverage = int(xAverage / count)
        yAverage = int(yAverage / count)

        return xAverage, yAverage

    xCenter, yCenter = centerCor(inputBGRImg)

    centerCheck = inputGrayImg
    centerCheck[xCenter - 10: xCenter + 10, yCenter - 10: yCenter + 10] = 0


    def toPolarImg(linearImg, xCenter, yCenter):
        linearArr = linearImg.astype(np.float64)
        maxSize = np.max(np.array([xCenter, linearArr.shape[0] - xCenter, yCenter, linearArr.shape[0] - yCenter]))
        polarArr = np.zeros((3600, maxSize)) #polarArr[angle][radius]
        for i in range(linearArr.shape[0]):
            for j in range(linearArr.shape[1]):
                radius = int(np.sqrt((i - xCenter) ** 2 + (j - yCenter) ** 2))
                if radius < maxSize:

                    if j == yCenter:
                        angle = 900
                    else:
                        angle = int(np.degrees(np.arctan((xCenter - i) / (j - yCenter))) * 10)
                    if j < yCenter:
                        angle += 1800
                    if j > yCenter and i > xCenter:
                        angle += 3600
                    
                    if angle == 3600:
                        angle -= 1          
                        
                    polarArr[angle][radius] = linearArr[i][j]

        polarImg= polarArr.astype(np.uint8)
        return polarImg

    polImg = toPolarImg(inputGrayImg, xCenter, yCenter)

    _, segImg = cv2.threshold(polImg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    morKernel = np.ones((5,5), np.uint8)
    segImg = cv2.dilate(segImg, morKernel, 1)
    segImg = cv2.erode(segImg, morKernel, 1)


    morKernel = np.ones((50, 50), np.uint8)
    segConnectImg = cv2.dilate(segImg, morKernel, 1)
    segConnectImg = cv2.erode(segImg, morKernel, 1)


    def det2Point(polImg):
        #transfer polar image into histogram showing how many pixels in each radius

        #intput arr[i][j], return output[j] about how many pixels on the jth col
        def count_i_In_j(arr):
            output = np.zeros(arr.shape[1])
            for j in range(arr.shape[1]):
                for i in range(arr.shape[0]):
                    if arr[i][j] == 0:
                        output[j] += 1
            return output

        #pixelPerRadius[i] counts at the i distance, how many pixels are white
        pixelPerRadius = count_i_In_j(polImg.astype(np.float64))

        def gauFilt1d(arr, sigma):
            size = int(6 * sigma) + 1
            kernel = np.exp(-0.5 * (np.arange(-size, size + 1) / sigma)**2)
            kernel /= np.sum(kernel)  # Normalize the kernel
            return np.convolve(arr, kernel, mode='same')

        pixelPerRadiusGau = gauFilt1d(pixelPerRadius, 25)


        pPRGG = np.gradient(pixelPerRadiusGau) #pixelPerRadiusGauGra
        for i in range(pPRGG.shape[0]):
            pPRGG[i] = int(pPRGG[i])


        #some debug operations to show the plot
        # plt.plot(np.arange(pixelPerRadiusGau.shape[0]), np.gradient(pixelPerRadiusGau))

        def detPoint(pixelPerRadius, minVal, maxVal,testNumber=5, risk=0.2):
            # the intend range is [minVal, maxVal)
            # risk means the affordable risk of not eligible gradients
            # testNumber means to test how many continuous times to make sure its property
            gradient = np.gradient(pixelPerRadius)
            for i in range(pixelPerRadius.shape[0] - testNumber + 1):
                riskCount = 0
                for j in range(testNumber):
                    if gradient[i + j] < minVal or gradient[i + j] >= maxVal:
                        riskCount += 1
                    if riskCount > risk * testNumber:
                        break
                    if j == testNumber - 1:
                        return i
            print('Something wrong with the point detection.')
            return -1
        
        startDown = detPoint(pixelPerRadiusGau[: ], -100, -1, 5) 
        if startDown == -1:
            print('startDown detection failure')
            return 0, 0  
        # print('startDown = ' + str(startDown))


        headB = detPoint(pixelPerRadiusGau[startDown: ], -0.5, 2, 5) + startDown    
        if headB == -1 + startDown:
            print('headB detection failure')
            return 0, 0
        # print('headB = ' + str(headB))

        headE = detPoint(pixelPerRadiusGau[headB: ], 0.5, 100, 5) + headB    
        if headE == -1 + headB:
            print('headE detection failure')
            return 0, 0,

        stemDown = detPoint(pixelPerRadiusGau[headE: ], -10, -1, 3) + headE    
        if stemDown == -1:
            print('stemDown detection failure')
            return 0, 0

        stemE = detPoint(pixelPerRadiusGau[stemDown: ], -0.5, 2, 5) + stemDown
        if stemE == -1 + stemDown:
            print('stemE detection failure')
            return 0, 0
        
        return headE, stemE


    headE, stemE = det2Point(segConnectImg)

    det2PointOp = np.zeros_like(segConnectImg)
    det2PointOp[:, :] = segConnectImg[:, :]
    det2PointOp[:, headE - 3: headE + 3] = 0
    det2PointOp[:, stemE - 3: stemE + 3] = 0


    def detStemRow(segImgConnect, whiteRowRate=0.8, steMinRow=5, stemNum=16):
        # whiteRowRate, if a row has more than this rate of white pixels, we view this row mainly white
        # steMinRow, stem minuimn row: a stem should have at least 'some' rows
        # stemNum = 16,how many feather, aka how many stems a badminton has
        
        stemSegImg = segImgConnect[: , headE: stemE]
        stemSegArr = stemSegImg.astype(np.float64)

        contRowNumSum = 0 #continuous row number sum
        contRowAmount = 0 #continuous row number amount: how many rows have been mainly white?
        #e.g. if row 101 ~ 119 are mainly white, then from 101,  contRowNumSum = 101 + 102 + ..., contRowAmount = 19, then the output we need is Sum / Amount

        stemRow = np.zeros(stemNum) # which row is the center of a single stem
        estimatedStemNum = 0
        for i in range(stemSegArr.shape[0]):
            count = 0 #count how many white pixels in a row
            for j in range(stemSegArr.shape[1]): 
                if stemSegArr[i][j] == 255:
                    count += 1
            count = count / stemSegArr.shape[1] #use count to restore the rate of white pixels in a row
            if count >= whiteRowRate: #when a row is detected as mainly white row
                contRowNumSum += i
                contRowAmount += 1
            else:
                if contRowAmount >= steMinRow:
                    if estimatedStemNum >= stemNum:
                        print('more stems than expected')
                    else:
                        stemRow[estimatedStemNum] = int(contRowNumSum / contRowAmount)
                        estimatedStemNum += 1
                contRowNumSum = 0
                contRowAmount = 0
        return stemRow

    stemRow = detStemRow(segConnectImg)
    # print(stemRow)

    stemRowOp = np.zeros_like(polImg)
    stemRowOp[:, :] = polImg.astype(np.float64)
    for i in range(stemRow.shape[0]):
        stemRowOp[int(stemRow[i]) - 3: int(stemRow[i]) + 3, : ] = 0

    def orderByStem(polImg,stemRow):

        startStemRow = int(stemRow[0])
        polArr = polImg.astype(np.float64)
        polArrOp = np.zeros_like(polArr)
        polArrOp[ : -startStemRow] = polArr[startStemRow:]
        polArrOp[-startStemRow:] = polArr[: startStemRow]
        
        stemRowOp = np.zeros_like(stemRow)
        for i in range(stemRow.shape[0]):
            stemRowOp[i] = stemRow[i] - stemRow[0]
        polImgOp = polArrOp.astype(np.uint8)

        return polImgOp, stemRowOp


    segConnectOrdImg, stemRowOrd = orderByStem(segConnectImg, stemRow)
    ordImg, _ = orderByStem(polImg, stemRow)
    segOrdImg, _ = orderByStem(segImg, stemRow)

    ordImgOp = np.zeros_like(ordImg)
    ordImgOp[:, :] = ordImg[:, :]
    for i in range(1, stemRowOrd.shape[0]):
        ordImgOp[int(stemRowOrd[i]) - 5: int(stemRowOrd[i]) + 5, :] = 0

    def knife(segConnectOrdImg, stemRowOrd):
        knifeArr = np.zeros((16, 2))

        for i in range(16):
            if i == 15:
                feaSegConnectedOrdImg = np.zeros_like(segConnectOrdImg[int(stemRowOrd[i]):])
                feaSegConnectedOrdImg[:, :] = segConnectOrdImg[int(stemRowOrd[i]):]    
            else:   
                feaSegConnectedOrdImg = np.zeros_like(segConnectOrdImg[int(stemRowOrd[i]): int(stemRowOrd[i + 1])])
                feaSegConnectedOrdImg[:, :] = segConnectOrdImg[int(stemRowOrd[i]): int(stemRowOrd[i + 1])]


            knifeArr[i, 0], knifeArr[i, 1] = det2Point(feaSegConnectedOrdImg)

            if knifeArr[i, 0] == 0 or knifeArr[0, 1] == 0:
                knifeArr[i, 0] = headE
                knifeArr[i, 1] = stemE
        
        return knifeArr

    knifeArr = knife(segConnectOrdImg, stemRowOrd)
    knifeImgOp = np.zeros_like(ordImgOp)
    knifeImgOp[: ] = ordImgOp[:,:]
    for i in range(16):
        if i == 15:
            knifeImgOp[int(stemRowOrd[i]):, int(knifeArr[i, 0]) - 3: int(knifeArr[i, 0]) + 3 ] = 0
            knifeImgOp[int(stemRowOrd[i]):, int(knifeArr[i, 1]) - 3: int(knifeArr[i, 1]) + 3 ] = 0
        else: 
            knifeImgOp[int(stemRowOrd[i]): int(stemRowOrd[i + 1]), int(knifeArr[i, 0]) - 3: int(knifeArr[i, 0]) + 3 ] = 0
            knifeImgOp[int(stemRowOrd[i]): int(stemRowOrd[i + 1]), int(knifeArr[i, 1]) - 3: int(knifeArr[i, 1]) + 3 ] = 0
    def judgeFeather(img, segImg, stemRowOrd, knifeArr):
        row = np.zeros(17)
        row[: 16]= stemRowOrd[:]
        row[16] = img.shape[0]
        col = np.zeros_like(knifeArr)
        col[:, :] = knifeArr[:, :]

        #clean the black part in the seg image
        morKernel = np.ones((60, 60), np.uint8)
        segCleanImg = np.zeros_like(segImg)
        segCleanImg[:, :] = segImg[:, :]
        segCleanImg = cv2.dilate(segCleanImg, morKernel, 1)
        # segCleanImg = cv2.erode(segCleanImg, morKernel, 1)
        # plt.imshow(segCleanImg, 'gray')

        edgeRate = np.zeros(16)
        blankRate = np.zeros(16)
        missingRate = np.zeros(16)

        for i in range(16):

            totalArea = 0
            edgeArea = 0
            blankArea = 0
            missingArea =0

            feaSegCleanImg = np.zeros_like(img[int(row[i]): int(row[i + 1]), int(col[i][1]):], np.uint8)
            feaSegCleanImg[:, :] = segCleanImg[int(row[i]): int(row[i + 1]), int(col[i][1]):]
            # plt.imshow(feaSegCleanImg, 'gray')
            judgeArr = np.zeros_like(feaSegCleanImg)
            judgeArr[:, :] = feaSegCleanImg / 255

            feaImg = np.zeros_like(img[int(row[i]): int(row[i + 1]), int(col[i][1]):], np.uint8)
            feaImg[:, :] = img[int(row[i]): int(row[i + 1]), int(col[i][1]):]
            morKernel = np.ones((2, 2), np.uint8)
            feaImg = cv2.dilate(feaImg,morKernel, 1)
            feaImg = cv2.erode(feaImg, morKernel, 1)

            feaEdgImg = np.zeros_like(feaImg, np.uint8)
            feaEdgImg[:, :] = feaImg[:, :]
            feaEdgImg = cv2.Canny(feaEdgImg, 40, 120)
            # plt.imshow(feaEdgImg, 'gray')

            feaSegImg = np.zeros_like(feaImg, np.uint8)
            feaSegImg[:, :] = feaImg[:, :]
            _, feaSegImg = cv2.threshold(feaSegImg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # plt.imshow(feaSegImg, 'gray')


            for s in range(feaEdgImg.shape[0]):
                for t in range(feaEdgImg.shape[1]):
                    if judgeArr[s][t] == 1:
                        totalArea += 1
                    if feaEdgImg[s][t] * judgeArr[s][t] > 0:
                        edgeArea += 1
                    if feaSegImg[s][t] * judgeArr[s][t] > 0:
                        blankArea += 1
                    if feaSegImg[s][t] == 0:
                        missingArea += 1
            edgeRate[i] = edgeArea / totalArea
            blankRate[i] = 1 - blankArea / totalArea
            missingRate[i] = missingArea / (feaImg.shape[0] * feaImg.shape[1])

        return edgeRate, blankRate, missingRate
        

    edgeRate = np.zeros(16)
    blankRate = np.zeros(16)
    missingRate = np.zeros(16)

    edgeRate, blankRate, missingRate = judgeFeather(ordImg, segOrdImg, stemRowOrd, knifeArr)

    def conclusion(edgeRate, blankRate, missingRate):
        #1 new, 2 used, 3 broken
        def former(i):
            if i == 0:
                return 15
            else:
                return i
        
        def later(i):
            if i == 15:
                return 0
            else:
                return i

        newStates = True
        brokenStatus = False

        missingFeather = 0
        missingFeatherIndex = 0
        for i in range(16):
            if missingRate[i] > 0.65:
                missingFeather += 1
                missingFeatherIndex = i
        if missingFeather > 0:
            print('not new because there is an missing feather')
            if missingRate[former(missingFeatherIndex)] > 0.40 or missingRate[later(missingFeatherIndex)] > 0.4:
                return 3
        elif missingFeather > 1:
            print('can not be used because there are more than 1 feather.')
            return 3
        
        

        
        if np.average(edgeRate) > 0.013 :
            print('not new because there are some edges')
            newStates = False
        if np.average(edgeRate) > 0.03:
            print('can not be used because there are too many edges')
            return 3

        if np.average(blankRate) > 0.09:
            print('not new because too there are some blanks')
            newStates = False
        if np.average(blankRate) > 0.125:
            print('can not be used because there are too many blanks')
            brokenStatus = True
        
        if brokenStatus:
            return 3
        if newStates:
            return 1

        return 2
                
    Conclusion = conclusion(edgeRate, blankRate, missingRate)


    def drawConclusion(conclusion):
        if conclusion == 1:
            print(fileName + ' is new or almost new')
        elif conclusion == 2:
            print(fileName + ' is slightly damaged but can still be used')
        elif conclusion == 3:
            print(fileName + ' can not be used')

    drawConclusion(Conclusion)


if __name__ == "__main__":
    fileName = 'badminton14.jpg'
    main(fileName)