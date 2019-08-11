from imageai.Detection import ObjectDetection
import os
import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
execution_path = os.getcwd()
img_name = "test_image\\slotsB.jpg"
image = cv2.imread(img_name)
print("Image shape:" + str(image.shape))
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path, "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
count = 1
count2 = 1

for i in range(0, image.shape[1], 585):
    slot = image[:, i:i + 570, :]
    print("slot A{} image shape: {}".format(count2, slot.shape))
    cv2.imwrite("slots_images\\slotA" + str(count2) + ".jpg", slot)
    count2 += 1
for j in range(5):
    slot1 = cv2.imread("slots_images\\slotA" + str(count) + ".jpg")
    detections = detector.detectObjectsFromImage(
        input_image=os.path.join(execution_path, "slots_images\\slotA" + str(count) + ".jpg"),
        output_image_path=os.path.join(execution_path, "out.jpg"))
    if detections:
        for eachObject in detections:
            print(eachObject["name"], " : ", eachObject["percentage_probability"])
            print(eachObject)

            if eachObject["name"] == "car":
                box = eachObject["box_points"]
                for k in range(len(box)):
                    if box[k] < 0:
                        box[k] = 0
                img = slot1[box[1]:box[3], box[0]:box[2]]
                cv2.imshow("slot A" + str(count), img)
                cv2.waitKey(0)
                haar_face_cascade = cv2.CascadeClassifier('number_plate.xml')
                number_plate = haar_face_cascade.detectMultiScale(img,)
                for (x, y, w, h) in number_plate:
                    orig = cv2.rectangle(img.copy(), (x, y), (x + w, y + h), (0, 255, 0), 2)
                    img2 = img[y:y + h, x:x + w]
                    print("CAR No of slotA{}: {}".format(str(count), str(pytesseract.image_to_string(img2))))
                    cv2.imshow("slot A" + str(count), orig)
                    cv2.waitKey(0)
            if eachObject["name"] == "motorcycle":
                print("motor cycle detected please start alarm for slotA{}".format(count))
            else:
                print("slot A{} is Empty".format(count))
    else:
        print("slot A{} is Empty".format(count))
    count += 1

