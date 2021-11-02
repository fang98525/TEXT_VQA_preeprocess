import numpy
import  cv2
from PIL import Image
import matplotlib.pyplot as plt
boxs=numpy.load("./test_data/000000_box_info.npy",allow_pickle=True)
print(boxs)
# boxes=infodata["bbox"]


#plt读取
image = cv2.imread("./test_data/000000.jpg")
image = cv2.resize(image, (800, 1137))
for box in boxs: #坐标不能有小数
    # print(box)
    startX=int(box[0])
    startY=int(box[1])
    endX=int(box[2])
    endY=int(box[3])

    cv2.rectangle(image, (startX, startY), (endX, endY), color=(0, 255, 0),
                  thickness=2)  # Draw Rectangle with the coordinates
    # cv2.putText(img, pred_cls[index], (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0),
    #             thickness=text_th)  # Write the prediction class

plt.figure(figsize=(5, 8))  # display the LMDB_output image
plt.imshow(image)
plt.xticks([])
plt.yticks([])
plt.show()

# cv2读取
# image=cv2.imread("./test_data/000000.jpg")
# image = cv2.resize(image, (800, 1137))
# img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
# cv2.namedWindow('result',0)
# cv2.resizeWindow('result', 800, 800)
# cv2.imshow("result",image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

