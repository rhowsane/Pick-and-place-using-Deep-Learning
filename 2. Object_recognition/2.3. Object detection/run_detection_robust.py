from typing import Union

import cv2 as cv
import argparse
import sys
import numpy as np
import os.path


class inferr:
    def __init__(self, weights, conf_thresh):
        # Initialize the parameters
        self.confThreshold = conf_thresh # Confidence threshold
        self.nmsThreshold = 0.4 # Non-maximum suppression threshold
        self.inpWidth = 416 # Width of network's input image
        self.inpHeight = 416 # Height of network's input image

        # Parser to run program from command line
        parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
        parser.add_argument('--image', help='Path to image file.')
        parser.add_argument('--video', help='Path to video file.')
        args = parser.parse_args()

        # Load names of classes
        classesFile = 'classes.txt'
        #classesFile = '/home/rhosane/Reconhecimento_de_imagens/3.Deteccao_de_objetos/classes.txt'
        self.classes = None
        with open(classesFile, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

        # Gve the configuration and weight files for the model and load the network using them.
        modelConfiguration = '/home/rhosane/Reconhecimento_de_imagens/2.Treinamento_do_modelo/Test_1/yolov3/yolov3_custom.cfg'
        modelWeights = weights
        print(modelConfiguration)#rho

        print('Antes da leitura da rede')#rho
        self.net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
        #self.net = cv.dnn.readNetFromDarknet(modelWeights, modelConfiguration) #rho
        print('reussite')
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        #self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
        # Run if GPU available
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)

        # Get the names of the output layers
    def getOutputsNames(self, net):
        # Get the names of all the layers in the network
        layersNames = net.getLayerNames()
        # Get the names of the ouput layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0]-1] for i in net.getUnconnectedOutLayers()]

    # Draw the predicted bounding box
    def drawPred(self, classId, conf, left, top, right, bottom):
        # Draw a bounding box
        cv.rectangle(self.frame, (left,top), (right, bottom), (255,178,50), 3)

        label = '%.2f'%conf

        # Get the label for the class name and its confidence
        if self.classes:
            assert (classId < len(self.classes))
            label = '%s:%s' % (self.classes[classId], label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv.rectangle(self.frame, (left, top - round(1.5*labelSize[1])), (left+round(1.5*labelSize[0]), top+baseLine), (255, 255, 255), cv.FILLED)
        cv.putText(self.frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

    # Remove the bounding boxes with low confidence using non-maxima suppression
    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        # Scan through all the bounding boxes output from the network and keep only
        # the ones with high confidence scores. Assign the box's class label as the class with the highest score
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.confThreshold:
                    center_x = int(detection[0]*frameWidth)
                    center_y = int(detection[1]*frameHeight)
                    width = int(detection[2]*frameWidth)
                    height = int(detection[3]*frameHeight)
                    left = int(center_x - width/2)
                    top = int(center_y - height/2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left,top, width, height])


        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        #print(indices)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            self.drawPred(classIds[i], confidences[i], left, top, left+width, top+height)

    def predict(self, cap, init_vid_writer, win_Name, outputFile, image):
        while cv.waitKey(1) < 0:

            # get frame from the video
            hasFrame, self.frame = cap.read()

            # Stop the program if reached end of video
            if not hasFrame:
                print("Done processing !!!")
                print("Output file is stored as ", outputFile)
                cv.waitKey(3000)
                # Release device
                cap.release()
                break

            # Create a 4D blob from a frame.
            blob = cv.dnn.blobFromImage(self.frame, 1 / 255, (self.inpWidth, self.inpHeight), [0, 0, 0], 1, crop=False)

            # Sets the input to the network
            self.net.setInput(blob)

            # Runs the forward pass to get output of the output layers
            outs = self.net.forward(self.getOutputsNames(self.net))

            # Remove the bounding boxes with low confidence
            self.postprocess(self.frame, outs) #rho

            # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
            t, _ = self.net.getPerfProfile()
            label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
            cv.putText(self.frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

            # Write the frame with the detection boxes
            if image == True:
                cv.imwrite(outputFile, self.frame.astype(np.uint8))
            else:
                init_vid_writer.write(self.frame.astype(np.uint8))

            cv.imshow(win_Name, self.frame)

    def image_predict(self, source_path, dest_path):
        self.frame = None
        for root, dirs, files in os.walk(source_path):
            for file in files:
                if file.endswith(".png") or file.endswith('.jpg'):
                    # Get file path
                    print('on file:', file)
                    #file_path_txt: Union[bytes, str] = os.path.join(root, file)
                    file_path_txt = os.path.join(root, file)
                    print('Starting prediction on file', file)

                    # Init window
                    winName = 'Deep learning object detection in OpenCCV'
                    cv.namedWindow(winName, cv.WINDOW_NORMAL)

                    # Edit file extension
                    if '.jpeg' in file:
                        file = file.replace('.jpeg', '')
                    elif '.jpg' in file:
                        file = file.replace('.jpg', '')
                    else:
                        file = file.replace('.png', '')

                    # Get media file
                    cap = cv.VideoCapture(file_path_txt)
                    print('Succesful media retreival')
                    outputname = file + '_yolo_out.png'
                    outputFile = dest_path +'/'+outputname

                    # Run inference
                    vid_writer = None
                    self.predict(cap, vid_writer, winName, outputFile, image=True)
                    print('Succesful inference on file', file)

    def video_predict(self, source_path, dest_path):
        for root, dirs, files in os.walk(source_path):
            for file in files:
                if file.endswith(".mp4"):
                    # Get file path
                    file_path_txt = os.path.join(root, file)
                    print('Starting prediction on file', file)
                    # Init window
                    winName = 'Deep learning object detection in OpenCCV'
                    cv.namedWindow(winName, cv.WINDOW_NORMAL)

                    # Get media file
                    cap = cv.VideoCapture(file_path_txt)
                    print('Succesful media retreival')
                    outputname = file[:-4] + '_yolo_out.mp4'
                    outputFile = dest_path + '/' + outputname

                    # Run inference
                    #vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc(*'LMP4'), 30, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
                    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc(*'FMP4'), 30, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
                    self.predict(cap, vid_writer, winName, outputFile, image=False)
                    print('Succesful inference on file', file)

    def webcam_predict(self, dest_path):
        # Init webcam
        cap = cv.VideoCapture(0)
        outfile = 'webcam_predict_yolo_out.mp4'
        outputFile = dest_path + '/' + outfile

        # Init window
        winName = 'Deep learning object detection in OpenCCV'
        cv.namedWindow(winName, cv.WINDOW_NORMAL)

        # Run inference
        vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc(*'LMP4'), 30,
                                    (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

        # Run inference
        self.predict(cap, vid_writer, winName, outputFile, image=False)

if __name__ == '__main__':
    # Test definition
    test_n = 1

    # Weights location definition
    weights_path = r'/home/rhosane/Reconhecimento_de_imagens/3.Deteccao_de_objetos/Pesos'
            
    weights_name = 'yolov3_custom_best.weights'
    #weights_final_path = weights_path + '/' + weights_name
    #weights_final_path = r'C:/home/rhosane/Reconhecimento_de_imagens/3.Deteccao_de_objetos/Pesos/' \
                        # r'yolov3_custom_best.weights'
    weights_final_path = r'/home/rhosane/Reconhecimento_de_imagens/3.Deteccao_de_objetos/Pesos/yolov3_custom_best.weights'

    # Initialize inferrence class
    confidence_threshold = 0.3
    #confidence_threshold = 0.4
    print('Initalizing inferrence class...')
    inferrence = inferr(weights_final_path, confidence_threshold)


    # Get source and destination file paths
    source_path_image = r'/home/rhosane/Reconhecimento_de_imagens/3.Deteccao_de_objetos/Data_test/Images'
                         # Path to source testing images
    #RHO -uncomment with videos
    source_path_video = r'/home/rhosane/Reconhecimento_de_imagens/3.Deteccao_de_objetos/Data_test/Videos'


    dest_path_image = r'/home/rhosane/Reconhecimento_de_imagens/3.Deteccao_de_objetos/Output/Images_resul' # Path to image output

    #RHO -uncomment with videos
    dest_path_video = r'/home/rhosane/Reconhecimento_de_imagens/3.Deteccao_de_objetos/Output/Videos_resul'



    # Run predictions
    print('Starting prediction on image files')
    inferrence.image_predict(source_path_image, dest_path_image)
    #print('Starting prediction on video files')
    #inferrence.video_predict(source_path_video, dest_path_video)
    #print()
    #inferrence.webcam_predict(dest_path_video)



