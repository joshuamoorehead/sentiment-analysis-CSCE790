import os
import cv2
import time
import threading


class CaptureVideo:

    def __init__(self, save_dir='./test', DEBUG=True):
        self.DEBUG = DEBUG
        self.CAM_IDX = 0

        self.filename = 'test.png'
        self.save_dir = save_dir
        self.inner_save_dir = '/vid/'

    def capture(self):
        print('Please look into the camera...')
        cap = cv2.VideoCapture(self.CAM_IDX)
        frames = []
        filenames = []
        for i in range(10):
            if self.DEBUG:
                print('capturing frame', i)
            ret, frame = cap.read()
            frames.append(frame)
            filenames.append(str(i) + '_' + self.filename)
            time.sleep(2)
        #cv2.imshow('cam', frame)
        self.save_dir = self.save_dir + self.inner_save_dir
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)

        for i in range(len(frames)):
            file_path = self.save_dir + filenames[i]
            cv2.imwrite(file_path, frames[i])
        #cap.release()
        #cv2.destroyAllWindows()
        return 0

    def record_video(self):
        ret = self.capture()

    def start(self, filename):
        self.filename=filename
        t = threading.Thread(target=self.record_video)
        t.start()
        #self.record_video()


if __name__ == "__main__":
    pass
