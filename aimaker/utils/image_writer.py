import os 
import PIL.Image as I
import aimaker.utils.util as util
import aimaker.utils.socket_connector as sc
import socket
import pickle
import cv2

class BaseImageWriter:
    def __init__(self, config, dump_path):
        self.config    = config
        self.dump_path = dump_path
        self.sock = sc.SocketConnector(int(config['image writer settings']['port']))

    def initialize(self):
        pass

    def dump(self, image=None):
        pass

    def release(self):
        pass

    def connectAsServer(self):
        self.sock.connectAsServer()

    def connectAsClient(self, host):
        self.sock.connectAsClient(host)

    def sendData(self, image):
        self.sock.sendData(image)
        self.sock.recvData(2**10) #for adjust

    def recvData(self, buf_size=2**10):
        self.dump(self.sock.recvData(buf_size))
        self.sock.sendData('send dummy data for adjust')

    def sendInfo(self):
        pass

    def recvInfo(self):
        pass

class SocketImageWriter(BaseImageWriter):
    def __init__(self, config, dump_path=None, fps=None, size=None):
        super(SocketImageWriter, self).__init__(config, dump_path)
        fps  = fps       if fps is not None else int(config['image writer settings']['fps'])

        self.sock.connectAsClient(dump_path)

    def recvInfo(self):
        info_dic    = recvData()
        self.fps    = info_dic['fps']
        self.width  = info_dic['width']
        self.height = info_dic['height']

    def dump(self, image=None):
        self.sendData(image)


class VideoImageWriter(BaseImageWriter):
    def __init__(self, config, dump_path, fps=None, size=None):
        super(VideoImageWriter, self).__init__(config, dump_path)

        

        self.fps = fps if fps is not None else int(config['image writer settings']['fps'])
        if size is None:
            self.width, self.height = [int(x) for x in config['image writer settings']['videoSize'].split("x")]
        else:
            self.width, self.height = size

        self.is_first = True

    def initialize(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.dump_path, fourcc, self.fps, (self.width, self.height))

    def dump(self, image=None):
        if self.is_first:
            self.height, self.width, c = image.shape
            self.initialize()
            self.is_first = False
        self.out.write(image)

    def release(self):
        self.out.release()

    def __del__(self):
        cv2.destroyAllWindows()

class FileImageWriter(BaseImageWriter):
    def __init__(self, config, dump_path):
        super(FileImageWriter, self).__init__(config, dump_path)
        self.ext = config['image writer settings']['extension']
        if not os.path.isdir(dump_path):
            os.makedirs(dump_path)


    def dump(self, image=None):
        img = I.fromarray(image[...,::-1])
        img.save(util.fcnt(dir=self.dump_path, fname = "image", ext=self.ext))


