import aimaker.utils.image_writer as iw

class ImageWriterFactory:
    def __init__(self, config):
        self.config = config
        self.image_writer_dic = {"movie" : iw.VideoImageWriter,
                                 "image" : iw.FileImageWriter,
                                 "socket": iw.SocketImageWriter
                                }
        self.support_movie_extension_lst = ['mp4', 'avi']

    def create(self, dump_path, fps=None, size=None):
        name = self._parseImageWriter(dump_path)
        print("image writer name is {}".format(name))
        self._is_exist(name)
        if name == 'movie':
            return self.image_writer_dic[name](self.config, dump_path, fps, size)
        elif name == 'socket':
            return self.image_writer_dic[name](self.config, dump_path, fps, size)
        else:
            return self.image_writer_dic[name](self.config, dump_path)
    
    def _parseImageWriter(self, dump_path):
        if dump_path.split('.')[-1] in self.support_movie_extension_lst:
            return 'movie'
        elif dump_path.count('.') == 3 or not len(dump_path):
            return 'socket'
        else:
            return 'image'


    def _is_exist(self, name):
        if not name in self.image_writer_dic:
            raise NotImplementedError(('{} is wrong key word for ' + \
                                       '{}. choose {}')\
                                   .format(name, self.__class__.__name__, self.loss_dic.keys()))
