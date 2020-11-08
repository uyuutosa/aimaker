import json
import cv2
from numpy import *

class AnnParser:
    def __init__(self, ann_path, img):
        o = open(ann_path)
        self._parseJSON(json.load(o), img)        
    
    def _parseJSON(self, obj, img):
        
        line_lst = []
        point_lst = []
        segment_lst = []

        for d in obj["objects"]:
            name = d["classTitle"]
            number = int(name.split("_")[0])
            if number // 100 == 1:
                line_lst += [d]    
            elif number // 100 == 2:
                segment_lst += [d]
            else:
                point_lst += [d]
        
        self.lh = LineHandler(line_lst)
        self.sh = SegmentHandler(segment_lst, img)
        self.ph = PointHandler(point_lst)
        
    def putParams(self, param_lst, handler_type):
        handler = {"segment":self.sh,
                   "line":self.lh,
                   "point":self.ph}[handler_type]
        lst = []
        for param in param_lst:
            index = handler.checkExist(param)
            if index is not None:
                lst += [handler[index]]
        return lst
    
    def __getitem__(self, param_dic):
        ret_dic = {}
        for k, v in param_dic.items():
            if k == 'segment':
                if v == "whole_body":
                    ret_dic['segment']  = self.putParams(self.sh.name_lst, 'segment')
                else:
                    ret_dic['segment']  = self.putParams(v, 'segment')
            elif k == 'line':
                print(self.lh.name_lst)
                if v == "whole_line":
                    ret_dic['line']  = self.putParams(self.lh.name_lst, 'line')
                else:
                    ret_dic['line']  = self.putParams(v, 'line')
            elif k == 'point':
                if v == "whole_point":
                    ret_dic['point']  = self.putParams(self.ph.name_lst, 'point')
                else:
                    ret_dic['point']  = self.putParams(v, 'point')
            else:
                raise ValueError("Invalid data type {}".format(k))
        return ret_dic


class LineHandler:
    def __init__(self, line_lst):
        self.line_lst = line_lst
        self.name_lst = self._getNameList()
            
    def _getNameList(self):
        name_lst = []
        for i in self.line_lst:
            name_lst += [i["classTitle"]]
        return name_lst
    
    def checkExist(self, name):
        if name in self.name_lst:
            return self.name_lst.index(name)
        else:
            return None
    
    def __getitem__(self, index):
        return self.line_lst[index]["points"]['exterior']
    
    def __len__(self):
        return len(self.line_lst)
    
class PointHandler:
    def __init__(self, point_lst):
        self.point_lst = point_lst
        self.name_lst = self._getNameList()

    def _getNameList(self):
        name_lst = []
        for i in self.point_lst:
            name_lst += [i["classTitle"]]
        return name_lst

    def checkExist(self, name):
        if name in self.name_lst:
            return self.name_lst.index(name)
        else:
            return None

        
    def __getitem__(self, index):
        return self.point_lst[index]["points"]['exterior'][0]
    
    def __len__(self):
        return len(self.point_lst)
    
class SegmentHandler:
    def __init__(self, segment_lst, img):
        self.segment_lst = segment_lst
        self.map = zeros_like(img, dtype=uint8)#[...,0]
        self.img = img
        self.name_lst = self._getNameList()
            
    def _getNameList(self):
        name_lst = []
        for i in self.segment_lst:
            name_lst += [i["classTitle"]]
        return name_lst
    
    def checkExist(self, name):
        if name in self.name_lst:
            return self.name_lst.index(name)
        else:
            return None

    
    def __getitem__(self, index):
        point_lst = [array(self.segment_lst[index]['points']['exterior']).astype(int).reshape(-1, 1, 2)]
        return cv2.drawContours(self.map.copy(), point_lst, -1, (255,255,255), -1)[...,0]
        
    def __len__(self):
        return len(self.segment_lst)
    

