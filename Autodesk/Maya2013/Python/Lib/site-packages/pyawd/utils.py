class AWDMatrix4x4:
    def __init__(self, raw_data=None):
        self.raw_data = raw_data

        if self.raw_data is None:
            self.raw_data = [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]

class AWDMatrix2x3:
    def __init__(self, raw_data=None):
        self.raw_data = raw_data

        if self.raw_data is None:
            self.raw_data = [1,0,0,1,0,0]
