class VOC2012(object):
    def __init__(self):
        self.label = {
            0: "background",
            1: "aeroplane",
            2: "bicycle",
            3: "bird",
            4: "boat",
            5: "bottle",
            6: "bus",
            7: "car",
            8: "cat",
            9: "chair",
            10: "cow",
            11: "diningtable",
            12: "dog",
            13: "horse",
            14: "motorbike",
            15: "person",
            16: "pottedplant",
            17: "sheep",
            18: "sofa",
            19: "train",
            20: "tvmonitor",
        }

        self.inv_label = {value: key for (key, value) in self.label.items()}
        self.num_classes = len(self.label.keys())

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.inv_label[key]
        elif isinstance(key, int):
            return self.label[key]

    def __len__(self):
        return self.num_classes
