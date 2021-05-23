class VOC2012(object):
    def __init__(self):
        self.label = {
            0: "aeroplane",
            1: "bicycle",
            2: "bird",
            3: "boat",
            4: "bottle",
            5: "bus",
            6: "car",
            7: "cat",
            8: "chair",
            9: "cow",
            10: "diningtable",
            11: "dog",
            12: "horse",
            13: "motorbike",
            14: "person",
            15: "pottedplant",
            16: "sheep",
            17: "sofa",
            18: "train",
            19: "tvmonitor",
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
