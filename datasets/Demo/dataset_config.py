class Configuration(object):
    def __init__(self):
        self.data_path = 'datasets/Demo'
        self.dataloader_worker = 8
        self.height = 960
        self.width = 1280
        self.aug_degrees = (-20, 20)
        self.aug_translate = (.2, .2)
        self.aug_scale = (0.5, 1.5)
        self.padm_default_distance = 75

        self.test_split_threshold = 1000
        self.test_split_depth = 2
        self.decoder_involve_threshold = 500

config = Configuration()