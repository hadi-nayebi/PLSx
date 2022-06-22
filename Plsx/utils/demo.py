"""Demo of the wapper class"""


class Training:
    def add_model(self, model):
        self.model = model

        #  train the  model
        self.model.add_encoder()
        self.model.add_decoder()
        self.model.add_classifier()

        self.model.add_devectorizer()

    def train(self, dataset):

        # general training path
        self.model.train_encoder_decoder(dataset)  # update weights for encoder decoder
        # continuity training path
        self.model.train_continuity(dataset)  # update weights for encoder
        if "class labels in input?":
            self.model.train_classifier()  #  updates weights of the encoder and classifier
        self.model.feedforward(dataset)
