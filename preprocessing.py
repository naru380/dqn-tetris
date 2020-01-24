from PIL import Image
import torchvision.transforms as T

class Prepocessing(object):

    def __init__(self, prepfn, size):
        self.prepfn = prepfn
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Lambda(lambda x: x.convert('L')),
            T.Scale(size, interpolation=T.Image.CUBIC),
            T.ToTensor()])

    def run(self, s):
        return self.transform(self.prepfn(s)).unsqueeze(0)
