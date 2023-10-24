import OmegaConf
from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument('--warp_checkpoint', type=str, default='checkpoints/PFAFN/warp_model_final.pth', help='load the pretrained model from the specified location')
        self.parser.add_argument('--gen_checkpoint', type=str, default='checkpoints/PFAFN/gen_model_final.pth', help='load the pretrained model from the specified location')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--unpaired', action='store_true', help='if enables, uses unpaired data from dataset')
        self.isTrain = False


class NonCmdOptions(object):
    def __init__(self):
        args = OmegaConf.create()

        args.warp_checkpoint = 'checkpoints/PFAFN/warp_model_final.pth'  # will be replaced
        args.gen_checkpoint = 'checkpoints/PFAFN/gen_model_final.pth'  # not used
        args.phase = 'test'
        args.unpaired = False
        args.fineSize = 512

        self.args = args

    def parse(self, verbose: bool = False):
        return self.args
