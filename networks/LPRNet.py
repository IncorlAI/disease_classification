import torch.nn as nn
import torch


def sparse_tuple_for_ctc(T_length, lengths):
    input_lengths = []
    target_lengths = []
    for ch in lengths:
        input_lengths.append(T_length)
        target_lengths.append(ch)
    return tuple(input_lengths), tuple(target_lengths)
    

class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)


class LPRNet(nn.Module):
    def __init__(self, class_num, dropout_rate=0.5):
        super(LPRNet, self).__init__()
        self.class_num = class_num
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),  # 0
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),  # 2
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            small_basic_block(ch_in=64, ch_out=128),  # *** 4 ***
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),  # 6
            nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 2, 1)), # 7
            small_basic_block(ch_in=64, ch_out=256),  # 8
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 10
            small_basic_block(ch_in=256, ch_out=256),  # *** 11 ***
            nn.BatchNorm2d(num_features=256),  # 12
            nn.ReLU(),  # 13
            nn.MaxPool3d(kernel_size=(4, 3, 3), stride=(4, 2, 1)),  # 14
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(2, 1), stride=(2, 1)),  # 16
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 18
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(1, 13), stride=(1, 2)),  # 20
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU(),  # *** 22 ***
        )
        self.container = nn.Sequential(
            nn.Conv2d(
                in_channels=256 + class_num + 128 + 64, out_channels=self.class_num, kernel_size=(1, 1), stride=(1, 1)
                # in_channels=68, out_channels=self.class_num, kernel_size=(1, 1), stride=(1, 1)
            ),
            # nn.BatchNorm2d(num_features=self.class_num),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=self.class_num, out_channe ls=self.lpr_max_len+1, kernel_size=3, stride=2),
            # nn.ReLU(),
        )

    def forward(self, x):
        keep_features = list()
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [2, 6, 13, 22]: # [2, 4, 8, 11, 22]
                keep_features.append(x)
        _, _, last_h, last_w = x.shape
        global_context = list()
        for i, f in enumerate(keep_features):
            b, c, h, w = f.shape
            f = nn.AvgPool2d(kernel_size=(h // last_h, w - last_w + 1), stride=(h // last_h, 1))(f)
            f_pow = torch.pow(f, 2)
            f_mean = torch.mean(f_pow)
            f = torch.div(f, f_mean)
            global_context.append(f)

        x = torch.cat(global_context, 1)
        logits = self.container(x)
        # logits = torch.mean(x, dim=2)
        b, c, h, w = logits.shape
        logits = logits.reshape(b, c, h*w)
        
        return logits
