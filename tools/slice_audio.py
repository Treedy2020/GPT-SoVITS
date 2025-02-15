import os
import traceback
from typing import Optional

import numpy as np
from scipy.io import wavfile
from slicer2 import Slicer

# parent_directory = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(parent_directory)
from utils import load_audio
from tqdm import tqdm


def slice(
    ipt,
    output_dir,
    threshold,
    min_length,
    min_interval,
    hop_size,
    max_sil_kept,
    _max,
    alpha,
    i_part: Optional[int] = None,
    all_part: Optional[int] = None,
):
    os.makedirs(output_dir, exist_ok=True)
    if os.path.isfile(ipt):
        ipt = [ipt]
    elif os.path.isdir(ipt):
        ipt = [os.path.join(ipt, name) for name in sorted(list(os.listdir(ipt)))]
    else:
        return "输入路径存在但既不是文件也不是文件夹"
    slicer = Slicer(
        sr=32000,  # 长音频采样率
        threshold=int(threshold),  # 音量小于这个值视作静音的备选切割点
        min_length=int(
            min_length
        ),  # 每段最小多长，如果第一段太短一直和后面段连起来直到超过这个值
        min_interval=int(min_interval),  # 最短切割间隔
        hop_size=int(
            hop_size
        ),  # 怎么算音量曲线，越小精度越大计算量越高（不是精度越大效果越好）
        max_sil_kept=int(max_sil_kept),  # 切完后静音最多留多长
    )
    _max = float(_max)
    alpha = float(alpha)
    i_part = 0 if i_part is None else i_part
    all_part = len(ipt) if all_part is None else all_part
    for inp_path in tqdm(ipt[int(i_part) :: int(all_part)]):
        try:
            name = os.path.basename(inp_path)
            audio = load_audio(inp_path, 32000)
            for chunk, start, end in slicer.slice(audio):  # start和end是帧数
                tmp_max = np.abs(chunk).max()
                if tmp_max > 1:
                    chunk /= tmp_max
                chunk = (chunk / tmp_max * (_max * alpha)) + (1 - alpha) * chunk
                wavfile.write(
                    "%s/%s_%010d_%010d.wav" % (output_dir, name, start, end),
                    32000,
                    (chunk * 32767).astype(np.int16),
                )
        except Exception as e:
            print(inp_path, "->fail->", traceback.format_exc())
    return "执行完毕，请检查输出文件"

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i",
                        type=str,
                        required=True,
                          help="The input audio file or directory.")
    parser.add_argument(
        "--output_dir", "-o",
        required=True,
        type=str, help="Output directory of the sliced audio clips"
    )
    parser.add_argument(
        "--db_thresh",
        type=float,
        required=False,
        default=-34.,
        help="The dB threshold for silence detection, default as -34.",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        required=False,
        default=3000,
        help="The minimum milliseconds required for each sliced audio clip, default as 3000.",
    )
    parser.add_argument(
        "--min_interval",
        type=int,
        required=False,
        default=300,
        help="The minimum milliseconds for a silence part to be sliced, defalt as 300.",
    )
    parser.add_argument(
        "--hop_size",
        type=int,
        required=False,
        default=10,
        help="FO hop size, the smaller the value, the higher the accuracy, default as 10.",
    )
    parser.add_argument(
        "--max_sil_kept",
        type=int,
        required=False,
        default=500,
        help="The maximum silence milliseconds kept in the sliced clip, default as 500.",
    )
    parser.add_argument(
        "--normalized_loudness_multiplier",
        type=float,
        required=False,
        default=0.9,
        help="The maximum value of the audio clip, defualt as 0.9 .",
    )
    parser.add_argument(
        "--alpha_mix",
        type=float,
        required=False,
        default=0.25,
        help="proportion of normalized audio merged into dataset, default as 0.25.",
    )
    parser.add_argument(
        "--n_threads",
        type=int,
        required=False,
        default=4,
        help="CPU threads used for audio slicing, default as 4."
    )

    args = parser.parse_args()
    slice(
        ipt=args.input,
        output_dir=args.output_dir,
        threshold=args.db_thresh,
        min_length=args.min_length,
        min_interval=args.min_interval,
        hop_size=args.hop_size,
        max_sil_kept=args.max_sil_kept,
        _max=args.normalized_loudness_multiplier,
        alpha=args.alpha_mix
    )



