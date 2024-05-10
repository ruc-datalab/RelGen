import argparse
import warnings
import pandas as pd

from relgen.utils.enum_type import SynthesisMethod
from relgen.data.metadata import Metadata
from relgen.data.dataset import Dataset
from relgen.synthesizer.arsynthesizer import MADESynthesizer
from relgen.synthesizer.diffusionsynthesizer import RelDDPMSynthesizer
from relgen.evaluator import Evaluator


warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str)
parser.add_argument('--synthesizer', type=str, choices=["made", "relddpm"], default="made")
parser.add_argument('--method', type=str, choices=["single_model", "multi_model"], default="multi_model")
args = parser.parse_args()

data_dir = args.dir
if not data_dir.endswith('/'):
    data_dir += '/'
# 1. 加载元数据
metadata = Metadata()
metadata.load_from_json(data_dir + "metadata.json")
# 2. 加载数据
data = {}
for table in metadata.tables.keys():
    data[table] = pd.read_csv(data_dir + f"{table}.csv")
# 3. 数据转换器
dataset = Dataset(metadata)
dataset.fit(data)
# 4. 数据合成器
synthesis_method_map = {"single_model": SynthesisMethod.SINGLE_MODEL, "multi_model": SynthesisMethod.MULTI_MODEL}
synthesizer_map = {
    "made": MADESynthesizer(dataset, method=synthesis_method_map[args.method]),
    "relddpm": RelDDPMSynthesizer(dataset, method=synthesis_method_map[args.method])
}
synthesizer = synthesizer_map[args.synthesizer]
synthesizer.fit(data)
# 5. 合成数据
sampled_data = synthesizer.sample()
for table_name, table_data in sampled_data.items():
    table_data.to_csv(data_dir + f"synthetic_{table_name}.csv", index=None)

join_data = dataset.join_data(data)
join_sampled_data = dataset.join_data(sampled_data)
if len(metadata.tables) > 1:
    join_sampled_data.to_csv(data_dir + f"synthetic_all.csv", index=None)
# 6. 数据评估器
evaluator = Evaluator(join_data, join_sampled_data)
evaluator.eval_fidelity(save_path=data_dir + "fidelity.csv")
evaluator.eval_privacy(save_path=data_dir + "privacy.csv")
evaluator.eval_diversity(save_path=data_dir + "diversity.csv")
evaluator.eval_histogram(save_path=data_dir + "histogram.png")
evaluator.eval_tsne(save_path=data_dir + "tsne.png")
