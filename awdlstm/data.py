from paddlenlp.datasets import load_dataset,MapDataset
from paddlenlp.transformers import SkepTokenizer

tokenizer_model_name = ""

def load_dataset(datafiles):
    def read(data_path):
        with open(data_path,"r",encoding="utf-8") as f:
            for line in f:
                s = line.split('","')
                label,title,text = s[0].strip('"'),s[1],s[2].strip('"\n')
                yield {"label":label,"title":title,"text":text}

    if isinstance(datafiles, str):
        return MapDataset(list(read(datafiles)))
    elif isinstance(datafiles, list) or isinstance(datafiles, tuple):
        return [MapDataset(list(read(datafile))) for datafile in datafiles]


tokenizer = SkepTokenizer.from_pretrained()