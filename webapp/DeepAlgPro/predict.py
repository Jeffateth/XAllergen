from model import convATTnet
from Bio import SeqIO
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import argparse
import os


def main():
    argparser = argparse.ArgumentParser(
        description="DeepAlgPro Network for predicting allergens with attention output."
    )
    argparser.add_argument(
        "-i", "--inputs", default="./", type=str, help="input file (FASTA)"
    )
    argparser.add_argument("-b", "--batch-size", default=1, type=int, metavar="N")
    argparser.add_argument(
        "-o", "--output", default="allergenic_predict.txt", type=str, help="output file"
    )
    argparser.add_argument(
        "--save-attention",
        action="store_true",
        help="Save attention weights as .npy files",
    )

    args = argparser.parse_args()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    predict(args)


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("We will use " + torch.cuda.get_device_name())

codeadict = {
    "A": "1",
    "C": "2",
    "D": "3",
    "E": "4",
    "F": "5",
    "G": "6",
    "H": "7",
    "I": "8",
    "K": "9",
    "L": "10",
    "M": "11",
    "N": "12",
    "P": "13",
    "Q": "14",
    "R": "15",
    "S": "16",
    "T": "17",
    "V": "18",
    "W": "19",
    "Y": "20",
}


class MyDataset(Dataset):
    def __init__(self, sequence, labels):
        self._data = sequence
        self._label = labels

    def __getitem__(self, idx):
        sequence = self._data[idx]
        label = self._label[idx]
        return sequence, label

    def __len__(self):
        return len(self._data)


def format(predict_fasta):
    formatfasta = []
    recordid = []
    for record in SeqIO.parse(predict_fasta, "fasta"):
        fastalist = []
        length = len(record.seq)
        if length <= 1000:
            for _ in range(1, 1000 - length + 1):
                fastalist.append(0)
            for a in record.seq:
                fastalist.append(
                    int(codeadict.get(a.upper(), "0"))
                )  # fallback for unknown chars
        formatfasta.append(fastalist)
        recordid.append(record.id)
    inputarray = np.array(formatfasta)
    idarray = np.array(recordid, dtype=object)
    return inputarray, idarray


def predict(args):
    profasta = torch.tensor(format(args.inputs)[0], dtype=torch.long)
    proid = format(args.inputs)[1]
    data_ids = MyDataset(profasta, proid)
    data_loader = DataLoader(
        dataset=data_ids, batch_size=args.batch_size, shuffle=False
    )

    model = convATTnet().to(device)
    model.load_state_dict(torch.load("model.pt", map_location=device), strict=True)
    model.eval()

    pred_r = []

    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            inputs, inputs_id = data
            inputs = inputs.to(device)
            outputs, attn = model(inputs, return_attention=True)

            for j in range(inputs.size(0)):
                prob = outputs[j].item()
                label = "allergenicity" if prob > 0.5 else "non-allergenicity"
                pid = inputs_id[j]
                pred_r.append([pid, prob, label])

                # Optional: Save attention weights
                if args.save_attention:
                    attn_matrix = attn[j].mean(dim=0).cpu().numpy()  # shape [L, L]
                    np.save(f"attention_{pid}.npy", attn_matrix)

    # Save predictions
    df = pd.DataFrame(pred_r, columns=["protein", "scores", "predict result"])
    df.to_csv(args.output, sep="\t", header=True, index=True)


if __name__ == "__main__":
    main()
