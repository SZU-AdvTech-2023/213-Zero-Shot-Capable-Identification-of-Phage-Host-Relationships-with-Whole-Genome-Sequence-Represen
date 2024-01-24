from functools import partial

import torch
from torch.utils.data import DataLoader

from data_loading import fasta_dataset, my_collate_fn2, my_collate_fn, strain_fasta_dataset, my_collate_strain
from eval import testStrain
from import_data import getStrainData
from model import cnn_module, cnn_module_3conv

if __name__ == "__main__":
    # strain
    model_file = "model_save/strain/strain_cnn3_model_margin-1-epoch-200.pth"
    device="cuda:1"
    kmer=6
    num_workers=32
    batch_size=32

    # 加载数据
    strainPhage, strainPhageDNA, strainHost, strainHostDNA, strainLabel = getStrainData(kmer)
    # 得到测试集
    strain_test_dataset = strain_fasta_dataset(strainPhage[11232:12480], strainPhageDNA[11232:12480],
                                                strainHost[11232:12480],
                                                strainHostDNA[11232:12480], strainLabel[11232:12480])
    print("	|-* Provided testing sets totally has [", len(strainPhage[11232:12480]), "] labels.")
    # preparing the test data for the evaluation.
    ## 1. loading model
    print("@ Loading model ... ", end="")
    ## parparing host data information.
    model = cnn_module_3conv()

    model.load_state_dict(torch.load(model_file))
    model = model.to(device)

    print("[ok]")

    ## 2. loading data
    valid_generator = DataLoader(strain_test_dataset, batch_size, collate_fn=partial(my_collate_strain),
                                 num_workers=num_workers)
    cached_test_ph, cached_test_bt, cached_test_label = [], [], []
    for phs, bts, labels in valid_generator:
        # X = torch.tensor(X, dtype = torch.float32).transpose(1,2)
        imgs_ph = torch.tensor(phs, dtype=torch.float32)
        imgs_bt = torch.tensor(bts, dtype=torch.float32)

        cached_test_ph.append(torch.unsqueeze(imgs_ph, dim=1))  # 增加维度1，通道数
        cached_test_bt.append(torch.unsqueeze(imgs_bt, dim=1))
        cached_test_label.append(torch.tensor(labels))
    print(" |- loading [ok].")

    acc_test, _, _ = testStrain(model, cached_test_ph, cached_test_bt, cached_test_label, device, 1, True)
    print(f"cnn3[Test acc]:{acc_test}")