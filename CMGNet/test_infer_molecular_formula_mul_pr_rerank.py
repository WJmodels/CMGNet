import os
import time
import json
import torch
import argparse
from tqdm import tqdm
from re import L
import numpy as np
from datasets import load_dataset
from multiprocessing import Queue, Process, Lock
from transformers import BartTokenizer
from collections import Counter
from my_datasets.nmr_datasets import NmrDataset
from my_models.multi_constraint_molecular_generator import MultiConstraintMolecularGenerator
TYPE_MODEL = "new"

try:
    from rdkit.Chem import DataStructs
    from rdkit.Chem import AllChem as Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem import Descriptors
    periodic_table_of_elements = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La",
                                  "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"]
except:
    print("rdkit is not installed.")


def calculate_tanimoto_similarity(smiles_1, smiles_2):
    if "Chem" in globals().keys() and "DataStructs" in globals().keys():
        try:
            fingerprint_1 = Chem.GetMorganFingerprintAsBitVect(
                Chem.MolFromSmiles(smiles_1), 2, nBits=1024)
            fingerprint_2 = Chem.GetMorganFingerprintAsBitVect(
                Chem.MolFromSmiles(smiles_2), 2, nBits=1024)
            tanimoto = DataStructs.FingerprintSimilarity(
                fingerprint_1, fingerprint_2)
            return tanimoto
        except:
            return 0
    else:
        print("calculate_tanimoto_similarity, rdkit is not installed.")
        return 0


def draw_smiles_save(smiles_pattern, save_path):
    if "Chem" in globals().keys() and "rdMolDraw2D" in globals().keys():
        try:
            mol = Chem.MolFromSmiles(smiles_pattern)
            d = rdMolDraw2D.MolDraw2DCairo(400, 200)
            # tmp = rdMolDraw2D.PrepareMolForDrawing(mol)
            d.DrawMolecule(mol)
            d.FinishDrawing()
            d.WriteDrawingText(save_path)
        except:
            pass
    else:
        print("rdkit is not installed.")
        pass


def get_molecular_formula(mol_):
    if "Chem" in globals().keys() and "periodic_table_of_elements" in globals().keys():
        try:
            if mol_ is None:
                return ""
            mol = Chem.AddHs(mol_)
            dict_ = dict(Counter(atom.GetSymbol() for atom in mol.GetAtoms()))
            str_ = ""
            for i in periodic_table_of_elements:
                value = dict_.pop(i) if i in dict_.keys() else None
                if value is not None:
                    str_ = str_ + i + str(value)
            if dict_:
                for k, v in dict_.items():
                    str_ = str_ + k + str(v)
                print(dict_)
                # raise
            return str_
        except:
            return ""
    else:
        print("rdkit is not installed.")
        return ""


def judge_unqualified(smiles, fragment=None, molecular_formula=None, molecular_weight=None, molecular_weight_range=5, use_dict=None):
    if "Chem" in globals().keys() and "Descriptors" in globals().keys():
        # use_dict = {"C13-NMR": True, "molecular_formula": True,
        #             "fragment": False, "molecular_weight": False, "SMILES": True}
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False

        if use_dict["molecular_formula"] is True:
            if molecular_formula == "":
                molecular_formula = None
            if molecular_formula is not None and get_molecular_formula(mol) != molecular_formula:
                return False

        if use_dict["molecular_weight"] is True:
            if molecular_weight == -1:
                molecular_weight = None
            if molecular_weight is not None and abs(Descriptors.ExactMolWt(mol) - molecular_weight) > molecular_weight_range:
                return False

        if use_dict["fragment"] is True:
            if isinstance(fragment, str):
                fragment_ = Chem.MolFromSmiles(fragment)
            else:
                fragment_ = fragment
            if fragment_ is not None and not mol.HasSubstructMatch(fragment_):
                return False
        return True
    else:
        print("rdkit is not installed.")
        return True


def remove_unqualified_list(smiles_list, fragment=None, molecular_formula=None, molecular_weight=None, use_dict=None):
    if "Chem" in globals().keys() and "Descriptors" in globals().keys():
        try:
            fragment_ = Chem.MolFromSmiles(
                fragment.replace("(*)", "").replace("*", "")) if fragment is not None else None
        except:
            fragment_ = None
        tmp = []
        for idx, i in enumerate(smiles_list):
            flag = judge_unqualified(i,
                                     fragment=fragment_,
                                     molecular_formula=molecular_formula,
                                     molecular_weight=molecular_weight,
                                     molecular_weight_range=5,
                                     use_dict=use_dict)
            if flag:
                tmp.append(i)
        print(len(tmp))
        return tmp
    else:
        print("rdkit is not installed.")
        return smiles_list


def get_token(data, tokenizer, use_dict, max_length, nmr_dataset):
    if use_dict["molecular_formula"] is True and "molecular_formula" in data.keys():
        tmp = tokenizer(data["molecular_formula"], max_length=max_length)
        tmp["input_ids"] = [181] + tmp["input_ids"][1:-1] + [182]
    else:
        tmp = {"input_ids": [], "attention_mask": []}

    if use_dict["C13-NMR"] is True and "nmr" in data.keys():
        tmp_2 = nmr_dataset.fill_item(data["nmr"][0])
        if TYPE_MODEL == "new":
            tmp["input_ids"] = tmp_2["input_ids"] + tmp["input_ids"]
            tmp["attention_mask"] = tmp_2["attention_mask"] + \
                tmp["attention_mask"]
        elif TYPE_MODEL == "old":
            tmp["input_ids"] += tmp_2["input_ids"]
            tmp["attention_mask"] += tmp_2["attention_mask"]

    if use_dict["molecular_weight"] is True and "molecular_weight" in data.keys():
        if data["molecular_weight"] != -1:
            tmp_2 = str(round(data["molecular_weight"], 2))
            tmp_2 = tokenizer(tmp_2, max_length=max_length)
            tmp_2["input_ids"] = [185] + tmp_2["input_ids"][1:-1] + [186]
            tmp["input_ids"] += tmp_2["input_ids"]
            tmp["attention_mask"] += tmp_2["attention_mask"]

    idx_frg_result = None
    if use_dict["fragment"] is not False and "fragments" in data.keys() and len(data["fragments"]) > 0:
        idx_frg_result = 0
        if use_dict["fragment"] != "use_3":
            if use_dict["fragment"] == "max":
                frg_max = len(data["fragments"][0])
                for idx_frg, k in enumerate(data["fragments"]):
                    if len(k) > frg_max:
                        idx_frg_result = idx_frg
                        frg_max = len(k)
            elif use_dict["fragment"] == "min":
                frg_min = len(data["fragments"][0])
                for idx_frg, k in enumerate(data["fragments"]):
                    if len(k) < frg_min:
                        idx_frg_result = idx_frg
                        frg_min = len(k)
            tmp_2 = tokenizer(
                data["fragments"][idx_frg_result], max_length=max_length)
            tmp_2["input_ids"] = [183] + tmp_2["input_ids"][1:-1] + [184]
            tmp["input_ids"] += tmp_2["input_ids"]
            tmp["attention_mask"] += tmp_2["attention_mask"]

        else:
            for idx_frg_result in range(3):
                if idx_frg_result == len(data["fragments"]):
                    break
                tmp_2 = tokenizer(
                    data["fragments"][idx_frg_result], max_length=max_length)
                tmp_2["input_ids"] = [183] + tmp_2["input_ids"][1:-1] + [184]
                tmp["input_ids"] += tmp_2["input_ids"]
                tmp["attention_mask"] += tmp_2["attention_mask"]

    if len(tmp["input_ids"]) > max_length:
        tmp["input_ids"] = tmp["input_ids"][:max_length]
        tmp["attention_mask"] = tmp["attention_mask"][:max_length]

    tmp["input_ids"] = [tmp["input_ids"]]
    tmp["attention_mask"] = [tmp["attention_mask"]]
    return tmp, idx_frg_result


def work_q_in(q, q_len, lock, val_folder, tokenizer_path, extension="json", use_dict=None):
    tokenizer = BartTokenizer.from_pretrained(tokenizer_path)
    # print(tokenizer.mask_token_id)
    # a = tokenizer.get_special_tokens_mask([0,1,2,3,4,5,6,7,8,9,10,181,182,182,190,191,192], already_has_special_tokens=True)
    nmr_dataset = NmrDataset()
    max_length = 512
    data_files = {"train": [os.path.join(
        val_folder, i) for i in os.listdir(val_folder)]}
    raw_datasets = load_dataset(extension, data_files=data_files)
    dataset_len = len(raw_datasets["train"])
    q_len.put(dataset_len)
    iter_loader = iter(raw_datasets["train"])
    idx = 0

    while 1:
        if idx == dataset_len:
            break
        if q.qsize() < 50:
            data = next(iter_loader)
            data["index"] = idx
            tmp, fragment_idx = get_token(
                data, tokenizer, use_dict, max_length, nmr_dataset)
            data.update(tmp)
            data["fragment_idx"] = fragment_idx
            # , "molecular_formula", "molecular_weight", "fragments"
            del_list = ["elements"]
            for i in del_list:
                if i in data.keys():
                    del data[i]
            lock.acquire()
            print("---"+str(idx)+"---")
            q.put(data)
            idx += 1
            lock.release()
    print("return")
    return


def work_calculate_cmc(q_in, q_out, lock_in, lock_out, device="cpu", kwargs={}, model_config=None, weights_path=None, val_rule=[1], use_dict=None):
    time.sleep(10)
    model = MultiConstraintMolecularGenerator(**model_config)
    model.load_weights(weights_path)
    model.to(device)
    model.eval()

    return_flag = False
    infer_dict = {"tokenizer": model.tokenizer}
    infer_dict.update(kwargs)

    while 1:
        lock_in.acquire()
        if not q_in.empty():
            item = q_in.get()
        else:
            return_flag = True
        lock_in.release()
        if return_flag:
            break
        smiles = item.pop("smiles") if "smiles" in item.keys() else None
        smiles_h = item.pop("smiles_h") if "smiles_h" in item.keys() else None
        nmr = item.pop("nmr") if "nmr" in item.keys() else None
        index = item.pop("index") if "index" in item.keys() else None

        molecular_formula = item.pop(
            "molecular_formula") if "molecular_formula" in item.keys() else None
        molecular_weight = item.pop(
            "molecular_weight") if "molecular_weight" in item.keys() else None
        fragments = item.pop(
            "fragments") if "fragments" in item.keys() else None
        fragment_idx = item.pop(
            "fragment_idx") if "fragment_idx" in item.keys() else None
        if fragments is not None and fragment_idx is not None and len(fragments) > fragment_idx:
            used_fragment = fragments[fragment_idx]
        else:
            used_fragment = None
        return_dict = {"smiles": smiles, "nmr": nmr,
                       "index": index, "fragment_idx": fragment_idx}
        # print(item.keys())
        item = model.tokenizer.pad(item, return_tensors="pt")

        for k in item.keys():
            item[k] = item[k].to(device)

        infer = infer_dict.copy()
        infer.update(item)
        with torch.no_grad():
            if TYPE_MODEL == "new":
                result = model.infer_2(**infer)["smiles"]
            elif TYPE_MODEL == "old":
                result = model.infer(**infer)["smiles"]

        return_dict["result"] = result
        # max_k = 0
        # for idx, k in enumerate(fragments):
        #         draw_smiles_save(k, "tmp/"+"frg"+str(idx)+".jpg")

        # for idx, k in enumerate([smiles[0]] + result):
        #     draw_smiles_save(k, "tmp/"+str(idx)+".jpg")
        # print(smiles, result)
        # rank_strict = 1001
        for j in range(len(val_rule)):
            if isinstance(val_rule[j], int) or isinstance(val_rule[j], float):
                rank = 1001
                for idx, i in enumerate(result):
                    if calculate_tanimoto_similarity(i, smiles[0]) >= val_rule[j]:
                        rank = idx
                        break
                return_dict["rank_"+str(val_rule[j])] = rank
            elif val_rule[j] == "removed":
                result_2 = remove_unqualified_list(result,
                                                   fragment=used_fragment,
                                                   molecular_formula=molecular_formula,
                                                   molecular_weight=molecular_weight,
                                                   use_dict=use_dict)
                rank = 1001
                for idx, i in enumerate(result_2):
                    if i in smiles:
                        rank = idx
                        break
                return_dict["rank_removed"] = rank
            print(rank)

        lock_out.acquire()
        q_out.put(return_dict)
        lock_out.release()
    model.train()
    print("return"+device)
    return


def cmc_calculate(rank_list, num_beams, dataset_len, save_path):
    rank_list = np.array(rank_list)
    cmc_list = []
    for i in range(num_beams):
        cmc_list.append(sum(rank_list <= i))
    cmc_list = [i/dataset_len for i in cmc_list]
    print(cmc_list)
    cmc_list = [str(i) for i in cmc_list]
    if save_path is not None:
        with open(save_path, "w") as f:
            f.write("\n".join(cmc_list))


def work_q_out(q_out, q_len, lock_out, num_beams, save_path, val_rule):
    time.sleep(10)
    dataset_len = q_len.get()
    rank_list = [[] for _ in range(len(val_rule))]
    all_result_list = []
    idx = 0
    while 1:
        if not q_out.empty():
            with lock_out:
                item = q_out.get()
                all_result_list.append(item)
                for i in range(len(val_rule)):
                    rank_list[i].append(item["rank_"+str(val_rule[i])])
                idx += 1
                # print("q_out idx", idx)
        if idx == dataset_len:
            break
    print("abc")
    for i in range(len(val_rule)):
        save_path_tmp = save_path.replace("val_rule", str(val_rule[i]))
        cmc_calculate(rank_list[i], num_beams, dataset_len, save_path_tmp)
        with open(save_path_tmp.replace(".cmc", ".json"), "w") as f:
            json.dump(all_result_list, f, indent=4)
            # json.dump(all_result_list, f)
    return


def main(args):
    val_folder = args.val_folder
    weights_path = args.weights_path
    use_dict = {"C13-NMR": True, "molecular_formula": True,
                "fragment": False, "molecular_weight": False, "SMILES": True}

    get_value(args.molecular_formula, use_dict, "molecular_formula")
    get_value(args.fragment, use_dict, "fragment")
    get_value(args.C13_NMR, use_dict, "C13-NMR")

    model_config = {
        "model_path": None,
        "config_json_path": "configs/bart.json",
        "tokenizer_path": "./tokenizer-smiles-bart",
    }
    val_rule = [1, ]

    save_path = weights_path.replace(
        ".pth", "_val_rule_fml_ex.cmc")
    save_path = weights_path.replace(
        ".pth", "_val_rule_" + args.save_path_end + ".cmc")
    kwargs = {"length_penalty": 0.4,
              "num_beams": 100}

    q_in, q_out, q_len = Queue(), Queue(), Queue()
    lock_in, lock_out = Lock(),  Lock()

    process_list = []
    p = Process(target=work_q_in, args=(
        q_in, q_len, lock_in, val_folder, model_config["tokenizer_path"],  "json", use_dict))
    p.start()
    process_list.append(p)
    p = Process(target=work_q_out, args=(
        q_out, q_len, lock_out, kwargs["num_beams"], save_path, val_rule))
    p.start()
    process_list.append(p)

    for i in range(4):
        # p = Process(target=work_calculate_cmc, args=(
        #     q_in, q_out, lock_in, lock_out, "cpu", kwargs, model_config, weights_path, val_rule, use_dict))
        p = Process(target=work_calculate_cmc, args=(
            q_in, q_out, lock_in, lock_out, "cuda:"+str(i), kwargs, model_config, weights_path, val_rule, use_dict))
        p.start()
        # p.join()
        process_list.append(p)

    for i in process_list:
        p.join()


def get_value(value, dict_, name):
    if value.lower() == "true":
        dict_[name] = True
    elif value.lower() == "false":
        dict_[name] = False
    else:
        dict_[name] = value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--val_folder",
        type=str,
        default=None,
        help="val_folder")
    parser.add_argument(
        "--weights_path",
        type=str,
        default=None,
        help="weights_path")
    parser.add_argument(
        "--save_path_end",
        type=str,
        default=None,
        help="save_path_end")
    parser.add_argument(
        "--molecular_formula",
        type=str,
        default=None,
        help="molecular_formula")
    parser.add_argument(
        "--fragment",
        type=str,
        default=None,
        help="fragment")
    parser.add_argument(
        "--C13_NMR",
        type=str,
        default="true",
        help="C13_NMR")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    args = parse_args()
    main(args)

    # python test_infer_molecular_formula_mul_pr.py > test_infer_molecular_formula_mul_pr.log 2>&1 &
