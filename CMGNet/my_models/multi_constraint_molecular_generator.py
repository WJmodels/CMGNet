
import os
import json
import torch
from torch import nn
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig


class MultiConstraintMolecularGenerator(nn.Module):
    def __init__(self, **kwargs):
        super(MultiConstraintMolecularGenerator, self).__init__()

        self.model_path = kwargs.pop(
            "model_path") if "model_path" in kwargs.keys() else None
        self.config_json_path = kwargs.pop(
            "config_json_path") if "config_json_path" in kwargs.keys() else None
        self.tokenizer_path = kwargs.pop(
            "tokenizer_path") if "tokenizer_path" in kwargs.keys() else None

        if self.model_path is not None and os.path.exists(self.model_path):
            assert self.config_json_path is None
            self.config = BartConfig.from_pretrained(self.model_path)
            self.model = BartForConditionalGeneration.from_pretrained(
                self.model_path)

        elif self.config_json_path is not None and os.path.exists(self.config_json_path):
            with open(self.config_json_path, "r") as f:
                json_dict = json.loads(f.read())
            self.config = BartConfig(**json_dict)
            self.model = BartForConditionalGeneration(config=self.config)

        else:
            raise "ERROR: No Model Found.\n"

        if self.tokenizer_path is not None:
            self.tokenizer = BartTokenizer.from_pretrained(self.tokenizer_path)

    def forward(self, **kwargs):
        nmr_feature = kwargs.pop(
            "nmr_feature") if "nmr_feature" in kwargs.keys() else None
        if nmr_feature is not None:
            input_ids = kwargs.pop(
                "input_ids") if "input_ids" in kwargs.keys() else None
            model_tmp = self.model.model.encoder
            kwargs["inputs_embeds"] = model_tmp.embed_tokens(
                input_ids) * model_tmp.embed_scale
            # print(kwargs["inputs_embeds"].shape, input_ids.shape, nmr_feature.shape)
            for i in range(input_ids.shape[0]):
                kwargs["inputs_embeds"][i][input_ids[i]
                                           == 191, :] = nmr_feature[i]
            kwargs["inputs_embeds"][:, 0, :] = nmr_feature
        return self.model(**kwargs)

    def infer(self, **kwargs):
        tokenizer = kwargs.pop(
            "tokenizer") if "tokenizer" in kwargs.keys() else None
        num_beams = kwargs.pop(
            "num_beams") if "num_beams" in kwargs.keys() else 1
        num_return_sequences = kwargs.pop(
            "num_return_sequences") if "num_return_sequences" in kwargs.keys() else num_beams
        max_length = kwargs.pop(
            "max_length") if "max_length" in kwargs.keys() else 512

        with torch.no_grad():
            result = self.model.generate(max_length=max_length,
                                         num_beams=num_beams,
                                         num_return_sequences=num_return_sequences,
                                         bos_token_id=187,
                                         pad_token_id=1,
                                         eos_token_id=188,
                                         forced_bos_token_id=187,
                                         forced_eos_token_id=188,
                                         **kwargs)
        dict_ = {"input_ids_tensor": result}
        if tokenizer is not None:
            smiles = [tokenizer.decode(i) for i in result]
            smiles = [i.replace("<SMILES>", "").replace(
                "</SMILES>", "").replace("<pad>", "").replace("</s>", "") for i in smiles]
            dict_["smiles"] = smiles
        return dict_

    def infer_2(self, **kwargs):
        tokenizer = kwargs.pop(
            "tokenizer") if "tokenizer" in kwargs.keys() else None
        num_beams = kwargs.pop(
            "num_beams") if "num_beams" in kwargs.keys() else 1
        num_return_sequences = kwargs.pop(
            "num_return_sequences") if "num_return_sequences" in kwargs.keys() else num_beams
        max_length = kwargs.pop(
            "max_length") if "max_length" in kwargs.keys() else 512

        with torch.no_grad():
            # result = self.model.generate(max_length=max_length,
            #                              num_beams=num_beams,
            #                              num_return_sequences=num_return_sequences,
            #                              bos_token_id=187,
            #                              pad_token_id=1,
            #                              eos_token_id=188,
            #                              decoder_start_token_id=187,
            #                              forced_bos_token_id=187,
            #                              forced_eos_token_id=188,
            #                              **kwargs)
            result = self.model.generate(max_length=max_length,
                                         num_beams=num_beams,
                                         num_return_sequences=num_return_sequences,
                                         bos_token_id=187,
                                         pad_token_id=1,
                                         eos_token_id=188,
                                         decoder_start_token_id=187,
                                         **kwargs)
        dict_ = {"input_ids_tensor": result}
        if tokenizer is not None:
            smiles = [tokenizer.decode(i) for i in result]
            # print(smiles)
            # raise
            smiles = [i.replace("<SMILES>", "").replace(
                "</SMILES>", "").replace("<pad>", "").replace("</s>", "") for i in smiles]
            dict_["smiles"] = smiles
        return dict_

    def infer_smiles2nmr(self, **kwargs):
        num_beams = kwargs.pop(
            "num_beams") if "num_beams" in kwargs.keys() else 1
        num_return_sequences = kwargs.pop(
            "num_return_sequences") if "num_return_sequences" in kwargs.keys() else num_beams
        max_length = kwargs.pop(
            "max_length") if "max_length" in kwargs.keys() else 512
        with torch.no_grad():
            result = self.model.generate(max_length=max_length,
                                         num_beams=num_beams,
                                         num_return_sequences=num_return_sequences,
                                         bos_token_id=189,
                                         pad_token_id=1,
                                         eos_token_id=190,
                                         decoder_start_token_id=189,
                                         **kwargs)
            result[result < 191] = 0
            # result[result>4190] = 0

        dict_ = {"input_ids_tensor": result}
        return dict_

    def load_weights(self, path):
        if path is not None:
            model_dict = torch.load(path, map_location=torch.device("cpu"))
            self.load_state_dict(model_dict)
