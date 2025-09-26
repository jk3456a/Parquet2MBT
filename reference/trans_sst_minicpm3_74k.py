# -*- coding: utf-8 -*-
"""
When: 2025-09-26
Where: Repo/trans_sst_minicpm3_74k.py
Repo: https://codeup.aliyun.com/modelbest/zyx_infra/tree/main
Description:
    Transform line-delimited JSON into SSTable using the MiniCPM3_74k tokenizer and thrift schema.
"""

import os
import sys

cur_path = os.path.split(os.path.realpath(__file__))[0]
# sys.path.append(os.path.abspath(os.path.join(cur_path, "../")))
sys.path.append(os.path.abspath(os.path.join(cur_path)))

import argparse
import json
import thriftpy2 as thriftpy

from base_handler import DocHandler
from tokenizer.tokenizer_py import add_tokenizer_args, build_tokenizer

doc_thrift = thriftpy.load("./proto/traindoc.thrift")


class BaseDocJsonHandler(DocHandler):
    def __init__(self, out_fname, tokenizer):
        super().__init__(out_fname)
        self.tokenizer = tokenizer
        self.tokenizer_version = tokenizer.tokenizer_version
        self.chatml_prefix_mask_len = len(self.tokenizer.tokenize("<|im_start|>assistant"))
        self.chatml_suffix_mask_len = len(self.tokenizer.tokenize("<|im_end|>\n"))
        self.output_thrift = getattr(doc_thrift, 'BaseDoc')
        self.total_token = 0

    def add_json_line(self, line, tags=None, fp=None):
        #import pdb;pdb.set_trace()
        data = json.loads(line)
        tokens, loss_mask = [], []
        # input/output pt format
        if "input" in data and "output" in data:
            # check data["input"] and data["output"]
            if (type(data["input"]) != str or type(data["output"]) != str or len(data["output"]) == 0):
                print("Illegal input/output data detected, skipped!!!")
                return None

            input_ids = self.tokenizer.encode(data["input"], add_special_tokens=True)
            output_ids = self.tokenizer.encode(data["output"], add_special_tokens=False)
            tokens.extend(input_ids)
            tokens.extend(output_ids)
            loss_mask.extend([1] * (len(input_ids) - 1) + [0])
            loss_mask.extend([0] * len(output_ids))
            tokens.append(self.tokenizer.eos_token_id)
            loss_mask.append(1)
            self.total_token+=len(tokens)

        # messages chatml format
        elif "messages" in data:
            tokens = [self.tokenizer.bos_token_id]
            loss_mask = [1]

            for session in data["messages"]:
                # convert role and content dtype
                role = session.get("role", "")
                # sometime do not want to calculate loss for some sessions
                calculate_loss = session.get("loss", True)

                if not isinstance(role, str):
                    role = str(role)
                content = session.get("content", "")
                if not isinstance(content, str):
                    content = str(content)
                # concat role and content
                content_ids = self.tokenizer.encode("<|im_start|>" + role + "\n" + content + "<|im_end|>\n", add_special_tokens=False)
                tokens.extend(content_ids)
                if role == "assistant":
                    # content of assistant will be used to calculate loss.
                    # loss_mask.extend([1] * self.chatml_prefix_mask_len + [0] * (len(content_ids) - self.chatml_prefix_mask_len - self.chatml_suffix_mask_len) + [1] * self.chatml_suffix_mask_len)
                    if calculate_loss:
                        loss_mask.extend([1] * self.chatml_prefix_mask_len + [0] * (len(content_ids) - self.chatml_prefix_mask_len - self.chatml_suffix_mask_len) + [1] * self.chatml_suffix_mask_len)
                    else:
                        # content of other roles will skip loss calculation.
                        loss_mask.extend([1] * len(content_ids))
                else:
                    # content of other roles will skip loss calculation.
                    loss_mask.extend([1] * len(content_ids))
            tokens.append(self.tokenizer.eos_token_id)
            self.total_token+=len(tokens)
            loss_mask.append(1)
        else:
            print("Illegal data detected, skipped!!!")
            return None
        
        assert len(tokens) == len(loss_mask)
        doc = self.output_thrift(
            token_ids=tokens,
            mask=loss_mask,
            tag=tags,
            docid=fp,
        )
        return super(BaseDocJsonHandler, self).add(doc, fp)
    
    def save(self):
        super(BaseDocJsonHandler, self).save(
                module_name=doc_thrift.__name__,
                class_name=self.output_thrift.__name__,
                tokenizer_version=self.tokenizer_version)
        print(f"self.total_token {self.total_token}")


def run():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='converter')
    group.add_argument('--json-file', type=str, required=True,
                       help='input raw json file.')
    group.add_argument('--output-sstable', type=str, required=True,
                       help='output sstable file path.')
    add_tokenizer_args(parser)
    args = parser.parse_args()

    # hard code here, force set minicpm3_74k
    args.tokenizer_version = "minicpm3_74k"
    tokenizer = build_tokenizer(args)
    doc_handler = BaseDocJsonHandler(args.output_sstable, tokenizer)

    with open(args.json_file, 'r', encoding='utf-8') as fin:
        for i, line in enumerate(fin):
            doc_handler.add_json_line(line)
            if i % 1000 == 0:
                print(i)

    doc_handler.save()


# Usage example
# python demo/json2sst_example.py --json_file=../test.json --output_sstable=demo/test_json.sstable 

if __name__ == '__main__':
    run()