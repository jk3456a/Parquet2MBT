import argparse
import multiprocessing
import os
import random
import re
import sys
import time
import uuid
from multiprocessing.pool import Pool

import pyarrow.parquet as pq

from mb_data_platform.common.obs_client import OBSCli
from mb_data_platform.common.redis_client import hw_bj_redis_cli
from mb_data_platform.dataset2sstable.content_process import LongTextFilter
from mb_data_platform.dataset2sstable.special_dataset_process import FilterContentWithoutBreak, UnUsualChar
from mb_data_platform.dataset2sstable.ref_replace import RefReplace
from mb_data_platform.dedup.dedup_bitmap.utils import logger

sys.path.append("/app/infra")
from tokenizer.tokenizer_py import build_tokenizer_helper

import json
from pathlib import Path

import thriftpy2 as thriftpy
from model_util.base_handler import BaseHandler
from transformers import AutoTokenizer

doc_thrift = thriftpy.load("/app/infra/proto/traindoc.thrift")


class BaseDocJsonHandler(BaseHandler):
    def __init__(self, out_file_name, tokenizer):
        super().__init__(out_file_name)
        self.tokenizer = tokenizer
        self.tokenizer_version = tokenizer.tokenizer_version
        self.chatml_prefix_mask_len = len(
            self.tokenizer.tokenize("<|im_start|>assistant")
        )
        self.chatml_suffix_mask_len = len(self.tokenizer.tokenize("<|im_end|>\n"))
        self.output_thrift = getattr(doc_thrift, "BaseDoc")

        self.ref_replace = RefReplace()
        self.regex_break_line = r"([ \t]*\n){3,}"
        self.total_token = 0

    def add_json_line(self, data):
        if isinstance(data, str):
            data = json.loads(data)
        self.add_data(data)

    def add_data(self, data):
        tokens, loss_mask = [], []
        # input/output pt format
        if "input" in data and "output" in data:
            # check data["input"] and data["output"]
            if (type(data["input"]) != str or type(data["output"]) != str or len(data["output"]) == 0):
                print("Illegal input/output data detected, skipped!!!")
                return None

            # data["input"] = self.ref_replace.process_single(data["input"])
            data["output"] = self.ref_replace.process_single(data["output"])
            # data["input"] = re.sub(self.regex_break_line, "\n\n", data["input"])
            data["output"] = re.sub(self.regex_break_line, "\n\n", data["output"])

            input_ids = self.tokenizer.encode(data["input"], add_special_tokens=True)
            output_ids = self.tokenizer.encode(data["output"], add_special_tokens=False)
            tokens.extend(input_ids)
            tokens.extend(output_ids)
            loss_mask.extend([1] * (len(input_ids) - 1) + [0])
            loss_mask.extend([0] * len(output_ids))
            tokens.append(self.tokenizer.eos_token_id)
            loss_mask.append(1)
            self.total_token += len(tokens)

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
                content = self.ref_replace.process_single(content)
                content = re.sub(self.regex_break_line, "\n\n", content)
                content_ids = self.tokenizer.encode("<|im_start|>" + role + "\n" + content + "<|im_end|>\n",
                                                    add_special_tokens=False)
                tokens.extend(content_ids)
                if role == "assistant":
                    # content of assistant will be used to calculate loss.
                    # loss_mask.extend([1] * self.chatml_prefix_mask_len + [0] * (len(content_ids) - self.chatml_prefix_mask_len - self.chatml_suffix_mask_len) + [1] * self.chatml_suffix_mask_len)
                    if calculate_loss:
                        loss_mask.extend([1] * self.chatml_prefix_mask_len + [0] * (
                                    len(content_ids) - self.chatml_prefix_mask_len - self.chatml_suffix_mask_len) + [
                                             1] * self.chatml_suffix_mask_len)
                    else:
                        # content of other roles will skip loss calculation.
                        loss_mask.extend([1] * len(content_ids))
                else:
                    # content of other roles will skip loss calculation.
                    loss_mask.extend([1] * len(content_ids))
            tokens.append(self.tokenizer.eos_token_id)
            self.total_token += len(tokens)
            loss_mask.append(1)
        else:
            print("Illegal data detected, skipped!!!")
            return None

        assert len(tokens) == len(loss_mask)
        # 这个实现有点丑陋 暂时也只能这样了
        tags = None
        fp = None
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
            tokenizer_version=self.tokenizer_version,
        )


def __dict2sstable(args):
    content_tag, data_list, output_dir, output_path_suffix, write_batch_num = args
    tokenizer_version = "minicpm3_74k"
    tokenizer = build_tokenizer_helper(tokenizer_version=tokenizer_version)
    logger.info(
        f"dict2sstable, data_list size: {len(data_list)}, {output_dir}, content_tag: {content_tag}"
    )
    sstable_file = f"/tmp/{uuid.uuid4()}.sstable"
    doc_handler = BaseDocJsonHandler(sstable_file, tokenizer)
    t0 = time.time()
    for data in data_list:
        doc_handler.add_data(data)
    doc_handler.save()
    t1 = time.time()

    output_prefix = output_path_suffix.replace(
        ".parquet", f"_{write_batch_num}.sstable"
    )
    if content_tag == "qa":
        obs_path = os.path.join(output_dir, output_prefix)
    else:
        obs_path = os.path.join(output_dir, content_tag, output_prefix)
        path_list = obs_path.split("/")
        obs_path = "/".join(path_list[:-1]) + "/" + content_tag + "_" + path_list[-1]

    logger.info(
        f"保存sstable完成 sstable size:{len(data_list)} time spend:{t1 - t0}, 上传到{obs_path}"
    )

    output_bucket_name = output_dir.lstrip("obs:/").lstrip("/").split("/")[0]
    upload_bucket = OBSCli(output_bucket_name)
    upload_bucket.upload_file(obs_path, sstable_file)
    os.remove(sstable_file)


def process_data_batch(batch_list):
    # 在外面进行多进程处理，这样能提供效率
    for batch in batch_list:
        __dict2sstable(batch)


class Parquet2SstableProcessor:
    def __init__(self, input_dir, output_dir, threshold):
        self.name = "parquet2sstable_processor"
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.threshold = threshold
        self.sstable_file_size = 20 * 10000
        self.max_process_cnt = -1
        self.all_input_parts = None
        self.max_process_time_per_part = 7 * 24 * 60 * 60
        self.long_text_filter = LongTextFilter()
        self.spec_filter = FilterContentWithoutBreak()
        self.unusual_char_filter = UnUsualChar()

    def get_all_input_parts(self):
        bucket_name = self.input_dir.lstrip("obs:/").lstrip("/").split("/")[0]
        bucket_cli = OBSCli(bucket_name)
        files = bucket_cli.list_folder(self.input_dir)

        files = [i for i in files if not i.endswith(".SUCCESS")]
        files.sort()
        logger.info(f"共 {len(files)} 个文件")
        return files

    def unlock_input_part(self, input_part):
        key = f"DP_DATASET_TASK_LOCK_{self.name}_{input_part}"
        hw_bj_redis_cli.delete(key)

    def get_input_part_redis_status(self, input_part):
        key = f"DP_DATASET_TASK_LOCK_{self.name}_{input_part}"
        return hw_bj_redis_cli._rc.get(key)

    def lock_input_part(self, input_part):
        key = f"DP_DATASET_TASK_LOCK_{self.name}_{input_part}"
        set_res = hw_bj_redis_cli.setnx(key, "RUNNING")
        if int(set_res) == 1:
            hw_bj_redis_cli.expire(key, self.max_process_time_per_part)
            return 1
        else:
            return 0

    def done_input_part(self, input_part):
        key = f"DP_DATASET_TASK_LOCK_{self.name}_{input_part}"
        hw_bj_redis_cli._rc.set(key, "DONE")

    def get_processable_part(self):
        if self.all_input_parts is None:
            self.all_input_parts = self.get_all_input_parts()
        while self.all_input_parts:
            input_path = self.all_input_parts.pop(0)
            if self.lock_input_part(input_path):
                logger.debug(f"获得锁, 开始处理{input_path}")
                return input_path
        return None

    def run(self):
        process_cnt = 0
        while process_cnt < self.max_process_cnt or self.max_process_cnt < 0:
            input_part = self.get_processable_part()
            if input_part:
                output_path_suffix = self.get_output_path_suffix(input_part)
                self.process_input_part(input_part, output_path_suffix)
                self.done_input_part(input_part)
                # self.unlock_input_part(input_part)
                process_cnt += 1
            else:
                logger.info(
                    f"任务结束, 共处理{process_cnt}个文件, input_dir: {self.input_dir}"
                )
                break

    def get_output_path_suffix(self, input_part):
        file_name = input_part.removeprefix(self.input_dir).lstrip("/")
        return file_name

    def process_input_part(self, input_part, output_path_suffix):
        logger.info(f"开始处理: {input_part} output:{output_path_suffix}")
        cur_sstable_size = 0
        sstable_sum_size = 0
        batch_list = []
        # 长度范围'<4k', '4k-12k', '12k-20k', '20k-28k', '28k-34k', '34k-68k', '68k-130k', '>130k'
        batch_data_mapping = {
            "lt_4k": [],
            "4k_12k": [],
            "12k_20k": [],
            "20k_28k": [],
            "28k_34k": [],
            "34k_68k": [],
            "68k_130k": [],
            "gt_130k": [],
            "qa": [],
        }

        write_batch_num = 0
        parquet_file_reader = pq.ParquetFile(input_part.replace("obs:/", "/mnt"))
        for batch_table in parquet_file_reader.iter_batches(batch_size=1000):
            for i, sample in enumerate(batch_table.to_pylist()):
                # 过滤掉空白数据,存在qa数据里有空白content的情况
                if (not sample.get("content")) and (not sample.get("messages")):
                    continue
                # 过滤soft delete
                try:
                    if sample["meta"] is None:
                        sample["meta"] = "{}"
                    else:
                        meta = json.loads(sample["meta"])
                        if "soft_delete" in meta and meta["soft_delete"]:
                            if meta["soft_delete"][0] is True:
                                # logger.info(f"soft delete filter")
                                continue
                except json.decoder.JSONDecodeError:
                    logger.error(f"json decode error: {sample}")
                    continue
                except KeyError:
                    logger.error(f"key error: {sample.keys()}")
                    continue

                # 过滤掉有问题的长文本
                if sample.get("content"):
                    if self.long_text_filter.process_single(sample["content"]):
                        continue
                    # if self.spec_filter.process_single(sample["content"]):
                    #     continue
                    if self.unusual_char_filter.process_single(sample["content"]):
                        continue
                # 抽样
                if (r := random.random()) > self.threshold:
                    # logger.info(f"random filter, {r}")
                    continue

                if sample.get("messages"):
                    batch_data_mapping["qa"].append({"messages": sample["messages"]})
                elif sample.get("content"):
                    content = sample["content"]
                    content_data = {"input": "", "output": content}
                    content_len = len(content)
                    if content_len == 0:
                        continue
                    elif content_len < 4000:
                        batch_data_mapping["lt_4k"].append(content_data)
                    elif content_len < 12000:
                        batch_data_mapping["4k_12k"].append(content_data)
                    elif content_len < 20000:
                        batch_data_mapping["12k_20k"].append(content_data)
                    elif content_len < 28000:
                        batch_data_mapping["20k_28k"].append(content_data)
                    elif content_len < 34000:
                        batch_data_mapping["28k_34k"].append(content_data)
                    elif content_len < 68000:
                        batch_data_mapping["34k_68k"].append(content_data)
                    elif content_len < 130000:
                        batch_data_mapping["68k_130k"].append(content_data)
                    else:
                        batch_data_mapping["gt_130k"].append(content_data)
                else:
                    continue

                cur_sstable_size += 1
                sstable_sum_size += 1
                if cur_sstable_size == self.sstable_file_size:
                    write_batch_num += 1
                    tmp_reset = []
                    for batch_k, batch_v in batch_data_mapping.items():
                        if batch_v:
                            batch_list.append(
                                (
                                    batch_k,
                                    batch_v,
                                    self.output_dir,
                                    output_path_suffix,
                                    write_batch_num,
                                )
                            )
                            tmp_reset.append(batch_k)

                    process_data_batch(batch_list)
                    cur_sstable_size = 0
                    batch_list = []
                    for batch_k in tmp_reset:
                        batch_data_mapping[batch_k] = []
        write_batch_num += 1
        for batch_k, batch_v in batch_data_mapping.items():
            if batch_v:
                batch_list.append(
                    (
                        batch_k,
                        batch_v,
                        self.output_dir,
                        output_path_suffix,
                        write_batch_num,
                    )
                )
        process_data_batch(batch_list)


def process_task(input_dir, output_dir, threshold):
    logger.info(f"开始处理: {input_dir}")
    processor = Parquet2SstableProcessor(input_dir, output_dir, threshold)
    processor.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--how", type=str, default=None, choices=["run", "stat", "reset"]
    )
    # 设置多进程数量参数
    parser.add_argument("--num_processes", type=int, default=1)
    args = parser.parse_args()

    # [(输入路径, 输出路径, 抽样率), (输入路径, 输出路径, 抽样率)]
    # tasks = [('obs://mb-datasets/text/10w_why/zh__10w_why__2025082500/',
    #          'obs://mb-datasets/old_tokenizer_sstable/10w_why/zh__10w_why__2025082500/', 1),
    #          ('obs://mb-datasets/text/en.wikihow/en__en.wikihow__2025082500/',
    #           'obs://mb-datasets/old_tokenizer_sstable/en.wikihow/en__en.wikihow__2025082500/', 0.05)
    #          ]
    tasks = [('obs://mb-datasets/text/10w_why/zh__10w_why__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/10w_why/zh__10w_why__2025082500/', 1.0),
     ('obs://mb-datasets/text/1905_film/zh__1905_film__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/1905_film/zh__1905_film__2025082500/', 0.1),
     ('obs://mb-datasets/text/2_qa_sst/zh__2_qa_sst__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/2_qa_sst/zh__2_qa_sst__2025082500/', 0.2),
     ('obs://mb-datasets/text/360_doc_personal_library/zh__360_doc_personal_library__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/360_doc_personal_library/zh__360_doc_personal_library__2025082500/',
      0.1), ('obs://mb-datasets/text/3_book_sst/zh__3_book_sst__2025082500/',
             'obs://mb-datasets/old_version_sstable_250915/3_book_sst/zh__3_book_sst__2025082500/', 0.2),
     ('obs://mb-datasets/text/4_faxinma_sst/zh__4_faxinma_sst__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/4_faxinma_sst/zh__4_faxinma_sst__2025082500/', 0.2),
     ('obs://mb-datasets/text/5_falvyange_sst/zh__5_falvyange_sst__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/5_falvyange_sst/zh__5_falvyange_sst__2025082500/', 0.2),
     ('obs://mb-datasets/text/6_tiantongma_sst/zh__6_tiantongma_sst__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/6_tiantongma_sst/zh__6_tiantongma_sst__2025082500/', 0.2),
     ('obs://mb-datasets/text/8_xingzhengchufa_sst/zh__8_xingzhengchufa_sst__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/8_xingzhengchufa_sst/zh__8_xingzhengchufa_sst__2025082500/', 0.2),
     ('obs://mb-datasets/text/CoT_collection_en/en__CoT_collection_en__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/CoT_collection_en/en__CoT_collection_en__2025082000/', 1.0),
     ('obs://mb-datasets/text/Nemotron_quality_high_kind_actual_kind2_actual/en__Nemotron_quality_high_kind_actual_kind2_actual__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/Nemotron_quality_high_kind_actual_kind2_actual/en__Nemotron_quality_high_kind_actual_kind2_actual__2025082500/',
      0.2),
     ('obs://mb-datasets/text/Nemotron_quality_high_kind_synthetic_kind2_distill/en__Nemotron_quality_high_kind_synthetic_kind2_distill__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/Nemotron_quality_high_kind_synthetic_kind2_distill/en__Nemotron_quality_high_kind_synthetic_kind2_distill__2025082500/',
      0.2),
     ('obs://mb-datasets/text/Nemotron_quality_high_kind_synthetic_kind2_diverse_qa_pairs/en__Nemotron_quality_high_kind_synthetic_kind2_diverse_qa_pairs__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/Nemotron_quality_high_kind_synthetic_kind2_diverse_qa_pairs/en__Nemotron_quality_high_kind_synthetic_kind2_diverse_qa_pairs__2025082500/',
      0.2),
     ('obs://mb-datasets/text/Nemotron_quality_high_kind_synthetic_kind2_extract_knowledge/en__Nemotron_quality_high_kind_synthetic_kind2_extract_knowledge__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/Nemotron_quality_high_kind_synthetic_kind2_extract_knowledge/en__Nemotron_quality_high_kind_synthetic_kind2_extract_knowledge__2025082500/',
      0.2),
     ('obs://mb-datasets/text/Nemotron_quality_high_kind_synthetic_kind2_knowledge_list/en__Nemotron_quality_high_kind_synthetic_kind2_knowledge_list__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/Nemotron_quality_high_kind_synthetic_kind2_knowledge_list/en__Nemotron_quality_high_kind_synthetic_kind2_knowledge_list__2025082500/',
      0.2),
     ('obs://mb-datasets/text/Nemotron_quality_high_kind_synthetic_kind2_wrap_medium/en__Nemotron_quality_high_kind_synthetic_kind2_wrap_medium__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/Nemotron_quality_high_kind_synthetic_kind2_wrap_medium/en__Nemotron_quality_high_kind_synthetic_kind2_wrap_medium__2025082500/',
      0.2), ('obs://mb-datasets/text/RefineCode-code-corpus/en__RefineCode-code-corpus__2025082500/',
             'obs://mb-datasets/old_version_sstable_250915/RefineCode-code-corpus/en__RefineCode-code-corpus__2025082500/',
             1.0), ('obs://mb-datasets/text/aiqu_27txt/zh__aiqu_27txt__2025082500/',
                    'obs://mb-datasets/old_version_sstable_250915/aiqu_27txt/zh__aiqu_27txt__2025082500/', 0.1),
     ('obs://mb-datasets/text/align_no_sys/zh__align_no_sys__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/align_no_sys/zh__align_no_sys__2025082500/', 1.0),
     ('obs://mb-datasets/text/alk_sst/zh__alk_sst__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/alk_sst/zh__alk_sst__2025082500/', 0.2),
     ('obs://mb-datasets/text/ancient_chinese_poetry_clean/zh__ancient_chinese_poetry_clean__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/ancient_chinese_poetry_clean/zh__ancient_chinese_poetry_clean__2025082500/',
      0.2), ('obs://mb-datasets/text/annas_en_pdf_2500w/en__annas_en_pdf_2500w__2025090600/',
             'obs://mb-datasets/old_version_sstable_250915/annas_en_pdf_2500w/en__annas_en_pdf_2500w__2025090600/',
             1.0), ('obs://mb-datasets/text/annas_qikan_1000w/en__annas_qikan_1000w__2025090600/',
                    'obs://mb-datasets/old_version_sstable_250915/annas_qikan_1000w/en__annas_qikan_1000w__2025090600/',
                    1.0), ('obs://mb-datasets/text/arxiv_ocr/en__arxiv_ocr__2025090600/',
                           'obs://mb-datasets/old_version_sstable_250915/arxiv_ocr/en__arxiv_ocr__2025090600/', 1.0),
     ('obs://mb-datasets/text/baike_chinese_new_all/zh__baike_chinese_new_all__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/baike_chinese_new_all/zh__baike_chinese_new_all__2025082500/', 0.2),
     ('obs://mb-datasets/text/bbh_improve_0829_v9/en__bbh_improve_0829_v9__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/bbh_improve_0829_v9/en__bbh_improve_0829_v9__2025082500/', 1.0),
     ('obs://mb-datasets/text/books1_long_context/en__books1_long_context__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/books1_long_context/en__books1_long_context__2025082000/', 0.2),
     ('obs://mb-datasets/text/books3_long_context/en__books3_long_context__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/books3_long_context/en__books3_long_context__2025082500/', 0.2),
     ('obs://mb-datasets/text/cc_hm_clean_pos_transformed/zh__cc_hm_clean_pos_transformed__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/cc_hm_clean_pos_transformed/zh__cc_hm_clean_pos_transformed__2025082500/',
      0.8), ('obs://mb-datasets/text/cc_math_transformed/en__cc_math_transformed__2025082000/',
             'obs://mb-datasets/old_version_sstable_250915/cc_math_transformed/en__cc_math_transformed__2025082000/',
             0.6), ('obs://mb-datasets/text/cci3_all/zh__cci3_all__2025082500/',
                    'obs://mb-datasets/old_version_sstable_250915/cci3_all/zh__cci3_all__2025082500/', 0.1),
     ('obs://mb-datasets/text/chinese fineweb edu v2/zh__chinese fineweb edu v2__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/chinese fineweb edu v2/zh__chinese fineweb edu v2__2025082000/',
      0.3), ('obs://mb-datasets/text/chinese-knowledge/zh__chinese-knowledge__2025082500/',
             'obs://mb-datasets/old_version_sstable_250915/chinese-knowledge/zh__chinese-knowledge__2025082500/', 1.0),
     ('obs://mb-datasets/text/chn_dic/zh__chn_dic__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/chn_dic/zh__chn_dic__2025082000/', 0.2),
     ('obs://mb-datasets/text/classical_chinese_full_new_0712/zh__classical_chinese_full_new_0712__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/classical_chinese_full_new_0712/zh__classical_chinese_full_new_0712__2025082500/',
      1.0), ('obs://mb-datasets/text/close_llm_instruct/en__close_llm_instruct__2025082000/',
             'obs://mb-datasets/old_version_sstable_250915/close_llm_instruct/en__close_llm_instruct__2025082000/',
             1.0), ('obs://mb-datasets/text/code_0/en__code_0__2025082500/',
                    'obs://mb-datasets/old_version_sstable_250915/code_0/en__code_0__2025082500/', 1.0),
     ('obs://mb-datasets/text/code_30w/en__code_30w__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/code_30w/en__code_30w__2025082500/', 1.0),
     ('obs://mb-datasets/text/code_dedup_new_rule_filtered_v1_transformed/en__code_dedup_new_rule_filtered_v1_transformed__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/code_dedup_new_rule_filtered_v1_transformed/en__code_dedup_new_rule_filtered_v1_transformed__2025082500/',
      0.5), ('obs://mb-datasets/text/code_feedback_241230/en__code_feedback_241230__2025082500/',
             'obs://mb-datasets/old_version_sstable_250915/code_feedback_241230/en__code_feedback_241230__2025082500/',
             1.0),
     ('obs://mb-datasets/text/code_python_to_testbook_250427/en__code_python_to_testbook_250427__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/code_python_to_testbook_250427/en__code_python_to_testbook_250427__2025082500/',
      1.0),
     ('obs://mb-datasets/text/code_to_strucutre_textbook_250407/en__code_to_strucutre_textbook_250407__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/code_to_strucutre_textbook_250407/en__code_to_strucutre_textbook_250407__2025082000/',
      0.6), ('obs://mb-datasets/text/congress/en__congress__2025090600/',
             'obs://mb-datasets/old_version_sstable_250915/congress/en__congress__2025090600/', 1.0),
     ('obs://mb-datasets/text/couplet_full_new_0712/zh__couplet_full_new_0712__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/couplet_full_new_0712/zh__couplet_full_new_0712__2025082500/', 1.0),
     ('obs://mb-datasets/text/csdn_parse_filter/zh__csdn_parse_filter__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/csdn_parse_filter/zh__csdn_parse_filter__2025082500/', 0.1),
     ('obs://mb-datasets/text/ctrip_trave_v1_clean/zh__ctrip_trave_v1_clean__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/ctrip_trave_v1_clean/zh__ctrip_trave_v1_clean__2025082500/', 0.1),
     ('obs://mb-datasets/text/cwp_clean_head_pos_transformed_new/zh__cwp_clean_head_pos_transformed_new__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/cwp_clean_head_pos_transformed_new/zh__cwp_clean_head_pos_transformed_new__2025082500/',
      0.8),
     ('obs://mb-datasets/text/cwp_clean_mid_pos_transformed_new/zh__cwp_clean_mid_pos_transformed_new__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/cwp_clean_mid_pos_transformed_new/zh__cwp_clean_mid_pos_transformed_new__2025082500/',
      0.8),
     ('obs://mb-datasets/text/cwp_clean_tail_pos_transformed_new/zh__cwp_clean_tail_pos_transformed_new__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/cwp_clean_tail_pos_transformed_new/zh__cwp_clean_tail_pos_transformed_new__2025082500/',
      0.8), ('obs://mb-datasets/text/dailyscript/en__dailyscript__2025090600/',
             'obs://mb-datasets/old_version_sstable_250915/dailyscript/en__dailyscript__2025090600/', 1.0),
     ('obs://mb-datasets/text/decay_en_if_081602_chatml/en__decay_en_if_081602_chatml__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/decay_en_if_081602_chatml/en__decay_en_if_081602_chatml__2025082500/',
      1.0), ('obs://mb-datasets/text/dffl_clean_sst/zh__dffl_clean_sst__2025082500/',
             'obs://mb-datasets/old_version_sstable_250915/dffl_clean_sst/zh__dffl_clean_sst__2025082500/', 0.2),
     ('obs://mb-datasets/text/dm_math_clean/en__dm_math_clean__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/dm_math_clean/en__dm_math_clean__2025082500/', 1.0),
     ('obs://mb-datasets/text/douban/zh__douban__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/douban/zh__douban__2025082500/', 0.1),
     ('obs://mb-datasets/text/ebook_ebooks_1_to_6/zh__ebook_ebooks_1_to_6__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/ebook_ebooks_1_to_6/zh__ebook_ebooks_1_to_6__2025082000/', 0.3),
     ('obs://mb-datasets/text/ebook_epubee/zh__ebook_epubee__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/ebook_epubee/zh__ebook_epubee__2025082000/', 0.3),
     ('obs://mb-datasets/text/ebook_kindle/zh__ebook_kindle__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/ebook_kindle/zh__ebook_kindle__2025082000/', 0.3),
     ('obs://mb-datasets/text/ebook_panda_reading/zh__ebook_panda_reading__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/ebook_panda_reading/zh__ebook_panda_reading__2025082000/', 0.3),
     ('obs://mb-datasets/text/ebook_sobooks/zh__ebook_sobooks__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/ebook_sobooks/zh__ebook_sobooks__2025082000/', 0.3),
     ('obs://mb-datasets/text/economic_information_daily_clean/zh__economic_information_daily_clean__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/economic_information_daily_clean/zh__economic_information_daily_clean__2025082500/',
      0.1), ('obs://mb-datasets/text/en.boke_kindle_mobi/en__en.boke_kindle_mobi__2025082000/',
             'obs://mb-datasets/old_version_sstable_250915/en.boke_kindle_mobi/en__en.boke_kindle_mobi__2025082000/',
             0.2), ('obs://mb-datasets/text/en.wikihow/en__en.wikihow__2025082500/',
                    'obs://mb-datasets/old_version_sstable_250915/en.wikihow/en__en.wikihow__2025082500/', 0.2),
     ('obs://mb-datasets/text/en_hqdata_v5/en__en_hqdata_v5__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/en_hqdata_v5/en__en_hqdata_v5__2025082500/', 0.2),
     ('obs://mb-datasets/text/eq_1103/zh__eq_1103__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/eq_1103/zh__eq_1103__2025082500/', 0.8),
     ('obs://mb-datasets/text/evol_code_clean_dedup_whoru/en__evol_code_clean_dedup_whoru__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/evol_code_clean_dedup_whoru/en__evol_code_clean_dedup_whoru__2025082500/',
      1.0), ('obs://mb-datasets/text/fbdata_v40/en__fbdata_v40__2025082500/',
             'obs://mb-datasets/old_version_sstable_250915/fbdata_v40/en__fbdata_v40__2025082500/', 1.0),
     ('obs://mb-datasets/text/feilu/zh__feilu__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/feilu/zh__feilu__2025082000/', 0.1),
     ('obs://mb-datasets/text/finemath-3plus/en__finemath-3plus__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/finemath-3plus/en__finemath-3plus__2025082500/', 0.6),
     ('obs://mb-datasets/text/finemath-4plus/en__finemath-4plus__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/finemath-4plus/en__finemath-4plus__2025082500/', 0.5),
     ('obs://mb-datasets/text/finemath_4plus_gen_grade_school_qa_dedup_250408/en__finemath_4plus_gen_grade_school_qa_dedup_250408__2025080100/',
      'obs://mb-datasets/old_version_sstable_250915/finemath_4plus_gen_grade_school_qa_dedup_250408/en__finemath_4plus_gen_grade_school_qa_dedup_250408__2025080100/',
      0.7),
     ('obs://mb-datasets/text/finemath_4plus_gen_middle_school_qa_dedup_250408/en__finemath_4plus_gen_middle_school_qa_dedup_250408__2025080100/',
      'obs://mb-datasets/old_version_sstable_250915/finemath_4plus_gen_middle_school_qa_dedup_250408/en__finemath_4plus_gen_middle_school_qa_dedup_250408__2025080100/',
      1.0),
     ('obs://mb-datasets/text/finemath_4plus_gen_mind_8_role_all/en__finemath_4plus_gen_mind_8_role_all__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/finemath_4plus_gen_mind_8_role_all/en__finemath_4plus_gen_mind_8_role_all__2025082500/',
      0.2), ('obs://mb-datasets/text/functioncalling_chatml/en__functioncalling_chatml__2025082500/',
             'obs://mb-datasets/old_version_sstable_250915/functioncalling_chatml/en__functioncalling_chatml__2025082500/',
             0.3), ('obs://mb-datasets/text/gaogov/en__gaogov__2025090600/',
                    'obs://mb-datasets/old_version_sstable_250915/gaogov/en__gaogov__2025090600/', 1.0),
     ('obs://mb-datasets/text/gaokao_essay/zh__gaokao_essay__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/gaokao_essay/zh__gaokao_essay__2025082500/', 1.0),
     ('obs://mb-datasets/text/gcode_0813/zh__gcode_0813__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/gcode_0813/zh__gcode_0813__2025082500/', 1.0),
     ('obs://mb-datasets/text/glaive/en__glaive__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/glaive/en__glaive__2025082500/', 1.0),
     ('obs://mb-datasets/text/gov_safety/zh__gov_safety__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/gov_safety/zh__gov_safety__2025082500/', 0.2),
     ('obs://mb-datasets/text/guowuyuan_qa/zh__guowuyuan_qa__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/guowuyuan_qa/zh__guowuyuan_qa__2025082500/', 1.0),
     ('obs://mb-datasets/text/hi138_data/zh__hi138_data__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/hi138_data/zh__hi138_data__2025082000/', 0.3),
     ('obs://mb-datasets/text/humanevallike_clean_dedup/none__humanevallike_clean_dedup__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/humanevallike_clean_dedup/none__humanevallike_clean_dedup__2025082500/',
      0.3),
     ('obs://mb-datasets/text/infinemath4plus_gen_textbook_part_250324/en__infinemath4plus_gen_textbook_part_250324__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/infinemath4plus_gen_textbook_part_250324/en__infinemath4plus_gen_textbook_part_250324__2025082500/',
      0.9), ('obs://mb-datasets/text/infiwebmath-3plus/en__infiwebmath-3plus__2025082500/',
             'obs://mb-datasets/old_version_sstable_250915/infiwebmath-3plus/en__infiwebmath-3plus__2025082500/', 0.6),
     ('obs://mb-datasets/text/infiwebmath-4plus/en__infiwebmath-4plus__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/infiwebmath-4plus/en__infiwebmath-4plus__2025082500/', 0.5),
     ('obs://mb-datasets/text/jianshu/zh__jianshu__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/jianshu/zh__jianshu__2025082500/', 0.1),
     ('obs://mb-datasets/text/jupyter_notebook_markdown_transformed/none__jupyter_notebook_markdown_transformed__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/jupyter_notebook_markdown_transformed/none__jupyter_notebook_markdown_transformed__2025082500/',
      0.5), ('obs://mb-datasets/text/k12_icl_no_sys/zh__k12_icl_no_sys__2025082500/',
             'obs://mb-datasets/old_version_sstable_250915/k12_icl_no_sys/zh__k12_icl_no_sys__2025082500/', 1.0),
     ('obs://mb-datasets/text/knowledge_rewrite_no_sys/zh__knowledge_rewrite_no_sys__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/knowledge_rewrite_no_sys/zh__knowledge_rewrite_no_sys__2025082500/',
      1.0), ('obs://mb-datasets/text/law_sst/zh__law_sst__2025082500/',
             'obs://mb-datasets/old_version_sstable_250915/law_sst/zh__law_sst__2025082500/', 0.2),
     ('obs://mb-datasets/text/law_sst_amk/zh__law_sst_amk__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/law_sst_amk/zh__law_sst_amk__2025082000/', 0.2),
     ('obs://mb-datasets/text/law_sst_cpws/zh__law_sst_cpws__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/law_sst_cpws/zh__law_sst_cpws__2025082500/', 0.2),
     ('obs://mb-datasets/text/law_sst_flqk/zh__law_sst_flqk__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/law_sst_flqk/zh__law_sst_flqk__2025082000/', 0.2),
     ('obs://mb-datasets/text/law_sst_flsy/zh__law_sst_flsy__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/law_sst_flsy/zh__law_sst_flsy__2025082500/', 0.2),
     ('obs://mb-datasets/text/law_sst_gjty/zh__law_sst_gjty__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/law_sst_gjty/zh__law_sst_gjty__2025082000/', 0.2),
     ('obs://mb-datasets/text/law_sst_lf/zh__law_sst_lf__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/law_sst_lf/zh__law_sst_lf__2025082000/', 0.2),
     ('obs://mb-datasets/text/law_sst_sf/zh__law_sst_sf__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/law_sst_sf/zh__law_sst_sf__2025082500/', 0.2),
     ('obs://mb-datasets/text/law_sst_twk/zh__law_sst_twk__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/law_sst_twk/zh__law_sst_twk__2025082000/', 0.2),
     ('obs://mb-datasets/text/law_sst_twsy/zh__law_sst_twsy__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/law_sst_twsy/zh__law_sst_twsy__2025082500/', 0.2),
     ('obs://mb-datasets/text/law_sst_xgk/zh__law_sst_xgk__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/law_sst_xgk/zh__law_sst_xgk__2025082000/', 0.2),
     ('obs://mb-datasets/text/law_sst_zyfl/zh__law_sst_zyfl__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/law_sst_zyfl/zh__law_sst_zyfl__2025082000/', 0.2),
     ('obs://mb-datasets/text/lccf_apps_ultracode_aug_data_0818/en__lccf_apps_ultracode_aug_data_0818__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/lccf_apps_ultracode_aug_data_0818/en__lccf_apps_ultracode_aug_data_0818__2025082500/',
      1.0), ('obs://mb-datasets/text/leetcode/zh__leetcode__2025080100/',
             'obs://mb-datasets/old_version_sstable_250915/leetcode/zh__leetcode__2025080100/', 1.0),
     ('obs://mb-datasets/text/leetcode_pass_code/zh__leetcode_pass_code__2025080100/',
      'obs://mb-datasets/old_version_sstable_250915/leetcode_pass_code/zh__leetcode_pass_code__2025080100/', 1.0),
     ('obs://mb-datasets/text/leetcode_pass_code_0125/zh__leetcode_pass_code_0125__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/leetcode_pass_code_0125/zh__leetcode_pass_code_0125__2025082500/',
      1.0), ('obs://mb-datasets/text/libretext/en__libretext__2025082500/',
             'obs://mb-datasets/old_version_sstable_250915/libretext/en__libretext__2025082500/', 1.0),
     ('obs://mb-datasets/text/logi_merge_no_sys/en__logi_merge_no_sys__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/logi_merge_no_sys/en__logi_merge_no_sys__2025082500/', 1.0),
     ('obs://mb-datasets/text/magpie/en__magpie__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/magpie/en__magpie__2025082500/', 1.0),
     ('obs://mb-datasets/text/magpie_gp/en__magpie_gp__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/magpie_gp/en__magpie_gp__2025082500/', 1.0),
     ('obs://mb-datasets/text/math_college_cnq_trans-from-enq/zh__math_college_cnq_trans-from-enq__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/math_college_cnq_trans-from-enq/zh__math_college_cnq_trans-from-enq__2025082500/',
      1.0), ('obs://mb-datasets/text/math_college_en/en__math_college_en__2025082500/',
             'obs://mb-datasets/old_version_sstable_250915/math_college_en/en__math_college_en__2025082500/', 1.0),
     ('obs://mb-datasets/text/math_k12_1103/zh__math_k12_1103__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/math_k12_1103/zh__math_k12_1103__2025082000/', 1.0),
     ('obs://mb-datasets/text/math_k12_cn/zh__math_k12_cn__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/math_k12_cn/zh__math_k12_cn__2025082500/', 1.0),
     ('obs://mb-datasets/text/math_k12_knowledge_wenku/zh__math_k12_knowledge_wenku__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/math_k12_knowledge_wenku/zh__math_k12_knowledge_wenku__2025082500/',
      1.0), ('obs://mb-datasets/text/math_k12_trans-to-en/en__math_k12_trans-to-en__2025082500/',
             'obs://mb-datasets/old_version_sstable_250915/math_k12_trans-to-en/en__math_k12_trans-to-en__2025082500/',
             1.0), ('obs://mb-datasets/text/math_k12en_numina_stepwise/en__math_k12en_numina_stepwise__2025082500/',
                    'obs://mb-datasets/old_version_sstable_250915/math_k12en_numina_stepwise/en__math_k12en_numina_stepwise__2025082500/',
                    1.0),
     ('obs://mb-datasets/text/math_k12en_numina_stepwise_error_recovery/en__math_k12en_numina_stepwise_error_recovery__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/math_k12en_numina_stepwise_error_recovery/en__math_k12en_numina_stepwise_error_recovery__2025082500/',
      1.0), ('obs://mb-datasets/text/math_merge_meta_instruct_plus/en__math_merge_meta_instruct_plus__2025082500/',
             'obs://mb-datasets/old_version_sstable_250915/math_merge_meta_instruct_plus/en__math_merge_meta_instruct_plus__2025082500/',
             1.0), ('obs://mb-datasets/text/math_numina/en__math_numina__2025082500/',
                    'obs://mb-datasets/old_version_sstable_250915/math_numina/en__math_numina__2025082500/', 1.0),
     ('obs://mb-datasets/text/math_primary_mwp/zh__math_primary_mwp__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/math_primary_mwp/zh__math_primary_mwp__2025082500/', 1.0),
     ('obs://mb-datasets/text/math_sft_gen_textbook_250304/en__math_sft_gen_textbook_250304__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/math_sft_gen_textbook_250304/en__math_sft_gen_textbook_250304__2025082500/',
      1.0), ('obs://mb-datasets/text/math_trans-from-en/zh__math_trans-from-en__2025082500/',
             'obs://mb-datasets/old_version_sstable_250915/math_trans-from-en/zh__math_trans-from-en__2025082500/',
             1.0), ('obs://mb-datasets/text/math_webinstructsub/en__math_webinstructsub__2025082500/',
                    'obs://mb-datasets/old_version_sstable_250915/math_webinstructsub/en__math_webinstructsub__2025082500/',
                    1.0), ('obs://mb-datasets/text/mb_zh_250113/zh__mb_zh_250113__2025082500/',
                           'obs://mb-datasets/old_version_sstable_250915/mb_zh_250113/zh__mb_zh_250113__2025082500/',
                           0.1), ('obs://mb-datasets/text/mbalib_data_clean/zh__mbalib_data_clean__2025082000/',
                                  'obs://mb-datasets/old_version_sstable_250915/mbalib_data_clean/zh__mbalib_data_clean__2025082000/',
                                  0.2),
     ('obs://mb-datasets/text/megamath_qa_qwen_25/en__megamath_qa_qwen_25__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/megamath_qa_qwen_25/en__megamath_qa_qwen_25__2025082000/', 0.7),
     ('obs://mb-datasets/text/megamath_text_code_block/en__megamath_text_code_block__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/megamath_text_code_block/en__megamath_text_code_block__2025082000/',
      0.3), ('obs://mb-datasets/text/megamath_translated_code/en__megamath_translated_code__2025082000/',
             'obs://mb-datasets/old_version_sstable_250915/megamath_translated_code/en__megamath_translated_code__2025082000/',
             0.6), ('obs://mb-datasets/text/megamath_web_pro/en__megamath_web_pro__2025082000/',
                    'obs://mb-datasets/old_version_sstable_250915/megamath_web_pro/en__megamath_web_pro__2025082000/',
                    1.0), ('obs://mb-datasets/text/moshadong/zh__moshadong__2025082000/',
                           'obs://mb-datasets/old_version_sstable_250915/moshadong/zh__moshadong__2025082000/', 0.2),
     ('obs://mb-datasets/text/mt_book_dic/multi__mt_book_dic__2025080100/',
      'obs://mb-datasets/old_version_sstable_250915/mt_book_dic/multi__mt_book_dic__2025080100/', 0.2),
     ('obs://mb-datasets/text/mt_lab/multi__mt_lab__2025080100/',
      'obs://mb-datasets/old_version_sstable_250915/mt_lab/multi__mt_lab__2025080100/', 0.2),
     ('obs://mb-datasets/text/mtbench_like_no_sys/en__mtbench_like_no_sys__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/mtbench_like_no_sys/en__mtbench_like_no_sys__2025082500/', 1.0),
     ('obs://mb-datasets/text/multifaceted_collection_sft/en__multifaceted_collection_sft__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/multifaceted_collection_sft/en__multifaceted_collection_sft__2025082500/',
      1.0), ('obs://mb-datasets/text/nemotron-medium-high/en__nemotron-medium-high__2025082500/',
             'obs://mb-datasets/old_version_sstable_250915/nemotron-medium-high/en__nemotron-medium-high__2025082500/',
             0.02), ('obs://mb-datasets/text/nllb/multi__nllb__2025080100/',
                     'obs://mb-datasets/old_version_sstable_250915/nllb/multi__nllb__2025080100/', 0.1),
     ('obs://mb-datasets/text/novel.17k/zh__novel.17k__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/novel.17k/zh__novel.17k__2025082000/', 0.1),
     ('obs://mb-datasets/text/novel.yd_baidu/zh__novel.yd_baidu__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/novel.yd_baidu/zh__novel.yd_baidu__2025082000/', 0.1),
     ('obs://mb-datasets/text/novel8080/zh__novel8080__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/novel8080/zh__novel8080__2025082000/', 0.1),
     ('obs://mb-datasets/text/num_format_task_0715_280k_wo_sys/zh__num_format_task_0715_280k_wo_sys__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/num_format_task_0715_280k_wo_sys/zh__num_format_task_0715_280k_wo_sys__2025082000/',
      1.0),
     ('obs://mb-datasets/text/opc_annealing_algorithmic_corpus/en__opc_annealing_algorithmic_corpus__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/opc_annealing_algorithmic_corpus/en__opc_annealing_algorithmic_corpus__2025082000/',
      0.6),
     ('obs://mb-datasets/text/opc_annealing_synthetic_code_snippet_corpus/en__opc_annealing_synthetic_code_snippet_corpus__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/opc_annealing_synthetic_code_snippet_corpus/en__opc_annealing_synthetic_code_snippet_corpus__2025082000/',
      1.0),
     ('obs://mb-datasets/text/opc_annealing_synthetic_qa_corpus/zh__opc_annealing_synthetic_qa_corpus__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/opc_annealing_synthetic_qa_corpus/zh__opc_annealing_synthetic_qa_corpus__2025082000/',
      0.8), ('obs://mb-datasets/text/opc_fineweb_math_corpus/en__opc_fineweb_math_corpus__2025082000/',
             'obs://mb-datasets/old_version_sstable_250915/opc_fineweb_math_corpus/en__opc_fineweb_math_corpus__2025082000/',
             1.0),
     ('obs://mb-datasets/text/openassistant_best_replies_train/en__openassistant_best_replies_train__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/openassistant_best_replies_train/en__openassistant_best_replies_train__2025082000/',
      1.0), ('obs://mb-datasets/text/openhermes2_5/en__openhermes2_5__2025082000/',
             'obs://mb-datasets/old_version_sstable_250915/openhermes2_5/en__openhermes2_5__2025082000/', 1.0),
     ('obs://mb-datasets/text/parallel_functioncall/zh__parallel_functioncall__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/parallel_functioncall/zh__parallel_functioncall__2025082500/', 1.0),
     ('obs://mb-datasets/text/peS2o/en__peS2o__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/peS2o/en__peS2o__2025082000/', 0.3),
     ('obs://mb-datasets/text/people_daily/zh__people_daily__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/people_daily/zh__people_daily__2025082000/', 0.1),
     ('obs://mb-datasets/text/proof-pile-algebraic-stack_transformed/en__proof-pile-algebraic-stack_transformed__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/proof-pile-algebraic-stack_transformed/en__proof-pile-algebraic-stack_transformed__2025082500/',
      0.7), ('obs://mb-datasets/text/proof-pile-arxiv_transformed/en__proof-pile-arxiv_transformed__2025082500/',
             'obs://mb-datasets/old_version_sstable_250915/proof-pile-arxiv_transformed/en__proof-pile-arxiv_transformed__2025082500/',
             0.7),
     ('obs://mb-datasets/text/proof-pile-open-web-math_transformed/en__proof-pile-open-web-math_transformed__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/proof-pile-open-web-math_transformed/en__proof-pile-open-web-math_transformed__2025082500/',
      0.6), ('obs://mb-datasets/text/qikan_chinese/zh__qikan_chinese__2025082000/',
             'obs://mb-datasets/old_version_sstable_250915/qikan_chinese/zh__qikan_chinese__2025082000/', 0.5),
     ('obs://mb-datasets/text/qinkan_long_context/zh__qinkan_long_context__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/qinkan_long_context/zh__qinkan_long_context__2025082000/', 0.1),
     ('obs://mb-datasets/text/reasoning_0730/en__reasoning_0730__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/reasoning_0730/en__reasoning_0730__2025082000/', 1.0),
     ('obs://mb-datasets/text/refined-anime-text/zh__refined-anime-text__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/refined-anime-text/zh__refined-anime-text__2025082000/', 0.1),
     ('obs://mb-datasets/text/sft_old_rewrite_no_sys/zh__sft_old_rewrite_no_sys__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/sft_old_rewrite_no_sys/zh__sft_old_rewrite_no_sys__2025082500/',
      1.0), ('obs://mb-datasets/text/shenqing_news_20231109/zh__shenqing_news_20231109__2025082000/',
             'obs://mb-datasets/old_version_sstable_250915/shenqing_news_20231109/zh__shenqing_news_20231109__2025082000/',
             0.2), ('obs://mb-datasets/text/shenqing_xingguang_new/zh__shenqing_xingguang_new__2025082000/',
                    'obs://mb-datasets/old_version_sstable_250915/shenqing_xingguang_new/zh__shenqing_xingguang_new__2025082000/',
                    0.2), ('obs://mb-datasets/text/shenqing_yuqing/zh__shenqing_yuqing__2025082000/',
                           'obs://mb-datasets/old_version_sstable_250915/shenqing_yuqing/zh__shenqing_yuqing__2025082000/',
                           0.2), ('obs://mb-datasets/text/shenqingbaike_sq/zh__shenqingbaike_sq__2025082500/',
                                  'obs://mb-datasets/old_version_sstable_250915/shenqingbaike_sq/zh__shenqingbaike_sq__2025082500/',
                                  0.2), ('obs://mb-datasets/text/size_math/none__size_math__2025082500/',
                                         'obs://mb-datasets/old_version_sstable_250915/size_math/none__size_math__2025082500/',
                                         1.0), ('obs://mb-datasets/text/souyun/zh__souyun__2025082000/',
                                                'obs://mb-datasets/old_version_sstable_250915/souyun/zh__souyun__2025082000/',
                                                1.0),
     ('obs://mb-datasets/text/stack_exchange_qa/en__stack_exchange_qa__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/stack_exchange_qa/en__stack_exchange_qa__2025082500/', 0.1),
     ('obs://mb-datasets/text/stack_overflow/en__stack_overflow__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/stack_overflow/en__stack_overflow__2025082500/', 0.1),
     ('obs://mb-datasets/text/stack_v2_fim/none__stack_v2_fim__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/stack_v2_fim/none__stack_v2_fim__2025082500/', 0.1),
     ('obs://mb-datasets/text/stack_v2_full_not_code_transformed/zh__stack_v2_full_not_code_transformed__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/stack_v2_full_not_code_transformed/zh__stack_v2_full_not_code_transformed__2025082500/',
      0.6), ('obs://mb-datasets/text/starting_point_chinese_website/zh__starting_point_chinese_website__2025082000/',
             'obs://mb-datasets/old_version_sstable_250915/starting_point_chinese_website/zh__starting_point_chinese_website__2025082000/',
             0.1), ('obs://mb-datasets/text/task_oriented/zh__task_oriented__2025082500/',
                    'obs://mb-datasets/old_version_sstable_250915/task_oriented/zh__task_oriented__2025082500/', 1.0),
     ('obs://mb-datasets/text/tele/zh__tele__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/tele/zh__tele__2025082500/', 0.1),
     ('obs://mb-datasets/text/the_stack_v2_rule_filtered_transformed/en__the_stack_v2_rule_filtered_transformed__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/the_stack_v2_rule_filtered_transformed/en__the_stack_v2_rule_filtered_transformed__2025082500/',
      0.8), ('obs://mb-datasets/text/tianya_forum/zh__tianya_forum__2025082500/',
             'obs://mb-datasets/old_version_sstable_250915/tianya_forum/zh__tianya_forum__2025082500/', 0.1),
     ('obs://mb-datasets/text/tinycode_sciphi/en__tinycode_sciphi__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/tinycode_sciphi/en__tinycode_sciphi__2025082500/', 0.1),
     ('obs://mb-datasets/text/ultrainteract_0813/en__ultrainteract_0813__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/ultrainteract_0813/en__ultrainteract_0813__2025082500/', 1.0),
     ('obs://mb-datasets/text/ultratextbook/en__ultratextbook__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/ultratextbook/en__ultratextbook__2025082500/', 0.3),
     ('obs://mb-datasets/text/waijiaobu_qa/zh__waijiaobu_qa__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/waijiaobu_qa/zh__waijiaobu_qa__2025082500/', 1.0),
     ('obs://mb-datasets/text/wanjuan_chinese_web/zh__wanjuan_chinese_web__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/wanjuan_chinese_web/zh__wanjuan_chinese_web__2025082500/', 0.05),
     ('obs://mb-datasets/text/web_code_en_dedup_new/en__web_code_en_dedup_new__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/web_code_en_dedup_new/en__web_code_en_dedup_new__2025082500/', 1.0),
     ('obs://mb-datasets/text/web_code_zh_dedup_new/zh__web_code_zh_dedup_new__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/web_code_zh_dedup_new/zh__web_code_zh_dedup_new__2025082500/', 1.0),
     ('obs://mb-datasets/text/wechat_account/zh__wechat_account__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/wechat_account/zh__wechat_account__2025082500/', 0.5),
     ('obs://mb-datasets/text/what_is_worth_buying/zh__what_is_worth_buying__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/what_is_worth_buying/zh__what_is_worth_buying__2025082000/', 0.1),
     ('obs://mb-datasets/text/wikipedia/en__wikipedia__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/wikipedia/en__wikipedia__2025082500/', 0.2),
     ('obs://mb-datasets/text/wordproblem_1013/zh__wordproblem_1013__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/wordproblem_1013/zh__wordproblem_1013__2025082500/', 1.0),
     ('obs://mb-datasets/text/xiaohongshu/zh__xiaohongshu__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/xiaohongshu/zh__xiaohongshu__2025082500/', 0.1),
     ('obs://mb-datasets/text/xiaoxiang_academy_data/zh__xiaoxiang_academy_data__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/xiaoxiang_academy_data/zh__xiaoxiang_academy_data__2025082000/',
      0.1), ('obs://mb-datasets/text/xlam_function_calling_60k/en__xlam_function_calling_60k__2025082500/',
             'obs://mb-datasets/old_version_sstable_250915/xlam_function_calling_60k/en__xlam_function_calling_60k__2025082500/',
             1.0), ('obs://mb-datasets/text/xuexiqiangguo/zh__xuexiqiangguo__2025082000/',
                    'obs://mb-datasets/old_version_sstable_250915/xuexiqiangguo/zh__xuexiqiangguo__2025082000/', 0.2),
     ('obs://mb-datasets/text/yayi_pos_transformed/zh__yayi_pos_transformed__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/yayi_pos_transformed/zh__yayi_pos_transformed__2025082500/', 0.8),
     ('obs://mb-datasets/text/zh.baijiahao/zh__zh.baijiahao__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/zh.baijiahao/zh__zh.baijiahao__2025082500/', 0.1),
     ('obs://mb-datasets/text/zh.boke_kindle_epub/zh__zh.boke_kindle_epub__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/zh.boke_kindle_epub/zh__zh.boke_kindle_epub__2025082500/', 0.3),
     ('obs://mb-datasets/text/zh.boke_kindle_mobi/zh__zh.boke_kindle_mobi__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/zh.boke_kindle_mobi/zh__zh.boke_kindle_mobi__2025082000/', 0.3),
     ('obs://mb-datasets/text/zh.chineseall_epub/zh__zh.chineseall_epub__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/zh.chineseall_epub/zh__zh.chineseall_epub__2025082000/', 0.5),
     ('obs://mb-datasets/text/zh.ebook_wode/zh__zh.ebook_wode__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/zh.ebook_wode/zh__zh.ebook_wode__2025082000/', 0.3),
     ('obs://mb-datasets/text/zh.epubee/zh__zh.epubee__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/zh.epubee/zh__zh.epubee__2025082500/', 0.5),
     ('obs://mb-datasets/text/zh.kara_jinjiang_literature_city/zh__zh.kara_jinjiang_literature_city__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/zh.kara_jinjiang_literature_city/zh__zh.kara_jinjiang_literature_city__2025082000/',
      0.1), ('obs://mb-datasets/text/zh.linux_cn/zh__zh.linux_cn__2025082500/',
             'obs://mb-datasets/old_version_sstable_250915/zh.linux_cn/zh__zh.linux_cn__2025082500/', 0.1),
     ('obs://mb-datasets/text/zh.magazine.1/zh__zh.magazine.1__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/zh.magazine.1/zh__zh.magazine.1__2025082000/', 0.5),
     ('obs://mb-datasets/text/zh.novel.c/zh__zh.novel.c__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/zh.novel.c/zh__zh.novel.c__2025082000/', 0.1),
     ('obs://mb-datasets/text/zh.novel.ijjxsw/zh__zh.novel.ijjxsw__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/zh.novel.ijjxsw/zh__zh.novel.ijjxsw__2025082000/', 0.1),
     ('obs://mb-datasets/text/zh.novel.pan_baidu/zh__zh.novel.pan_baidu__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/zh.novel.pan_baidu/zh__zh.novel.pan_baidu__2025082000/', 0.1),
     ('obs://mb-datasets/text/zh.novel.pan_jinjiang/zh__zh.novel.pan_jinjiang__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/zh.novel.pan_jinjiang/zh__zh.novel.pan_jinjiang__2025082000/', 0.1),
     ('obs://mb-datasets/text/zh.novel.taobao/zh__zh.novel.taobao__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/zh.novel.taobao/zh__zh.novel.taobao__2025082000/', 0.1),
     ('obs://mb-datasets/text/zh.novel.zzdxss/zh__zh.novel.zzdxss__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/zh.novel.zzdxss/zh__zh.novel.zzdxss__2025082000/', 0.1),
     ('obs://mb-datasets/text/zh.q4_news_0_10/zh__zh.q4_news_0_10__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/zh.q4_news_0_10/zh__zh.q4_news_0_10__2025082500/', 0.1),
     ('obs://mb-datasets/text/zh.q4_news_10_100/zh__zh.q4_news_10_100__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/zh.q4_news_10_100/zh__zh.q4_news_10_100__2025082500/', 0.1),
     ('obs://mb-datasets/text/zh.q4_news_gt_100/zh__zh.q4_news_gt_100__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/zh.q4_news_gt_100/zh__zh.q4_news_gt_100__2025082500/', 0.1),
     ('obs://mb-datasets/text/zh.sj_txt/zh__zh.sj_txt__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/zh.sj_txt/zh__zh.sj_txt__2025082000/', 0.5),
     ('obs://mb-datasets/text/zh.weibo_0320/zh__zh.weibo_0320__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/zh.weibo_0320/zh__zh.weibo_0320__2025082500/', 0.1),
     ('obs://mb-datasets/text/zh.wikihow/zh__zh.wikihow__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/zh.wikihow/zh__zh.wikihow__2025082500/', 0.1),
     ('obs://mb-datasets/text/zh.yayi/zh__zh.yayi__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/zh.yayi/zh__zh.yayi__2025082000/', 0.1),
     ('obs://mb-datasets/text/zh.zhihu-answer_0321/zh__zh.zhihu-answer_0321__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/zh.zhihu-answer_0321/zh__zh.zhihu-answer_0321__2025082500/', 0.1),
     ('obs://mb-datasets/text/zh.zhihu-article/zh__zh.zhihu-article__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/zh.zhihu-article/zh__zh.zhihu-article__2025082500/', 0.1),
     ('obs://mb-datasets/text/zh.zhihu_0320/zh__zh.zhihu_0320__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/zh.zhihu_0320/zh__zh.zhihu_0320__2025082000/', 0.1),
     ('obs://mb-datasets/text/zh.zp_ebook/zh__zh.zp_ebook__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/zh.zp_ebook/zh__zh.zp_ebook__2025082000/', 0.5),
     ('obs://mb-datasets/text/zh_cc_hm/zh__zh_cc_hm__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/zh_cc_hm/zh__zh_cc_hm__2025082000/', 0.1),
     ('obs://mb-datasets/text/zh_common_crawl_pos_transformed/zh__zh_common_crawl_pos_transformed__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/zh_common_crawl_pos_transformed/zh__zh_common_crawl_pos_transformed__2025082500/',
      0.8), ('obs://mb-datasets/text/zh_cwp_head_transformed/zh__zh_cwp_head_transformed__2025082500/',
             'obs://mb-datasets/old_version_sstable_250915/zh_cwp_head_transformed/zh__zh_cwp_head_transformed__2025082500/',
             0.05), ('obs://mb-datasets/text/zh_cwp_mid_transformed/zh__zh_cwp_mid_transformed__2025082500/',
                     'obs://mb-datasets/old_version_sstable_250915/zh_cwp_mid_transformed/zh__zh_cwp_mid_transformed__2025082500/',
                     0.05), ('obs://mb-datasets/text/zh_hqdata_v3/zh__zh_hqdata_v3__2025082500/',
                             'obs://mb-datasets/old_version_sstable_250915/zh_hqdata_v3/zh__zh_hqdata_v3__2025082500/',
                             0.6),
     ('obs://mb-datasets/text/zh_kara_netease_cloud_reading/zh__zh_kara_netease_cloud_reading__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/zh_kara_netease_cloud_reading/zh__zh_kara_netease_cloud_reading__2025082000/',
      0.1), ('obs://mb-datasets/text/zhengfu_qa/zh__zhengfu_qa__2025082500/',
             'obs://mb-datasets/old_version_sstable_250915/zhengfu_qa/zh__zhengfu_qa__2025082500/', 1.0),
     ('obs://mb-datasets/text/zhihu_qa/zh__zhihu_qa__2025082500/',
      'obs://mb-datasets/old_version_sstable_250915/zhihu_qa/zh__zhihu_qa__2025082500/', 0.1),
     ('obs://mb-datasets/text/zongheng_chinese_network/zh__zongheng_chinese_network__2025082000/',
      'obs://mb-datasets/old_version_sstable_250915/zongheng_chinese_network/zh__zongheng_chinese_network__2025082000/',
      0.1), ('obs://mb-datasets/text/zxcs_long_context/zh__zxcs_long_context__2025082500/',
             'obs://mb-datasets/old_version_sstable_250915/zxcs_long_context/zh__zxcs_long_context__2025082500/', 0.1)]

    # tasks = []
    # task_str = """proof-pile-algebraic-stack_transformed,0.7,obs://mb-datasets/text/proof-pile-algebraic-stack_transformed/en__proof-pile-algebraic-stack_transformed__2025082500/"""
    # for line in task_str.splitlines():
    #     dataset_name, threshold, input_dir = line.split(",")
    #     if float(threshold) > 0:
    #         tasks.append((input_dir, input_dir.replace("obs://mb-datasets/text/", "obs://mb-datasets/sstable/"), float(threshold)))
    # for i in tasks:
    #     print(f"ll /paratera_ceph/user/xueshuaikang/sstable_250915/{i[0]}")
    # return
    # #     print(i[0], i[1], i[2])

    # 多进程处理tasks列表里的任务
    if args.how == "run":
        logger.info(f"进程数量: {args.num_processes}, 总任务数量: {len(tasks)}")
        multi_tasks = [i for i in tasks for _ in range(args.num_processes)]
        with multiprocessing.Pool(args.num_processes) as pool:
            pool.starmap(process_task, multi_tasks)
    elif args.how == "reset":
        for input_dir, output_dir, threshold in tasks:
            processor = Parquet2SstableProcessor(input_dir, output_dir, threshold)
            files = processor.get_all_input_parts()
            for file in files:
                processor.unlock_input_part(file)
    elif args.how == "stat":
        import tqdm

        hw_bj_redis_cli.delete("parquet2sstable_init")
        for input_dir, output_dir, threshold in tasks:
            processor = Parquet2SstableProcessor(input_dir, output_dir, threshold)
            files = processor.get_all_input_parts()
            running_num = 0
            done_num = 0
            not_deal = 0
            for file in tqdm.tqdm(files):
                cur_res = processor.get_input_part_redis_status(file)
                if cur_res and cur_res.decode() == "RUNNING":
                    # processor.unlock_input_part(file)
                    running_num += 1
                if cur_res and cur_res.decode() == "DONE":
                    done_num += 1
                if not cur_res:
                    not_deal += 1
            logger.info(
                f"{input_dir}, 已经处理 {done_num}/{len(files)}, 共 {running_num} 个文件正在运行, 未处理 {not_deal} 个文件, 加起来{done_num + running_num + not_deal}/{len(files)}"
            )


if __name__ == "__main__":
    main()


