import csv
import logging
import os
import subprocess
from collections.abc import Callable

from jaxalgo.datasets.coco._dao import SQLITE_TEST
from jaxalgo.datasets.coco._dao import SQLITE_TRAIN
from jaxalgo.datasets.coco._dao import dao_test
from jaxalgo.datasets.coco._dao import dao_train

LOGGER = logging.getLogger(__name__)

__all__ = ["CocoAnnotation"]


class CocoAnnotation:

    @staticmethod
    def _trans_imagetag_dict(
        dc: dict,
        folder: str,
    ) -> dict[str, str | int | float]:
        return {
            "imageid": dc["id"],
            "name": dc["file_name"],
            "height": dc["height"],
            "width": dc["width"],
            "url": dc["coco_url"],
            "path": os.path.abspath(os.path.join(folder, dc["file_name"])),
        }

    @staticmethod
    def _trans_cate_dict(dc: dict) -> dict[str, int | str]:
        return {
            "cateid": dc["id"],
            "name": dc["name"],
        }

    @staticmethod
    def _trans_box_dict(dc: dict) -> dict[str, int | float]:
        return {
            "boxid": dc["id"],
            "imageid": dc["image_id"],
            "cateid": dc["category_id"],
            "box1": dc["bbox"][0],
            "box2": dc["bbox"][1],
            "box3": dc["bbox"][2],
            "box4": dc["bbox"][3],
        }

    @staticmethod
    def _dicts2csv(
        f: Callable[[dict], dict],
        seq: list[dict],
        csvpath: str,
    ) -> None:
        seq_ = list(map(f, seq))
        headers = seq_[0].keys()

        with open(csvpath, "w", encoding="utf8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(seq_)

    @classmethod
    def imgtag2csv(cls, seq: list[dict], csvpath: str, folder: str) -> None:

        def helper(dc: dict):
            return cls._trans_imagetag_dict(dc, folder)

        return cls._dicts2csv(helper, seq, csvpath)

    @classmethod
    def cate2csv(cls, seq: list[dict], csvpath: str) -> None:
        return cls._dicts2csv(cls._trans_cate_dict, seq, csvpath)

    @classmethod
    def box2csv(cls, seq: list[dict], csvpath: str) -> None:
        return cls._dicts2csv(cls._trans_box_dict, seq, csvpath)

    @staticmethod
    def db_reset(train: bool) -> None:
        dao = dao_train if train else dao_test
        dao.drop_all()
        dao.create_all()

    @staticmethod
    def create_labels(train: bool) -> None:
        dao = dao_train if train else dao_test
        dao.create_label()

    @staticmethod
    def load_annot_csv(
        imgtag_csv: str,
        cate_csv: str,
        box_csv: str,
        train: bool,
    ) -> None:
        """Load COCO annotation CSV files."""
        cmd_img = f".import --csv --skip 1 {imgtag_csv} f_image"
        cmd_cate = f".import --csv --skip 1 {cate_csv} d_cate"
        cmd_box = f".import --csv --skip 1 {box_csv} f_box"
        cmd = f"{cmd_img}\n{cmd_cate}\n{cmd_box}"
        sqlite_path = SQLITE_TRAIN if train else SQLITE_TEST
        sp = subprocess.Popen(
            ["sqlite3", sqlite_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out, err = sp.communicate(input=bytes(cmd, encoding="utf8"))
        rc = sp.wait()
        LOGGER.info(f"return code = {rc} out = {out}; err = {err}")
