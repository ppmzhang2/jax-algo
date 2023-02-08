import logging
import os

from sqlalchemy import Column
from sqlalchemy import Table
from sqlalchemy import create_engine
from sqlalchemy import func
from sqlalchemy.engine.cursor import CursorResult
from sqlalchemy.engine.row import Row
from sqlalchemy.sql import select
from sqlalchemy.sql import text

from jaxalgo import cfg
from jaxalgo.datasets.coco._tables import boxes
from jaxalgo.datasets.coco._tables import cates
from jaxalgo.datasets.coco._tables import images
from jaxalgo.datasets.coco._tables import labels

LOGGER = logging.getLogger(__name__)

MAX_REC = 10000000
SQLITE_TRAIN = os.path.join(cfg.DATADIR, "coco_annot_train.db")
SQLITE_TEST = os.path.join(cfg.DATADIR, "coco_annot_test.db")

# format COCO box as a relative representation
# COCO box format:
#   - bbox1, bbox2: the upper-left coordinates of the bounding box
#   - bbox3, bbox4: the width and height of the bounding box
# new format (all in range [0, 1]):
#   - x, y: coordinates of the bounding box center
#   - w, h: width and height of the bounding box
FORMAT_QUERY = """
SELECT box.boxid
     , box.imageid
     , box.cateid
     , box.bbox1 / img.width + 0.5 * (box.bbox3 / img.width)    AS x
     , box.bbox2 / img.height + 0.5 * (box.bbox4 / img.height)  AS y
     , box.bbox3 / img.width                                    AS w
     , box.bbox4 / img.height                                   AS h
     , cat.name                                                 AS cate_name
     , img.name                                                 AS image_name
  FROM f_box AS box
 INNER
  JOIN d_cate AS cat
    ON box.cateid = cat.cateid
 INNER
  JOIN f_image AS img
    ON box.imageid = img.imageid
"""

_TABLES = (boxes, cates, images, labels)


class Dao:

    __slots__ = ["_engine"]

    def __init__(self, train: bool = True):
        sqlite_path = SQLITE_TRAIN if train else SQLITE_TEST
        self._engine = create_engine(f"sqlite:///{sqlite_path}")

    def create_all(self) -> None:
        for table in reversed(_TABLES):
            table.create(bind=self._engine, checkfirst=True)

    def drop_all(self) -> None:
        """Drop all tables defined in `redshift.tables`.

        there's no native `DROP TABLE ... CASCADE ...` method and tables should
        be dropped from the leaves of the dependency tree back to the root
        """
        for table in _TABLES:
            table.drop(bind=self._engine, checkfirst=True)

    def _exec(self, stmt: str, *args, **kwargs) -> CursorResult:
        with self._engine.begin() as conn:
            res = conn.execute(stmt, *args, **kwargs)
        return res

    def _count(self, column: Column) -> int:
        stmt = select(func.count(column))
        res = self._exec(stmt)
        return res.first()[0]

    def count_box(self) -> int:
        return self._count(boxes.c.boxid)

    def count_cate(self) -> int:
        return self._count(cates.c.cateid)

    def count_image(self) -> int:
        return self._count(images.c.imageid)

    def _lookup(
        self,
        table: Table,
        column: Column,
        key: str | int | float,
    ) -> Row | None:
        stmt = select(table).where(column == key)
        return self._exec(stmt).first()

    def lookup_image_rowid(self, rowid: int) -> Row | None:
        stmt = select(images).where(text(f"rowid = {rowid}"))
        return self._exec(stmt).first()

    def lookup_image_id(self, image_id: int) -> Row | None:
        return self._lookup(images, images.c.imageid, image_id)

    def lookup_image_name(self, name: str) -> Row | None:
        return self._lookup(images, images.c.name, name)

    def create_label(self) -> int:
        stmt_trunc = text(f"DELETE FROM {labels.name};")
        stmt_ins = text(f"INSERT INTO {labels.name} {FORMAT_QUERY};")
        rows_del = self._exec(stmt_trunc).rowcount
        LOGGER.debug(f"{rows_del} records deleted.")
        return self._exec(stmt_ins).rowcount

    def labels_by_img_id(self, image_id: int) -> list[Row]:
        stmt = text(f"SELECT * FROM {labels.name} "
                    f"WHERE imageid = {image_id};")
        return self._exec(stmt).all()

    def labels_by_img_name(self, image_name: str) -> list[Row]:
        stmt = text(f"SELECT * FROM {labels.name} "
                    f"WHERE image_name = '{image_name}';")
        return self._exec(stmt).all()

    def all_labels(self, limit: int = MAX_REC) -> list[Row]:
        stmt = text(f"SELECT * FROM {labels.name} "
                    f"LIMIT {limit};")
        return self._exec(stmt).all()

    def categories(self) -> dict[int, str]:
        stmt = select(cates)
        rows = self._exec(stmt).all()
        return {r.cateid: r.name for r in rows}


dao_train = Dao(train=True)
dao_test = Dao(train=False)
